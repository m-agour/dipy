"""GPU compute-shader baking of Hermite LUT for SH billboard glyphs.

This module provides :func:`populate_hermite_lut_cube_gpu`, which replaces
the slow CPU path ``_populate_hermite_lut_cube_cpu_chunked`` by running
two compute passes on the GPU:

  1. **Matmul pass** – evaluates ``coeffs[G,C] @ basis[D,C]^T`` to produce
     an intermediate ``values[G, 6×S²]`` buffer (S = internal grid size).
  2. **Finite-diff pass** – extracts the padded output grid and computes
     4th-order finite differences for ``∂u``, ``∂v``, ``∂²uv``, writing
     ``vec4(value, du, dv, dudv)`` into the final Hermite LUT buffer.

The SH basis matrix is precomputed on CPU (small – depends only on
``lut_res`` and ``l_max``) and uploaded once.  All heavy per-glyph work
runs on the GPU.

Falls back to CPU path if wgpu device is unavailable or dispatch fails.
"""

from __future__ import annotations

import numpy as np
import wgpu

from fury.utils import create_sh_basis_matrix


_MATMUL_WGSL = """
struct Params {
    glyph_count:  u32,
    total_dirs:   u32,
    n_coeffs:     u32,
    _pad:         u32,
};

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       coeffs:  array<f32>;
@group(0) @binding(2) var<storage, read>       basis:   array<f32>;
@group(0) @binding(3) var<storage, read_write> values:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.glyph_count * params.total_dirs;
    if (idx >= total) { return; }

    let g = idx / params.total_dirs;
    let d = idx % params.total_dirs;
    let C = params.n_coeffs;

    var acc: f32 = 0.0;
    let c_off = g * C;
    let b_off = d * C;
    for (var c: u32 = 0u; c < C; c = c + 1u) {
        acc += coeffs[c_off + c] * basis[b_off + c];
    }
    values[idx] = acc;
}
"""

_FD_WGSL = """
struct Params {
    glyph_count:    u32,
    size_internal:  u32,
    out_size:       u32,
    start:          u32,
    total_texels:   u32,
    _pad1:          u32,
    _pad2:          u32,
    _pad3:          u32,
};

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       values:  array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_texels) { return; }

    let S  = params.out_size;
    let Si = params.size_internal;
    let face_texels = S * S;

    let g    = idx / (6u * face_texels);
    let rem  = idx % (6u * face_texels);
    let face = rem / face_texels;
    let texel = rem % face_texels;
    let ou   = texel / S;
    let ov   = texel % S;

    let iu = ou + params.start;
    let iv = ov + params.start;

    let dirs_per_face = Si * Si;
    let base = g * 6u * dirs_per_face + face * dirs_per_face;

    let c1: f32 = 0.6666666666666666;
    let c2: f32 = -0.0833333333333333;

    let val = values[base + iu * Si + iv];

    let du = c1 * (values[base + iu * Si + (iv + 1u)]
                  - values[base + iu * Si + (iv - 1u)])
           + c2 * (values[base + iu * Si + (iv + 2u)]
                  - values[base + iu * Si + (iv - 2u)]);

    let dv = c1 * (values[base + (iu + 1u) * Si + iv]
                  - values[base + (iu - 1u) * Si + iv])
           + c2 * (values[base + (iu + 2u) * Si + iv]
                  - values[base + (iu - 2u) * Si + iv]);

    let du_p1 = c1 * (values[base + (iu + 1u) * Si + (iv + 1u)]
                     - values[base + (iu + 1u) * Si + (iv - 1u)])
              + c2 * (values[base + (iu + 1u) * Si + (iv + 2u)]
                     - values[base + (iu + 1u) * Si + (iv - 2u)]);
    let du_m1 = c1 * (values[base + (iu - 1u) * Si + (iv + 1u)]
                     - values[base + (iu - 1u) * Si + (iv - 1u)])
              + c2 * (values[base + (iu - 1u) * Si + (iv + 2u)]
                     - values[base + (iu - 1u) * Si + (iv - 2u)]);
    let du_p2 = c1 * (values[base + (iu + 2u) * Si + (iv + 1u)]
                     - values[base + (iu + 2u) * Si + (iv - 1u)])
              + c2 * (values[base + (iu + 2u) * Si + (iv + 2u)]
                     - values[base + (iu + 2u) * Si + (iv - 2u)]);
    let du_m2 = c1 * (values[base + (iu - 2u) * Si + (iv + 1u)]
                     - values[base + (iu - 2u) * Si + (iv - 1u)])
              + c2 * (values[base + (iu - 2u) * Si + (iv + 2u)]
                     - values[base + (iu - 2u) * Si + (iv - 2u)]);

    let dudv = c1 * (du_p1 - du_m1) + c2 * (du_p2 - du_m2);

    output[idx] = vec4<f32>(val, du, dv, dudv);
}
"""


_GPU_DEVICE_CACHE = {}
_BASIS_CACHE = {}


def _get_wgpu_device():
    """Get or create a wgpu device for compute (cached)."""
    if "device" in _GPU_DEVICE_CACHE:
        return _GPU_DEVICE_CACHE["device"]
    device = None
    try:
        import pygfx as gfx
        shared = gfx.renderers.wgpu.Shared.get_instance()
        device = shared.device
    except Exception:
        pass
    if device is None:
        try:
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            device = adapter.request_device_sync()
        except Exception:
            pass
    _GPU_DEVICE_CACHE["device"] = device
    return device


_PIPELINE_CACHE = {}


def _get_pipelines(device):
    """Compile and cache compute pipelines."""
    if "matmul" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE["matmul"], _PIPELINE_CACHE["fd"]
    matmul_module = device.create_shader_module(code=_MATMUL_WGSL)
    matmul_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": matmul_module, "entry_point": "main"},
    )
    fd_module = device.create_shader_module(code=_FD_WGSL)
    fd_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute={"module": fd_module, "entry_point": "main"},
    )
    _PIPELINE_CACHE["matmul"] = matmul_pipeline
    _PIPELINE_CACHE["fd"] = fd_pipeline
    return matmul_pipeline, fd_pipeline


def _build_basis_matrices_flat(l_max, lut_res):
    """Build the 6-face basis matrix on CPU and return flat f32 array.

    Results are cached by ``(l_max, lut_res)``.

    Returns
    -------
    basis_flat : ndarray, shape (6 * Si * Si * n_coeffs,), dtype float32
        Basis matrix flattened in row-major order:
        ``basis[face * Si*Si + row*Si + col, coeff]``.
    size_internal : int
    """
    N = lut_res
    g_internal = 3
    size_internal = N + 2 * g_internal

    step = 2.0 / (N - 1)
    px = np.arange(size_internal, dtype=np.float32)
    py = np.arange(size_internal, dtype=np.float32)
    u_vals = -1.0 + (px - g_internal) * step
    v_vals = -1.0 + (py - g_internal) * step
    uu, vv = np.meshgrid(u_vals, v_vals)
    uu = uu.flatten()
    vv = vv.flatten()

    all_dirs = []
    for face in range(6):
        if face == 0:
            x, y, z = np.ones_like(uu), -vv, -uu
        elif face == 1:
            x, y, z = -np.ones_like(uu), -vv, uu.copy()
        elif face == 2:
            x, y, z = uu.copy(), np.ones_like(uu), vv.copy()
        elif face == 3:
            x, y, z = uu.copy(), -np.ones_like(uu), -vv
        elif face == 4:
            x, y, z = uu.copy(), -vv, np.ones_like(uu)
        else:
            x, y, z = -uu, -vv, -np.ones_like(uu)

        norm = np.sqrt(x * x + y * y + z * z)
        x /= norm
        y /= norm
        z /= norm
        all_dirs.append(np.column_stack((x, y, z)))

    dirs = np.concatenate(all_dirs, axis=0)
    basis = create_sh_basis_matrix(dirs, l_max)
    result = np.ascontiguousarray(basis, dtype=np.float32), size_internal
    _BASIS_CACHE[(l_max, lut_res)] = result
    return result


def populate_hermite_lut_cube_gpu(
    actor, lut_res, glyph_count, n_coeffs, chunk_info, use_float16=False
):
    """Bake Hermite LUT via GPU compute shaders.

    Replaces ``_populate_hermite_lut_cube_cpu_chunked``.

    Returns True on success, False on failure (caller should fall back
    to CPU path).
    """
    device = _get_wgpu_device()
    if device is None:
        return False

    N = lut_res
    g_internal = 3
    size_internal = N + 2 * g_internal
    g = 1
    out_size = N + 2 * g
    start = g_internal - g

    dirs_per_face = size_internal * size_internal
    total_dirs = 6 * dirs_per_face

    l_max = actor._l_max

    basis_flat, _ = _build_basis_matrices_flat(l_max, lut_res)
    if basis_flat.shape[1] > n_coeffs:
        basis_flat = basis_flat[:, :n_coeffs]
    basis_flat = np.ascontiguousarray(basis_flat, dtype=np.float32)

    basis_buf = device.create_buffer_with_data(
        data=basis_flat,
        usage=wgpu.BufferUsage.STORAGE,
    )

    matmul_pipeline, fd_pipeline = _get_pipelines(device)

    coeffs_data = actor.sh_coeffs
    if hasattr(coeffs_data, "data"):
        coeffs_data = coeffs_data.data
    if not isinstance(coeffs_data, np.ndarray):
        coeffs_data = np.asarray(coeffs_data)
    if coeffs_data.ndim == 1:
        coeffs_data = coeffs_data.reshape(-1, n_coeffs)

    glyph_offset = 0
    for chunk_idx, chunk_glyphs in enumerate(chunk_info["chunk_sizes"]):
        chunk_coeffs = np.ascontiguousarray(
            coeffs_data[glyph_offset : glyph_offset + chunk_glyphs],
            dtype=np.float32,
        )

        matmul_total = chunk_glyphs * total_dirs

        matmul_params = np.array(
            [chunk_glyphs, total_dirs, n_coeffs, 0],
            dtype=np.uint32,
        )
        params1_buf = device.create_buffer_with_data(
            data=matmul_params,
            usage=wgpu.BufferUsage.UNIFORM,
        )
        coeffs_buf = device.create_buffer_with_data(
            data=chunk_coeffs,
            usage=wgpu.BufferUsage.STORAGE,
        )
        values_buf = device.create_buffer(
            size=matmul_total * 4,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
            ),
        )

        bg_layout1 = matmul_pipeline.get_bind_group_layout(0)
        bg1 = device.create_bind_group(
            layout=bg_layout1,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": params1_buf,
                        "offset": 0,
                        "size": params1_buf.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": coeffs_buf,
                        "offset": 0,
                        "size": coeffs_buf.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": basis_buf,
                        "offset": 0,
                        "size": basis_buf.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": values_buf,
                        "offset": 0,
                        "size": values_buf.size,
                    },
                },
            ],
        )

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(matmul_pipeline)
        cp.set_bind_group(0, bg1)
        cp.dispatch_workgroups((matmul_total + 255) // 256)
        cp.end()
        device.queue.submit([enc.finish()])

        total_texels = chunk_glyphs * 6 * out_size * out_size

        fd_params = np.array(
            [
                chunk_glyphs,
                size_internal,
                out_size,
                start,
                total_texels,
                0,
                0,
                0,
            ],
            dtype=np.uint32,
        )
        params2_buf = device.create_buffer_with_data(
            data=fd_params,
            usage=wgpu.BufferUsage.UNIFORM,
        )

        output_buf = device.create_buffer(
            size=total_texels * 16,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
            ),
        )

        bg_layout2 = fd_pipeline.get_bind_group_layout(0)
        bg2 = device.create_bind_group(
            layout=bg_layout2,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": params2_buf,
                        "offset": 0,
                        "size": params2_buf.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": values_buf,
                        "offset": 0,
                        "size": values_buf.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": output_buf,
                        "offset": 0,
                        "size": output_buf.size,
                    },
                },
            ],
        )

        enc2 = device.create_command_encoder()
        cp2 = enc2.begin_compute_pass()
        cp2.set_pipeline(fd_pipeline)
        cp2.set_bind_group(0, bg2)
        cp2.dispatch_workgroups((total_texels + 255) // 256)
        cp2.end()
        device.queue.submit([enc2.finish()])

        result_mem = device.queue.read_buffer(output_buf)
        result = np.frombuffer(result_mem, dtype=np.float32).reshape(-1, 4)

        if use_float16:
            result = result.astype(np.float16)

        actor._sh_hermite_lut_buffers[chunk_idx].data[:] = result
        actor._sh_hermite_lut_buffers[chunk_idx].update_range()

        glyph_offset += chunk_glyphs

        params1_buf.destroy()
        coeffs_buf.destroy()
        values_buf.destroy()
        params2_buf.destroy()
        output_buf.destroy()

    basis_buf.destroy()

    return True
