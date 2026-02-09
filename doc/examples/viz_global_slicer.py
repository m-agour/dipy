"""
=====================================================
Global Multi-Layer Slicer — SH Glyphs + DWI + Peaks
=====================================================

Unified slicer that simultaneously slices SH glyphs (Hermite billboard),
DWI volume planes, and CSD peak directions, all controlled by a single
ImGui GUI with three axis indices and three visibility toggles.

Run::

    python viz_global_slicer.py
    python viz_global_slicer.py --roi-size 15 --multishell
    python viz_global_slicer.py --synthetic

Controls
--------
* **X / Y / Z index** — shared slice position across all layers
* **Show X / Y / Z** — toggle axis visibility for all layers
* **Layer toggles** — independently hide SH glyphs, DWI volume, or peaks
* **SH scale** — adjust glyph size interactively
"""

from __future__ import annotations

import argparse
import time

from fury import actor, window
from imgui_bundle import imgui
import numpy as np

from dipy.viz.sh_glyph import sh_slicer


def load_dmri_data(roi_size=None, sh_order=8, multishell=False):
    """Fetch Stanford HARDI, fit CSD, extract SH + peaks + FA volume.

    When *roi_size* is ``None`` (default) the full masked brain is used.
    Otherwise a cube of half-width *roi_size* centred on the mask is
    extracted.
    """
    from dipy.core.gradients import gradient_table
    from dipy.data import fetch_stanford_hardi, read_stanford_hardi
    from dipy.direction import peaks_from_model
    from dipy.reconst.csdeconv import (
        ConstrainedSphericalDeconvModel,
        auto_response_ssst,
    )
    from dipy.reconst.dti import TensorModel, fractional_anisotropy
    from dipy.segment.mask import median_otsu

    print("Fetching Stanford HARDI ...")
    fetch_stanford_hardi()
    img, gtab = read_stanford_hardi()
    data = img.get_fdata()
    affine = img.affine
    voxel_sizes = tuple(np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)))

    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1,
    )

    if roi_size is not None:
        mc = [np.where(mask)[i] for i in range(3)]
        cx, cy, cz = [int(np.mean(c)) for c in mc]
        h = roi_size // 2
        sl = tuple(
            slice(max(0, c - h), min(s, c + h))
            for c, s in zip((cx, cy, cz), data.shape[:3])
        )
        roi_data = data[sl + (slice(None),)]
        roi_mask = mask[sl]
    else:
        roi_data = maskdata
        roi_mask = mask

    unique_shells = np.unique(np.round(gtab.bvals[gtab.bvals > 50], -2))
    is_multishell = len(unique_shells) > 1

    if not multishell and is_multishell:
        first_shell = unique_shells[0]
        shell_idx = (gtab.bvals < 50) | (
            np.abs(gtab.bvals - first_shell) < 200
        )
        roi_data_shell = roi_data[..., shell_idx]
        gtab_shell = gradient_table(
            gtab.bvals[shell_idx], bvecs=gtab.bvecs[shell_idx],
        )
    else:
        roi_data_shell = roi_data
        gtab_shell = gtab

    print(f"ROI shape: {roi_data_shell.shape[:3]}, "
          f"shells: {np.unique(np.round(gtab_shell.bvals, -2))}")

    response, _ = auto_response_ssst(
        gtab, maskdata,
        roi_radii=10, fa_thr=0.7,
    )

    csd_model = ConstrainedSphericalDeconvModel(
        gtab_shell, response, sh_order_max=sh_order,
    )
    csd_fit = csd_model.fit(roi_data_shell, mask=roi_mask)
    sh_coeffs = csd_fit.shm_coeff.astype(np.float32)

    from dipy.data import default_sphere

    peaks = peaks_from_model(
        csd_model, roi_data_shell, default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=roi_mask,
        return_sh=False,
        npeaks=5,
        normalize_peaks=True,
    )

    dti_model = TensorModel(gtab_shell)
    dti_fit = dti_model.fit(roi_data_shell, mask=roi_mask)
    fa = fractional_anisotropy(dti_fit.evals).astype(np.float32)
    fa = np.clip(fa, 0, 1)

    return {
        "sh_coeffs": sh_coeffs,
        "peak_dirs": peaks.peak_dirs.astype(np.float32),
        "peak_values": peaks.peak_values.astype(np.float32),
        "fa": fa,
        "mask": roi_mask,
        "voxel_sizes": voxel_sizes,
        "roi_shape": sh_coeffs.shape[:3],
    }


def make_synthetic_data(roi_size=10, l_max=8):
    """Generate synthetic SH + peaks + FA for testing without download."""
    nx = ny = nz = roi_size
    n_coeffs = (l_max + 1) ** 2

    sh_coeffs = np.zeros((nx, ny, nz, n_coeffs), dtype=np.float32)
    rng = np.random.default_rng(42)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                phase = (x + y + z) * 0.3
                sh_coeffs[x, y, z, 0] = 0.5 + 0.2 * np.sin(phase)
                for big_l in range(2, l_max + 1, 2):
                    for m in range(-big_l, big_l + 1):
                        idx = big_l * big_l + big_l + m
                        if idx < n_coeffs:
                            sh_coeffs[x, y, z, idx] = (
                                rng.normal() * 0.15 / big_l
                            )

    peak_dirs = np.zeros((nx, ny, nz, 3, 3), dtype=np.float32)
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                t = (x + y * 0.5 + z * 0.3) * 0.4
                peak_dirs[x, y, z, 0] = [
                    np.cos(t), np.sin(t), 0.1 * np.sin(2 * t),
                ]
                peak_dirs[x, y, z, 1] = [
                    -np.sin(t), np.cos(t), 0.2,
                ]
                peak_dirs[x, y, z, 2] = [0, 0, 1]
    norms = np.linalg.norm(peak_dirs, axis=-1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    peak_dirs /= norms
    peak_dirs *= 0.4

    fa = np.clip(
        0.3 + 0.4 * np.sin(
            np.linspace(0, np.pi, nx)[:, None, None]
            * np.linspace(0, np.pi, ny)[None, :, None]
            * np.linspace(0, np.pi, nz)[None, None, :],
        ),
        0, 1,
    ).astype(np.float32)

    mask = np.ones((nx, ny, nz), dtype=bool)

    return {
        "sh_coeffs": sh_coeffs,
        "peak_dirs": peak_dirs,
        "peak_values": np.ones((nx, ny, nz, 3), dtype=np.float32) * 0.5,
        "fa": fa,
        "mask": mask,
        "voxel_sizes": (1.0, 1.0, 1.0),
        "roi_shape": (nx, ny, nz),
    }


class GlobalSlicer:
    """Unified multi-layer slicer.

    Coordinates slice indices across:
    * SH billboard glyphs (``SHSlicer``)
    * DWI volume planes (``actor.data_slicer``)
    * CSD peak lines (``actor.peaks_slicer``)

    All three layers share a single set of (x, y, z) indices and
    (show_x, show_y, show_z) visibility flags.
    """

    def __init__(
        self,
        dmri,
        *,
        sh_scale=0.4,
        lut_res=8,
        use_hermite=True,
        mapping_mode="cube",
        sh_order=8,
    ):
        self.shape = dmri["roi_shape"]
        nx, ny, nz = self.shape

        self.x_idx = nx // 2
        self.y_idx = ny // 2
        self.z_idx = nz // 2
        self.show_x = True
        self.show_y = True
        self.show_z = True

        self.show_sh = True
        self.show_vol = True
        self.show_peaks = True
        self.sh_scale = sh_scale
        self.sh_opacity = 1.0
        self.vol_opacity = 0.6
        self.peaks_opacity = 1.0

        print(f"Building SH slicer ({nx}x{ny}x{nz}) ...")
        t0 = time.time()
        sh_coeffs = dmri["sh_coeffs"]
        max_val = np.abs(sh_coeffs[..., 0]).max()
        if max_val > 0:
            sh_coeffs = sh_coeffs / max_val * 0.5

        self.sh_slicer = sh_slicer(
            sh_coeffs,
            voxel_sizes=dmri["voxel_sizes"],
            scale=sh_scale,
            l_max=sh_order,
            spacing=1.0,
            lut_res=lut_res,
            use_hermite=use_hermite,
            mapping_mode=mapping_mode,
            mask=dmri["mask"],
            basis_type="descoteaux07",
        )
        self.sh_group = self.sh_slicer.build(axes="xyz")
        dt = time.time() - t0
        print(f"  SH built in {dt:.1f}s  "
              f"({self.sh_slicer.get_total_glyphs()} slice actors)")

        print("Building DWI volume slicer ...")
        self.vol_group = actor.data_slicer(
            dmri["fa"],
            initial_slices=(self.x_idx, self.y_idx, self.z_idx),
            visibility=(self.show_x, self.show_y, self.show_z),
            interpolation="linear",
            opacity=0.6,
        )

        print("Building peaks slicer ...")
        self.peaks_actor = actor.peaks_slicer(
            dmri["peak_dirs"],
            peak_values=dmri["peak_values"],
            visibility=(self.show_x, self.show_y, self.show_z),
            opacity=1.0,
        )
        self.peaks_actor.cross_section = np.array(
            [self.x_idx, self.y_idx, self.z_idx], dtype=np.int32,
        )

        self._sync()

    def add_to_scene(self, scene):
        """Add all layers to the scene."""
        scene.add(self.sh_group)
        scene.add(self.vol_group)
        scene.add(self.peaks_actor)

    def _sync(self):
        """Push current state to all layers."""
        vis = (
            self.show_x and self.show_sh,
            self.show_y and self.show_sh,
            self.show_z and self.show_sh,
        )
        if vis[0]:
            self.sh_slicer.set_x_slice(self.x_idx)
        else:
            self.sh_slicer.hide_all("x")
        if vis[1]:
            self.sh_slicer.set_y_slice(self.y_idx)
        else:
            self.sh_slicer.hide_all("y")
        if vis[2]:
            self.sh_slicer.set_z_slice(self.z_idx)
        else:
            self.sh_slicer.hide_all("z")

        vol_vis = (
            self.show_x and self.show_vol,
            self.show_y and self.show_vol,
            self.show_z and self.show_vol,
        )
        for i, child in enumerate(self.vol_group.children):
            child.visible = vol_vis[i]
        actor.show_slices(
            self.vol_group,
            (self.x_idx, self.y_idx, self.z_idx),
        )

        peaks_vis = (
            self.show_x and self.show_peaks,
            self.show_y and self.show_peaks,
            self.show_z and self.show_peaks,
        )
        self.peaks_actor.visibility = peaks_vis
        self.peaks_actor.cross_section = np.array(
            [self.x_idx, self.y_idx, self.z_idx], dtype=np.int32,
        )

        self._apply_opacity()

    def _apply_opacity(self):
        """Push per-layer opacity to all actors."""
        for actors in (
            self.sh_slicer._x_slice_actors,
            self.sh_slicer._y_slice_actors,
            self.sh_slicer._z_slice_actors,
        ):
            for a in actors:
                if a is not None:
                    a.material.opacity = self.sh_opacity

        for child in self.vol_group.children:
            child.material.opacity = self.vol_opacity

        self.peaks_actor.material.opacity = self.peaks_opacity

    def _apply_sh_scale(self):
        """Set per-glyph scale on every SH slice actor (scales in place)."""
        for actors in (
            self.sh_slicer._x_slice_actors,
            self.sh_slicer._y_slice_actors,
            self.sh_slicer._z_slice_actors,
        ):
            for a in actors:
                if a is not None:
                    a.material.scale = self.sh_scale

    def set_indices(self, x=None, y=None, z=None):
        """Update one or more axis indices and sync all layers."""
        if x is not None:
            self.x_idx = int(np.clip(x, 0, self.shape[0] - 1))
        if y is not None:
            self.y_idx = int(np.clip(y, 0, self.shape[1] - 1))
        if z is not None:
            self.z_idx = int(np.clip(z, 0, self.shape[2] - 1))
        self._sync()

    def set_visibility(self, show_x=None, show_y=None, show_z=None):
        """Update axis visibility and sync all layers."""
        if show_x is not None:
            self.show_x = show_x
        if show_y is not None:
            self.show_y = show_y
        if show_z is not None:
            self.show_z = show_z
        self._sync()


def build_imgui(gs):
    """Return the imgui draw function for the global slicer."""
    nx, ny, nz = gs.shape

    def gui():
        changed = False

        if imgui.begin("Global Slicer"):
            imgui.text(f"Volume {nx} x {ny} x {nz}")
            imgui.separator()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.6, 0.2, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "Slice Indices", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()

                c, v = imgui.slider_int("X index", gs.x_idx, 0, nx - 1)
                if c:
                    gs.x_idx = v
                    changed = True

                c, v = imgui.slider_int("Y index", gs.y_idx, 0, ny - 1)
                if c:
                    gs.y_idx = v
                    changed = True

                c, v = imgui.slider_int("Z index", gs.z_idx, 0, nz - 1)
                if c:
                    gs.z_idx = v
                    changed = True

                if imgui.button("Center"):
                    gs.x_idx = nx // 2
                    gs.y_idx = ny // 2
                    gs.z_idx = nz // 2
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()
            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.2, 0.6, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "Axis Visibility", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()

                c, v = imgui.checkbox("Show X", gs.show_x)
                if c:
                    gs.show_x = v
                    changed = True

                c, v = imgui.checkbox("Show Y", gs.show_y)
                if c:
                    gs.show_y = v
                    changed = True

                c, v = imgui.checkbox("Show Z", gs.show_z)
                if c:
                    gs.show_z = v
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()
            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.2, 0.2, 0.6, 0.8),
            )
            if imgui.collapsing_header(
                "Layer Visibility", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()

                c, v = imgui.checkbox("SH Glyphs", gs.show_sh)
                if c:
                    gs.show_sh = v
                    changed = True

                c, v = imgui.checkbox("DWI Volume", gs.show_vol)
                if c:
                    gs.show_vol = v
                    changed = True

                c, v = imgui.checkbox("Peaks", gs.show_peaks)
                if c:
                    gs.show_peaks = v
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()
            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.5, 0.4, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "Settings", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()

                c, v = imgui.slider_float(
                    "SH Scale", gs.sh_scale, 0.05, 2.0, "%.2f",
                )
                if c:
                    gs.sh_scale = v
                    gs._apply_sh_scale()

                imgui.spacing()
                imgui.text("Opacity")

                c, v = imgui.slider_float(
                    "SH Glyphs##op", gs.sh_opacity, 0.0, 1.0, "%.2f",
                )
                if c:
                    gs.sh_opacity = v
                    changed = True

                c, v = imgui.slider_float(
                    "DWI Volume##op", gs.vol_opacity, 0.0, 1.0, "%.2f",
                )
                if c:
                    gs.vol_opacity = v
                    changed = True

                c, v = imgui.slider_float(
                    "Peaks##op", gs.peaks_opacity, 0.0, 1.0, "%.2f",
                )
                if c:
                    gs.peaks_opacity = v
                    changed = True
            else:
                imgui.pop_style_color()

        imgui.end()

        if changed:
            gs._sync()

    return gui


def main():
    parser = argparse.ArgumentParser(
        description="Global Multi-Layer Slicer: SH + DWI + Peaks",
    )
    parser.add_argument(
        "--roi-size", type=int, default=None,
        help="Half-width of the ROI cube (default: full brain)",
    )
    parser.add_argument(
        "--sh-order", type=int, default=8,
        help="SH order for CSD fitting (default: 8)",
    )
    parser.add_argument(
        "--lut-res", type=int, default=8,
        help="LUT resolution per glyph (default: 8)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.4,
        help="Initial SH glyph scale (default: 0.4)",
    )
    parser.add_argument(
        "--multishell", action="store_true",
        help="Use all shells instead of first shell only",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no download needed)",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic data ...")
        dmri = make_synthetic_data(args.roi_size, args.sh_order)
    else:
        try:
            dmri = load_dmri_data(
                roi_size=args.roi_size,
                sh_order=args.sh_order,
                multishell=args.multishell,
            )
        except Exception as e:
            print(f"dMRI load failed ({e}), falling back to synthetic.")
            dmri = make_synthetic_data(args.roi_size, args.sh_order)

    gs = GlobalSlicer(
        dmri,
        sh_scale=args.scale,
        lut_res=args.lut_res,
        use_hermite=True,
        mapping_mode="cube",
        sh_order=args.sh_order,
    )

    scene = window.Scene()
    scene.background = (0.05, 0.05, 0.1)
    gs.add_to_scene(scene)

    gui_fn = build_imgui(gs)

    sm = window.ShowManager(
        scene=scene,
        size=(1400, 900),
        title="Global Slicer — SH + DWI + Peaks",
        show_fps=True,
        imgui=True,
        imgui_draw_function=gui_fn,
    )
    sm.start()


if __name__ == "__main__":
    main()
