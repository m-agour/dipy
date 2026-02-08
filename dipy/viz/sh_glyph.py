"""SH glyph visualization — real-time, LUT-based, and volume slicer modes."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, _setup_msg = optional_package("fury", min_version="0.10.0")
if not has_fury:
    fury, has_fury, _setup_msg = optional_package("fury")

_FURY_MSG = (
    "FURY >= 2.0 is required for SH billboard visualization. "
    "Install with:  pip install git+https://github.com/fury-gl/fury.git@v2"
)


def _require_fury():
    if not has_fury:
        raise ImportError(_FURY_MSG)


if has_fury:
    try:
        from dipy.viz.wgsl import _register_dipy_wgsl_loader

        _register_dipy_wgsl_loader()
    except Exception:
        pass

    try:
        import dipy.viz.sh_billboard  # noqa: F401
    except Exception:
        pass


def sh_glyph_billboard(
    sh_coeffs,
    *,
    centers=None,
    basis_type="standard",
    color_type="orientation",
    l_max=None,
    scale=1.0,
    opacity=None,
):
    """Create SH glyph billboards with real-time per-fragment evaluation."""
    _require_fury()
    from dipy.viz.sh_billboard import sph_glyph_billboard as _bb

    sh_coeffs = np.asarray(sh_coeffs, dtype=np.float32)
    if basis_type == "descoteaux07":
        _l = l_max
        if _l is None:
            from fury.utils import get_lmax

            _l = get_lmax(
                sh_coeffs.shape[-1], basis_type="descoteaux07",
            )
        sh_coeffs = descoteaux07_to_standard(sh_coeffs, _l)
        basis_type = "standard"

    return _bb(
        sh_coeffs,
        centers=centers,
        basis_type=basis_type,
        color_type=color_type,
        l_max=l_max,
        scale=scale,
        opacity=opacity,
        use_precomputation=False,
    )


def sh_glyph_billboard_lut(
    sh_coeffs,
    *,
    centers=None,
    basis_type="standard",
    color_type="orientation",
    l_max=None,
    scale=1.0,
    opacity=None,
    mapping_mode="cube",
    lut_res=64,
    use_hermite=True,
    use_float16=False,
    interpolation_mode=2,
):
    """Create SH glyph billboards with precomputed radius LUT."""
    _require_fury()
    from dipy.viz.sh_billboard import create_large_scale_sh_billboards

    sh_coeffs = np.asarray(sh_coeffs, dtype=np.float32)
    if basis_type == "descoteaux07":
        _l = l_max
        if _l is None:
            from fury.utils import get_lmax

            _l = get_lmax(
                sh_coeffs.shape[-1], basis_type="descoteaux07",
            )
        sh_coeffs = descoteaux07_to_standard(sh_coeffs, _l)
        basis_type = "standard"

    return create_large_scale_sh_billboards(
        sh_coeffs,
        centers=centers,
        basis_type=basis_type,
        color_type=color_type,
        l_max=l_max,
        scale=scale,
        opacity=opacity,
        lut_res=lut_res,
        mapping_mode=mapping_mode,
        use_hermite=use_hermite,
        use_float16=use_float16,
        interpolation_mode=interpolation_mode,
    )


def descoteaux07_to_standard(sh_coeffs, l_max):
    """Convert descoteaux07 (even-only) SH coefficients to standard basis."""
    n_std = (l_max + 1) ** 2
    shape_out = sh_coeffs.shape[:-1] + (n_std,)
    coeffs_std = np.zeros(shape_out, dtype=np.float32)

    desc_idx = 0
    for l_order in range(0, l_max + 1, 2):
        for m in range(-l_order, l_order + 1):
            std_idx = l_order * l_order + l_order + m
            coeffs_std[..., std_idx] = sh_coeffs[..., desc_idx]
            desc_idx += 1

    return coeffs_std


class SHSlicer:
    """Interactive SH glyph slicer for 3-D volumes."""

    def __init__(
        self,
        coeffs_4d: np.ndarray,
        voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        scale: float = 1.0,
        l_max: int = 8,
        spacing: float = 1.0,
        lut_res: int = 32,
        use_hermite: bool = True,
        mapping_mode: str = "cube",
        mask: Optional[np.ndarray] = None,
        basis_type: str = "descoteaux07",
        color_type: str = "orientation",
    ):
        self.coeffs_4d = coeffs_4d
        self.shape = coeffs_4d.shape[:3]
        self.n_coeffs = coeffs_4d.shape[-1]
        self.voxel_sizes = np.array(voxel_sizes)
        self.scale = scale
        self.l_max = l_max
        self.spacing = spacing
        self.lut_res = lut_res
        self.use_hermite = use_hermite
        self.mapping_mode = mapping_mode
        self.mask = mask
        self.color_type = color_type

        if basis_type == "descoteaux07":
            self.coeffs_4d = descoteaux07_to_standard(
                self.coeffs_4d, l_max,
            )
            self.n_coeffs = self.coeffs_4d.shape[-1]
            self.basis_type = "standard"
        else:
            self.basis_type = basis_type

        ref_size = self.voxel_sizes.min()
        self.spacing_ratios = self.voxel_sizes / ref_size

        self._z_slice_actors: List[Optional[object]] = (
            [None] * self.shape[2]
        )
        self._x_slice_actors: List[Optional[object]] = (
            [None] * self.shape[0]
        )
        self._y_slice_actors: List[Optional[object]] = (
            [None] * self.shape[1]
        )

        self._built = False
        self._built_axes = ""

        _require_fury()
        from fury.actor import Group

        self.actor = Group()

    def build(self, axes: str = "z"):
        """Build slice actors for the requested axes and return the Group."""
        if self._built:
            return self.actor

        axes = axes.lower()
        if axes == "all":
            axes = "xyz"

        total_glyphs = 0

        if "z" in axes:
            for z in range(self.shape[2]):
                sc = self.coeffs_4d[:, :, z, :]
                a, n = self._build_slice_actor(sc, z, axis="z")
                if a is not None:
                    self._z_slice_actors[z] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "x" in axes:
            for x in range(self.shape[0]):
                sc = self.coeffs_4d[x, :, :, :]
                a, n = self._build_slice_actor(sc, x, axis="x")
                if a is not None:
                    self._x_slice_actors[x] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "y" in axes:
            for y in range(self.shape[1]):
                sc = self.coeffs_4d[:, y, :, :]
                a, n = self._build_slice_actor(sc, y, axis="y")
                if a is not None:
                    self._y_slice_actors[y] = a
                    self.actor.add(a)
                    total_glyphs += n

        self._built_axes = axes
        self._built = True
        return self.actor

    def _build_slice_actor(
        self,
        slice_coeffs: np.ndarray,
        slice_idx: int,
        axis: str = "z",
    ) -> Tuple[Optional[object], int]:
        positions, flat_coeffs = self._slice_positions(
            slice_coeffs, slice_idx, axis
        )

        valid = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            if axis == "z":
                sm = self.mask[:, :, slice_idx].ravel()
            elif axis == "x":
                sm = self.mask[slice_idx, :, :].ravel()
            else:
                sm = self.mask[:, slice_idx, :].ravel()
            valid &= sm

        if not np.any(valid):
            return None, 0

        from dipy.viz.sh_billboard import sph_glyph_billboard as _bb

        glyph = _bb(
            flat_coeffs[valid],
            centers=positions[valid],
            scale=self.scale,
            l_max=self.l_max,
            basis_type=self.basis_type,
            color_type=self.color_type,
            use_precomputation=True,
            use_bicubic=True,
            use_hermite=self.use_hermite,
            mapping_mode=self.mapping_mode,
            lut_res=self.lut_res,
        )
        return glyph, int(valid.sum())

    def _slice_positions(self, slice_coeffs, slice_idx, axis):
        sr = self.spacing_ratios * self.spacing

        if axis == "z":
            nx, ny = slice_coeffs.shape[:2]
            ix, iy = np.meshgrid(
                np.arange(nx), np.arange(ny), indexing="ij"
            )
            pos = np.zeros((nx * ny, 3), dtype=np.float32)
            pos[:, 0] = ix.ravel() * sr[0]
            pos[:, 1] = iy.ravel() * sr[1]
            pos[:, 2] = slice_idx * sr[2]
        elif axis == "x":
            ny, nz = slice_coeffs.shape[:2]
            iy, iz = np.meshgrid(
                np.arange(ny), np.arange(nz), indexing="ij"
            )
            pos = np.zeros((ny * nz, 3), dtype=np.float32)
            pos[:, 0] = slice_idx * sr[0]
            pos[:, 1] = iy.ravel() * sr[1]
            pos[:, 2] = iz.ravel() * sr[2]
        else:
            nx, nz = slice_coeffs.shape[:2]
            ix, iz = np.meshgrid(
                np.arange(nx), np.arange(nz), indexing="ij"
            )
            pos = np.zeros((nx * nz, 3), dtype=np.float32)
            pos[:, 0] = ix.ravel() * sr[0]
            pos[:, 1] = slice_idx * sr[1]
            pos[:, 2] = iz.ravel() * sr[2]

        flat = slice_coeffs.reshape(-1, self.n_coeffs)
        return pos, flat

    def set_z_range(self, z_min: int, z_max: int):
        """Set visible Z-slice range ``[z_min, z_max)``."""
        for z, a in enumerate(self._z_slice_actors):
            if a is not None:
                a.visible = z_min <= z < z_max

    def set_z_slice(self, z: int):
        """Show only a single Z slice."""
        self.set_z_range(z, z + 1)

    def show_all_z(self):
        """Show all Z slices."""
        self.set_z_range(0, self.shape[2])

    def set_x_range(self, x_min: int, x_max: int):
        """Set visible X-slice range ``[x_min, x_max)``."""
        for x, a in enumerate(self._x_slice_actors):
            if a is not None:
                a.visible = x_min <= x < x_max

    def set_x_slice(self, x: int):
        """Show only a single X slice."""
        self.set_x_range(x, x + 1)

    def show_all_x(self):
        """Show all X slices."""
        self.set_x_range(0, self.shape[0])

    def set_y_range(self, y_min: int, y_max: int):
        """Set visible Y-slice range ``[y_min, y_max)``."""
        for y, a in enumerate(self._y_slice_actors):
            if a is not None:
                a.visible = y_min <= y < y_max

    def set_y_slice(self, y: int):
        """Show only a single Y slice."""
        self.set_y_range(y, y + 1)

    def show_all_y(self):
        """Show all Y slices."""
        self.set_y_range(0, self.shape[1])

    def set_range(self, axis: str, min_idx: int, max_idx: int):
        a = axis.lower()
        if a == "x":
            self.set_x_range(min_idx, max_idx)
        elif a == "y":
            self.set_y_range(min_idx, max_idx)
        elif a == "z":
            self.set_z_range(min_idx, max_idx)

    def set_slice(self, axis: str, idx: int):
        self.set_range(axis, idx, idx + 1)

    def show_all(self, axis: Optional[str] = None):
        if axis is None:
            for ax in self._built_axes:
                self.show_all(ax)
        else:
            a = axis.lower()
            if a == "x":
                self.show_all_x()
            elif a == "y":
                self.show_all_y()
            elif a == "z":
                self.show_all_z()

    def hide_all(self, axis: Optional[str] = None):
        if axis is None:
            for actors in (
                self._x_slice_actors,
                self._y_slice_actors,
                self._z_slice_actors,
            ):
                for a in actors:
                    if a is not None:
                        a.visible = False
        else:
            self.set_range(axis, 0, 0)

    def get_slice_count(self, axis: str) -> int:
        a = axis.lower()
        if a == "x":
            return self.shape[0]
        if a == "y":
            return self.shape[1]
        if a == "z":
            return self.shape[2]
        return 0

    def get_total_glyphs(self) -> int:
        n = 0
        for actors in (
            self._x_slice_actors,
            self._y_slice_actors,
            self._z_slice_actors,
        ):
            n += sum(1 for a in actors if a is not None)
        return n


def sh_slicer(
    coeffs_4d: np.ndarray,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    scale: float = 1.0,
    l_max: int = 8,
    spacing: float = 1.0,
    lut_res: int = 32,
    use_hermite: bool = True,
    mapping_mode: str = "cube",
    mask: Optional[np.ndarray] = None,
    basis_type: str = "descoteaux07",
    color_type: str = "orientation",
) -> SHSlicer:
    """Create an SHSlicer instance."""
    return SHSlicer(
        coeffs_4d=coeffs_4d,
        voxel_sizes=voxel_sizes,
        scale=scale,
        l_max=l_max,
        spacing=spacing,
        lut_res=lut_res,
        use_hermite=use_hermite,
        mapping_mode=mapping_mode,
        mask=mask,
        basis_type=basis_type,
        color_type=color_type,
    )
