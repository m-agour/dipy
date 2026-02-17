"""SH Glyph Slicer Visualization for Skyline.

This module provides an ``SHGlyph3D`` visualization class that wraps
the interactive SH glyph slicer for use in the Skyline viewer.

The slicer renders spherical harmonic (SH) glyphs from dMRI data
with interactive per-axis slice control via imgui widgets.
"""

import numpy as np

from fury.actor import Group
from imgui_bundle import imgui

from dipy.viz.sh_billboard import sph_glyph_billboard

from dipy.viz.skyline.UI.elements import (
    render_group,
    thin_slider,
)
from dipy.viz.skyline.render.renderer import Visualization


# ---------------------------------------------------------------------------
# Low-level SH slicer (actor-per-slice, visibility-based culling)
# ---------------------------------------------------------------------------


class SHSlicer:
    """Interactive SH glyph slicer for 3D dMRI volumes.

    One billboard actor is created per Z-slice (and optionally per X/Y slice).
    Visibility is controlled by toggling ``actor.visible``.

    Parameters
    ----------
    coeffs_4d : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    voxel_sizes : tuple of 3 floats
        Physical voxel sizes in mm.
    scale : float
        Per-glyph scale factor.
    l_max : int
        Maximum SH order.
    spacing : float
        Spacing multiplier between glyphs.
    lut_res : int
        LUT resolution for billboard precomputation.
    use_hermite : bool
        Use Hermite analytic normals.
    mapping_mode : str
        Billboard mapping mode (``"cube"``, ``"octahedral"``).
    mask : ndarray, optional
        Boolean mask of valid voxels.
    basis_type : str
        SH basis convention (``"standard"``, ``"descoteaux07"``,
        ``"tournier07"``).
    color_type : str
        Colour mapping type (``"orientation"``, ``"radius"``).
    """

    def __init__(
        self,
        coeffs_4d,
        voxel_sizes=(1.0, 1.0, 1.0),
        scale=1.0,
        l_max=8,
        spacing=1.0,
        lut_res=32,
        use_hermite=True,
        mapping_mode="cube",
        mask=None,
        basis_type="standard",
        color_type="orientation",
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
        self.basis_type = basis_type
        self.color_type = color_type

        ref_size = self.voxel_sizes.min()
        self.spacing_ratios = self.voxel_sizes / ref_size

        self._z_slice_actors = [None] * self.shape[2]
        self._x_slice_actors = [None] * self.shape[0]
        self._y_slice_actors = [None] * self.shape[1]

        self._built = False
        self.actor = Group()

    # ----- build ----------------------------------------------------------

    def build(self, axes="z"):
        """Build slice actors for the requested axes.

        Parameters
        ----------
        axes : str
            ``'z'``, ``'x'``, ``'y'``, ``'xy'``, ``'xz'``, ``'yz'``,
            ``'xyz'``, or ``'all'``.

        Returns
        -------
        Group
            Root actor group.
        """
        if self._built:
            return self.actor

        axes = axes.lower()
        if axes == "all":
            axes = "xyz"

        total_glyphs = 0

        if "z" in axes:
            for z in range(self.shape[2]):
                sl = self.coeffs_4d[:, :, z, :]
                a, n = self._build_slice_actor(sl, z, axis="z")
                if a is not None:
                    self._z_slice_actors[z] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "x" in axes:
            for x in range(self.shape[0]):
                sl = self.coeffs_4d[x, :, :, :]
                a, n = self._build_slice_actor(sl, x, axis="x")
                if a is not None:
                    self._x_slice_actors[x] = a
                    self.actor.add(a)
                    total_glyphs += n

        if "y" in axes:
            for y in range(self.shape[1]):
                sl = self.coeffs_4d[:, y, :, :]
                a, n = self._build_slice_actor(sl, y, axis="y")
                if a is not None:
                    self._y_slice_actors[y] = a
                    self.actor.add(a)
                    total_glyphs += n

        self._built_axes = axes
        self._built = True
        return self.actor

    def _build_slice_actor(self, slice_coeffs, slice_idx, axis="z"):
        """Create a billboard actor for one slice."""
        if axis == "z":
            nx, ny = slice_coeffs.shape[0], slice_coeffs.shape[1]
            positions = np.zeros((nx * ny, 3), dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    idx = i * ny + j
                    positions[idx] = [
                        i * self.spacing_ratios[0] * self.spacing,
                        j * self.spacing_ratios[1] * self.spacing,
                        slice_idx * self.spacing_ratios[2] * self.spacing,
                    ]
        elif axis == "x":
            ny, nz = slice_coeffs.shape[0], slice_coeffs.shape[1]
            positions = np.zeros((ny * nz, 3), dtype=np.float32)
            for j in range(ny):
                for k in range(nz):
                    idx = j * nz + k
                    positions[idx] = [
                        slice_idx * self.spacing_ratios[0] * self.spacing,
                        j * self.spacing_ratios[1] * self.spacing,
                        k * self.spacing_ratios[2] * self.spacing,
                    ]
        elif axis == "y":
            nx, nz = slice_coeffs.shape[0], slice_coeffs.shape[1]
            positions = np.zeros((nx * nz, 3), dtype=np.float32)
            for i in range(nx):
                for k in range(nz):
                    idx = i * nz + k
                    positions[idx] = [
                        i * self.spacing_ratios[0] * self.spacing,
                        slice_idx * self.spacing_ratios[1] * self.spacing,
                        k * self.spacing_ratios[2] * self.spacing,
                    ]

        flat_coeffs = slice_coeffs.reshape(-1, self.n_coeffs)

        valid_mask = np.any(flat_coeffs != 0, axis=1)
        if self.mask is not None:
            if axis == "z":
                sm = self.mask[:, :, slice_idx].ravel()
            elif axis == "x":
                sm = self.mask[slice_idx, :, :].ravel()
            else:
                sm = self.mask[:, slice_idx, :].ravel()
            valid_mask &= sm

        if not np.any(valid_mask):
            return None, 0

        valid_coeffs = flat_coeffs[valid_mask]
        valid_positions = positions[valid_mask]

        glyph = sph_glyph_billboard(
            valid_coeffs,
            centers=valid_positions,
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
        return glyph, len(valid_coeffs)

    # ----- slice visibility -----------------------------------------------

    def set_z_range(self, z_min, z_max):
        for z, a in enumerate(self._z_slice_actors):
            if a is not None:
                a.visible = z_min <= z < z_max

    def set_z_slice(self, z):
        self.set_z_range(z, z + 1)

    def show_all_z(self):
        self.set_z_range(0, self.shape[2])

    def set_x_range(self, x_min, x_max):
        for x, a in enumerate(self._x_slice_actors):
            if a is not None:
                a.visible = x_min <= x < x_max

    def set_x_slice(self, x):
        self.set_x_range(x, x + 1)

    def show_all_x(self):
        self.set_x_range(0, self.shape[0])

    def set_y_range(self, y_min, y_max):
        for y, a in enumerate(self._y_slice_actors):
            if a is not None:
                a.visible = y_min <= y < y_max

    def set_y_slice(self, y):
        self.set_y_range(y, y + 1)

    def show_all_y(self):
        self.set_y_range(0, self.shape[1])

    def set_range(self, axis, min_idx, max_idx):
        axis = axis.lower()
        if axis == "x":
            self.set_x_range(min_idx, max_idx)
        elif axis == "y":
            self.set_y_range(min_idx, max_idx)
        elif axis == "z":
            self.set_z_range(min_idx, max_idx)

    def hide_all(self, axis=None):
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


# ---------------------------------------------------------------------------
# Skyline Visualization wrapper
# ---------------------------------------------------------------------------


class SHGlyph3D(Visualization):
    """SH glyph slicer visualization for the Skyline viewer.

    Wraps :class:`SHSlicer` and exposes imgui widgets for interactive
    per-axis slice control, scale, and SH-order adjustment.

    Parameters
    ----------
    name : str
        Display name in the UI sidebar.
    coeffs : ndarray, shape (X, Y, Z, n_coeffs)
        SH coefficients.
    affine : ndarray (4, 4), optional
        Voxel-to-world affine.  Voxel sizes are extracted from the first
        three diagonal elements.
    render_callback : callable, optional
        Called after parameter changes to trigger a scene re-render.
    scale : float
        Initial per-glyph scale.
    l_max : int
        Maximum SH order.
    lut_res : int
        LUT resolution.
    use_hermite : bool
        Whether to use Hermite analytic normals.
    mapping_mode : str
        Billboard mapping mode.
    basis_type : str
        SH basis convention (``"standard"``, ``"descoteaux07"``,
        ``"tournier07"``).
    color_type : str
        Colour mapping.
    mask : ndarray, optional
        Boolean mask of valid voxels.
    """

    def __init__(
        self,
        name,
        coeffs,
        *,
        affine=None,
        render_callback=None,
        scale=0.4,
        l_max=8,
        lut_res=8,
        use_hermite=True,
        mapping_mode="cube",
        basis_type="standard",
        color_type="orientation",
        mask=None,
    ):
        super().__init__(name, render_callback)

        # Extract voxel sizes from affine
        if affine is not None:
            voxel_sizes = tuple(np.abs(np.diag(affine)[:3]).astype(float))
        else:
            voxel_sizes = (1.0, 1.0, 1.0)

        self._slicer = SHSlicer(
            coeffs,
            voxel_sizes=voxel_sizes,
            scale=scale,
            l_max=l_max,
            lut_res=lut_res,
            use_hermite=use_hermite,
            mapping_mode=mapping_mode,
            mask=mask,
            basis_type=basis_type,
            color_type=color_type,
        )
        self._slicer.build(axes="z")

        self.shape = coeffs.shape[:3]
        # Current Z-slice index (start at midpoint)
        self._z_idx = self.shape[2] // 2
        self._z_min = 0
        self._z_max = self.shape[2] - 1

        # Start by showing a single mid-Z slice
        self._slicer.set_z_slice(self._z_idx)

        self.bounds = [
            [0, 0, 0],
            list(self.shape),
        ]

    @property
    def actor(self):
        return self._slicer.actor

    # ----- imgui widgets --------------------------------------------------

    def render_widgets(self):
        """Render slicer control widgets in the Skyline sidebar."""
        # Z-slice slider
        changed, _ptr, new_z = thin_slider(
            "Z Slice",
            self._z_idx,
            0,
            self.shape[2] - 1,
            value_type="int",
            text_format=".0f",
            step=1,
        )
        if changed:
            self._z_idx = int(new_z)
            self._slicer.set_z_slice(self._z_idx)
            self.render()

        imgui.spacing()

        # Z-range sliders (min / max)
        slicers = [
            (
                thin_slider,
                ("Z Min", self._z_min, 0, self.shape[2] - 1),
                {"value_type": "int", "text_format": ".0f", "step": 1},
            ),
            (
                thin_slider,
                ("Z Max", self._z_max, 0, self.shape[2] - 1),
                {"value_type": "int", "text_format": ".0f", "step": 1},
            ),
        ]
        render_data = render_group("Z Range", slicers)
        range_changed = False
        for idx, (changed, _ptr, new_val) in enumerate(render_data):
            if changed:
                range_changed = True
                if idx == 0:
                    self._z_min = int(new_val)
                else:
                    self._z_max = int(new_val)
        if range_changed:
            self._slicer.set_z_range(self._z_min, self._z_max + 1)
            self.render()

        imgui.spacing()
