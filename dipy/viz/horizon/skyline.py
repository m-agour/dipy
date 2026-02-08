"""Skyline — GPU-accelerated SH billboard slicer (FURY v2 / pygfx).

Parallel to :class:`Horizon` but built on the pygfx/wgpu/imgui stack
instead of VTK.  Accepts NIfTI images, PAM objects, or raw SH
coefficient arrays and renders an interactive multi-layer slicer
combining:

* SH billboard glyphs  (Hermite analytic normals, cube mapping)
* DWI / FA volume planes
* CSD peak directions

All layers share a single set of (x, y, z) slice indices and
(show_x, show_y, show_z) visibility toggles, controlled through an
ImGui panel.

Usage
-----
From Python::

    from dipy.viz.horizon.skyline import skyline
    skyline(images=[(fa_data, affine, "FA")],
            sh_coeffs=sh, peak_dirs=peaks.peak_dirs)

From the command line::

    dipy_skyline my_dwi.nii.gz my_fit.pam5
"""

from __future__ import annotations

import time

import numpy as np

from dipy.utils.optpkg import optional_package

fury, has_fury, setup_module = optional_package("fury", min_version="0.10.0")


class Skyline:
    """Interactive multi-layer slicer using FURY v2 (pygfx / imgui).

    Parameters
    ----------
    images : list of (ndarray, ndarray, str), optional
        Each entry is ``(data_3d, affine_4x4, label)``.  The first
        scalar volume is used for the background slicer plane.
    pams : list of (PeakAndMetrics, str), optional
        PAM objects produced by ``peaks_from_model`` or loaded from
        ``.pam5`` files.  ``pam.shm_coeff`` supplies SH coefficients
        and ``pam.peak_dirs`` supplies peak directions.
    sh_coeffs : ndarray, optional
        Explicit SH coefficient array ``(X, Y, Z, n_coeffs)``.  If a
        PAM is also given the explicit array takes priority.
    peak_dirs : ndarray, optional
        Explicit peak directions ``(X, Y, Z, N, 3)``.
    peak_values : ndarray, optional
        Explicit peak values ``(X, Y, Z, N)``.
    mask : ndarray, optional
        Boolean brain mask.
    basis_type : str
        SH basis convention.  ``"descoteaux07"`` (default) is the DIPY
        standard.
    sh_order : int
        Maximum SH order (default 8).
    sh_scale : float
        Initial per-glyph scale (default 0.4).
    lut_res : int
        LUT resolution per glyph tile (default 8).
    use_hermite : bool
        Use Hermite analytic normals (default True).
    mapping_mode : str
        Billboard mapping mode (default ``"cube"``).
    bg_color : tuple
        Scene background colour (default dark blue-black).
    title : str
        Window title.
    size : tuple
        Window ``(width, height)`` in pixels.
    interactive : bool
        If False the window is rendered once and saved (stealth mode).
    out_png : str
        Path used by stealth mode to save a screenshot.
    """

    def __init__(
        self,
        images=None,
        pams=None,
        *,
        sh_coeffs=None,
        peak_dirs=None,
        peak_values=None,
        mask=None,
        basis_type="descoteaux07",
        sh_order=8,
        sh_scale=0.4,
        lut_res=8,
        use_hermite=True,
        mapping_mode="cube",
        bg_color=(0.05, 0.05, 0.10),
        title="Skyline — SH + Volume + Peaks",
        size=(1400, 900),
        interactive=True,
        out_png="skyline.png",
    ):
        self.images = images or []
        self.pams = pams or []

        self._sh_coeffs = sh_coeffs
        self._peak_dirs = peak_dirs
        self._peak_values = peak_values
        self._mask = mask
        self._basis_type = basis_type
        self._sh_order = sh_order
        self._sh_scale = sh_scale
        self._lut_res = lut_res
        self._use_hermite = use_hermite
        self._mapping_mode = mapping_mode
        self._bg_color = bg_color
        self._title = title
        self._size = size
        self._interactive = interactive
        self._out_png = out_png

        self._resolve_data()

    def _resolve_data(self):
        """Extract SH / peaks / volume from PAMs and images."""
        if self._sh_coeffs is None and self.pams:
            pam_obj = self.pams[0][0]
            if hasattr(pam_obj, "shm_coeff") and pam_obj.shm_coeff is not None:
                self._sh_coeffs = pam_obj.shm_coeff.astype(np.float32)

        if self._peak_dirs is None and self.pams:
            pam_obj = self.pams[0][0]
            if hasattr(pam_obj, "peak_dirs") and pam_obj.peak_dirs is not None:
                self._peak_dirs = pam_obj.peak_dirs.astype(np.float32)
            if hasattr(pam_obj, "peak_values") and pam_obj.peak_values is not None:
                self._peak_values = pam_obj.peak_values.astype(np.float32)

        if self._mask is None and self._sh_coeffs is not None:
            self._mask = np.any(self._sh_coeffs != 0, axis=-1)

        self._vol_data = None
        self._affine = None
        for img in self.images:
            data, affine = img[0], img[1]
            if data.ndim == 3:
                self._vol_data = data.astype(np.float32)
                self._affine = affine
                break
            elif data.ndim == 4:
                self._vol_data = np.mean(data, axis=-1).astype(np.float32)
                self._affine = affine
                break

        if self._vol_data is None and self._sh_coeffs is not None:
            self._vol_data = np.abs(self._sh_coeffs[..., 0])
            self._vol_data = np.clip(
                self._vol_data / (self._vol_data.max() + 1e-8), 0, 1
            ).astype(np.float32)

        if self._affine is not None:
            self._voxel_sizes = tuple(
                np.sqrt(np.sum(self._affine[:3, :3] ** 2, axis=0))
            )
        else:
            self._voxel_sizes = (1.0, 1.0, 1.0)

        if self._sh_coeffs is not None:
            self._roi_shape = self._sh_coeffs.shape[:3]
        elif self._vol_data is not None:
            self._roi_shape = self._vol_data.shape[:3]
        else:
            self._roi_shape = None

    def build_scene(self):
        """Construct the FURY v2 scene and all layer actors."""
        from fury import window

        self._scene = window.Scene()
        self._scene.background = self._bg_color

        if self._roi_shape is None:
            raise ValueError(
                "Skyline needs at least one of: sh_coeffs, a PAM with "
                "shm_coeff, or a 3-D NIfTI image."
            )

        self._build_layers()
        self._gs.add_to_scene(self._scene)
        return self._scene

    def _build_layers(self):
        """Instantiate the three-layer GlobalSlicer."""
        from fury import actor

        from dipy.viz.sh_glyph import sh_slicer

        nx, ny, nz = self._roi_shape
        x0, y0, z0 = nx // 2, ny // 2, nz // 2

        has_sh = self._sh_coeffs is not None
        has_vol = self._vol_data is not None
        has_peaks = self._peak_dirs is not None

        self._sh_slicer_obj = None
        self._sh_group = None
        self._vol_group = None
        self._peaks_actor = None

        if has_sh:
            sh = self._sh_coeffs.copy()
            mx = np.abs(sh[..., 0]).max()
            if mx > 0:
                sh = sh / mx * 0.5

            print(f"Building SH slicer ({nx}x{ny}x{nz}) ...")
            t0 = time.time()
            self._sh_slicer_obj = sh_slicer(
                sh,
                voxel_sizes=self._voxel_sizes,
                scale=self._sh_scale,
                l_max=self._sh_order,
                spacing=1.0,
                lut_res=self._lut_res,
                use_hermite=self._use_hermite,
                mapping_mode=self._mapping_mode,
                mask=self._mask,
                basis_type=self._basis_type,
            )
            self._sh_group = self._sh_slicer_obj.build(axes="xyz")
            dt = time.time() - t0
            print(
                f"  SH built in {dt:.1f}s  "
                f"({self._sh_slicer_obj.get_total_glyphs()} slice actors)"
            )

        if has_vol:
            print("Building volume slicer ...")
            self._vol_group = actor.data_slicer(
                self._vol_data,
                initial_slices=(x0, y0, z0),
                visibility=(True, True, True),
                interpolation="linear",
                opacity=0.6,
            )

        if has_peaks:
            print("Building peaks slicer ...")
            pv = self._peak_values
            if pv is None:
                pv = np.ones(
                    self._peak_dirs.shape[:-1], dtype=np.float32
                ) * 0.5
            self._peaks_actor = actor.peaks_slicer(
                self._peak_dirs,
                peak_values=pv,
                visibility=(True, True, True),
                opacity=1.0,
            )
            self._peaks_actor.cross_section = np.array(
                [x0, y0, z0], dtype=np.int32
            )

        self._gs = _LayerBundle(
            roi_shape=self._roi_shape,
            sh_slicer=self._sh_slicer_obj,
            sh_group=self._sh_group,
            vol_group=self._vol_group,
            peaks_actor=self._peaks_actor,
            sh_scale=self._sh_scale,
        )

    def build_show(self, scene=None):
        """Create the ShowManager with imgui and start the event loop.

        Parameters
        ----------
        scene : fury.window.Scene, optional
            If None the scene from :meth:`build_scene` is used.

        Returns
        -------
        sm : ShowManager or None
            Returned only when ``interactive=False``.
        """
        from fury import window

        if scene is None:
            scene = self._scene

        gui_fn = _build_imgui(self._gs)

        sm = window.ShowManager(
            scene=scene,
            size=self._size,
            title=self._title,
            show_fps=True,
            imgui=True,
            imgui_draw_function=gui_fn,
        )

        if self._interactive:
            sm.start()
        else:
            from fury.window import snapshot

            snapshot(scene=scene, fname=self._out_png)
            return sm


class _LayerBundle:
    """Coordinates slice indices across SH / volume / peaks layers.

    Internal helper used by :class:`Skyline`.  Keeps the same API surface
    as ``GlobalSlicer`` in the standalone example but accepts optional
    layers (any of the three may be *None*).
    """

    def __init__(
        self,
        roi_shape,
        sh_slicer=None,
        sh_group=None,
        vol_group=None,
        peaks_actor=None,
        sh_scale=0.4,
    ):
        self.shape = roi_shape
        nx, ny, nz = self.shape

        self.x_idx = nx // 2
        self.y_idx = ny // 2
        self.z_idx = nz // 2
        self.show_x = True
        self.show_y = True
        self.show_z = True

        self.show_sh = sh_group is not None
        self.show_vol = vol_group is not None
        self.show_peaks = peaks_actor is not None
        self.has_sh = sh_group is not None
        self.has_vol = vol_group is not None
        self.has_peaks = peaks_actor is not None

        self.sh_scale = sh_scale
        self.sh_opacity = 1.0
        self.vol_opacity = 0.6
        self.peaks_opacity = 1.0

        self._sh_slicer = sh_slicer
        self._sh_group = sh_group
        self._vol_group = vol_group
        self._peaks_actor = peaks_actor

        self._sync()

    def add_to_scene(self, scene):
        """Add all available layers to the scene."""
        if self._sh_group is not None:
            scene.add(self._sh_group)
        if self._vol_group is not None:
            scene.add(self._vol_group)
        if self._peaks_actor is not None:
            scene.add(self._peaks_actor)

    def _sync(self):
        """Push current state to all layers."""
        if self._sh_slicer is not None:
            vis = (
                self.show_x and self.show_sh,
                self.show_y and self.show_sh,
                self.show_z and self.show_sh,
            )
            if vis[0]:
                self._sh_slicer.set_x_slice(self.x_idx)
            else:
                self._sh_slicer.hide_all("x")
            if vis[1]:
                self._sh_slicer.set_y_slice(self.y_idx)
            else:
                self._sh_slicer.hide_all("y")
            if vis[2]:
                self._sh_slicer.set_z_slice(self.z_idx)
            else:
                self._sh_slicer.hide_all("z")

        if self._vol_group is not None:
            from fury import actor

            vol_vis = (
                self.show_x and self.show_vol,
                self.show_y and self.show_vol,
                self.show_z and self.show_vol,
            )
            for i, child in enumerate(self._vol_group.children):
                child.visible = vol_vis[i]
            actor.show_slices(
                self._vol_group,
                (self.x_idx, self.y_idx, self.z_idx),
            )

        if self._peaks_actor is not None:
            peaks_vis = (
                self.show_x and self.show_peaks,
                self.show_y and self.show_peaks,
                self.show_z and self.show_peaks,
            )
            self._peaks_actor.visibility = peaks_vis
            self._peaks_actor.cross_section = np.array(
                [self.x_idx, self.y_idx, self.z_idx], dtype=np.int32
            )

        self._apply_opacity()

    def _apply_opacity(self):
        """Push per-layer opacity to all actors."""
        if self._sh_slicer is not None:
            for actors in (
                self._sh_slicer._x_slice_actors,
                self._sh_slicer._y_slice_actors,
                self._sh_slicer._z_slice_actors,
            ):
                for a in actors:
                    if a is not None:
                        a.material.opacity = self.sh_opacity

        if self._vol_group is not None:
            for child in self._vol_group.children:
                child.material.opacity = self.vol_opacity

        if self._peaks_actor is not None:
            self._peaks_actor.material.opacity = self.peaks_opacity

    def _apply_sh_scale(self):
        """Set per-glyph scale on every SH slice actor."""
        if self._sh_slicer is None:
            return
        for actors in (
            self._sh_slicer._x_slice_actors,
            self._sh_slicer._y_slice_actors,
            self._sh_slicer._z_slice_actors,
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


def _build_imgui(gs):
    """Return the imgui draw function for the layer bundle."""
    from imgui_bundle import imgui

    nx, ny, nz = gs.shape

    def gui():
        changed = False

        if imgui.begin("Skyline"):
            imgui.text(f"Volume {nx} x {ny} x {nz}")
            imgui.separator()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.6, 0.2, 0.2, 0.8)
            )
            if imgui.collapsing_header(
                "Slice Indices", imgui.TreeNodeFlags_.default_open
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
                imgui.Col_.header, imgui.ImVec4(0.2, 0.6, 0.2, 0.8)
            )
            if imgui.collapsing_header(
                "Axis Visibility", imgui.TreeNodeFlags_.default_open
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
                imgui.Col_.header, imgui.ImVec4(0.2, 0.2, 0.6, 0.8)
            )
            if imgui.collapsing_header(
                "Layer Visibility", imgui.TreeNodeFlags_.default_open
            ):
                imgui.pop_style_color()

                if gs.has_sh:
                    c, v = imgui.checkbox("SH Glyphs", gs.show_sh)
                    if c:
                        gs.show_sh = v
                        changed = True

                if gs.has_vol:
                    c, v = imgui.checkbox("Volume", gs.show_vol)
                    if c:
                        gs.show_vol = v
                        changed = True

                if gs.has_peaks:
                    c, v = imgui.checkbox("Peaks", gs.show_peaks)
                    if c:
                        gs.show_peaks = v
                        changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()
            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.5, 0.4, 0.2, 0.8)
            )
            if imgui.collapsing_header(
                "Settings", imgui.TreeNodeFlags_.default_open
            ):
                imgui.pop_style_color()

                if gs.has_sh:
                    c, v = imgui.slider_float(
                        "SH Scale", gs.sh_scale, 0.05, 2.0, "%.2f"
                    )
                    if c:
                        gs.sh_scale = v
                        gs._apply_sh_scale()

                imgui.spacing()
                imgui.text("Opacity")

                if gs.has_sh:
                    c, v = imgui.slider_float(
                        "SH Glyphs##op", gs.sh_opacity, 0.0, 1.0, "%.2f"
                    )
                    if c:
                        gs.sh_opacity = v
                        changed = True

                if gs.has_vol:
                    c, v = imgui.slider_float(
                        "Volume##op", gs.vol_opacity, 0.0, 1.0, "%.2f"
                    )
                    if c:
                        gs.vol_opacity = v
                        changed = True

                if gs.has_peaks:
                    c, v = imgui.slider_float(
                        "Peaks##op", gs.peaks_opacity, 0.0, 1.0, "%.2f"
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


def skyline(
    images=None,
    pams=None,
    *,
    sh_coeffs=None,
    peak_dirs=None,
    peak_values=None,
    mask=None,
    basis_type="descoteaux07",
    sh_order=8,
    sh_scale=0.4,
    lut_res=8,
    use_hermite=True,
    mapping_mode="cube",
    bg_color=(0.05, 0.05, 0.10),
    title="Skyline — SH + Volume + Peaks",
    size=(1400, 900),
    interactive=True,
    out_png="skyline.png",
    return_showm=False,
):
    """Interactive SH billboard slicer — Invert the Skyline!

    Convenience function that creates a :class:`Skyline` instance,
    builds the scene and starts the event loop.

    Parameters
    ----------
    images : list of (ndarray, ndarray, str), optional
        Each entry is ``(data_3d, affine_4x4, label)``.
    pams : list of (PeakAndMetrics, str), optional
        PAM objects with ``shm_coeff`` and ``peak_dirs``.
    sh_coeffs : ndarray, optional
        SH coefficient array ``(X, Y, Z, n_coeffs)``.
    peak_dirs : ndarray, optional
        Peak directions ``(X, Y, Z, N, 3)``.
    peak_values : ndarray, optional
        Peak values ``(X, Y, Z, N)``.
    mask : ndarray, optional
        Boolean brain mask.
    basis_type : str
        SH basis convention (default ``"descoteaux07"``).
    sh_order : int
        Maximum SH order (default 8).
    sh_scale : float
        Initial per-glyph scale (default 0.4).
    lut_res : int
        LUT resolution (default 8).
    use_hermite : bool
        Use Hermite analytic normals (default True).
    mapping_mode : str
        Billboard mapping mode (default ``"cube"``).
    bg_color : tuple
        Scene background colour.
    title : str
        Window title.
    size : tuple
        Window ``(width, height)``.
    interactive : bool
        If False render once and save to *out_png*.
    out_png : str
        Stealth-mode screenshot path.
    return_showm : bool
        If True return the ``ShowManager`` instead of blocking.

    Returns
    -------
    sm : ShowManager or None
        Returned only when *return_showm* is True.
    """
    sk = Skyline(
        images=images,
        pams=pams,
        sh_coeffs=sh_coeffs,
        peak_dirs=peak_dirs,
        peak_values=peak_values,
        mask=mask,
        basis_type=basis_type,
        sh_order=sh_order,
        sh_scale=sh_scale,
        lut_res=lut_res,
        use_hermite=use_hermite,
        mapping_mode=mapping_mode,
        bg_color=bg_color,
        title=title,
        size=size,
        interactive=interactive,
        out_png=out_png,
    )
    scene = sk.build_scene()
    result = sk.build_show(scene)
    if return_showm:
        return result
