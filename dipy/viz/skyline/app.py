from pathlib import Path

from fury.actor import Actor, show_slices
from fury.colormap import distinguishable_colormap
from fury.io import load_image_as_wgpu_texture_view

from dipy.viz.skyline.UI.manager import UIWindow
from dipy.viz.skyline.UI.theme import LOGO
from dipy.viz.skyline.render.image import Image3D
from dipy.viz.skyline.render.peak import Peak3D
from dipy.viz.skyline.render.renderer import create_window
from dipy.viz.skyline.render.roi import ROI3D
from dipy.viz.skyline.render.sh_slicer import SHGlyph3D
from dipy.viz.skyline.render.streamline import ClusterStreamline3D, Streamline3D
from dipy.viz.skyline.render.surface import Surface


class Skyline:
    def __init__(
        self,
        visualizer_type="standalone",
        images=None,
        peaks=None,
        rois=None,
        surfaces=None,
        tractograms=None,
        sh_coeffs=None,
        is_cluster=False,
        is_light_version=False,
    ):
        self.size = (1200, 1000)
        self.ui_size = (400, self.size[1])

        self.window = create_window(
            visualizer_type=visualizer_type,
            size=self.size,
            screen_config=[
                (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1]),
            ],
        )

        self._image_visualizations = []
        self._peak_visualizations = []
        self._roi_visualizations = []
        self._surface_visualizations = []
        self._tractogram_visualizations = []
        self._sh_glyph_visualizations = []
        gpu_texture = load_image_as_wgpu_texture_view(str(LOGO), self.window.device)
        logo_tex_ref = self.window._imgui.backend.register_texture(gpu_texture)
        self.window.renderer.add_event_handler(self.handle_key_events, "key_down")

        self.UI_window = UIWindow(
            "Image Controls",
            size=self.ui_size,
            render_callback=self.before_render,
            logo_tex_ref=logo_tex_ref,
        )
        self.window.resize_callback(self.handle_resize)
        self._color_gen = distinguishable_colormap()

        if images is not None:
            for idx, (img, affine, path) in enumerate(images):
                fname = Path(path).name if path is not None else f"Image {idx}"
                image3d = Image3D(
                    fname,
                    img,
                    affine=affine,
                    render_callback=self.before_render,
                    interpolation="linear",
                )
                self._image_visualizations.append(image3d)
                self.UI_window.add(fname, image3d.renderer)
        if peaks is not None:
            for idx, (pam, path) in enumerate(peaks):
                fname = Path(path).name if path is not None else f"Peaks {idx}"
                peak3d = Peak3D(
                    fname,
                    pam.peak_dirs,
                    affine=pam.affine,
                    render_callback=self.before_render,
                )
                self._peak_visualizations.append(peak3d)
                self.UI_window.add(fname, peak3d.renderer)

                # If PAM has SH coefficients, also create SH glyph slicer
                if hasattr(pam, "shm_coeff") and pam.shm_coeff is not None:
                    sh_name = f"SH Glyphs ({fname})"
                    sh3d = SHGlyph3D(
                        sh_name,
                        pam.shm_coeff,
                        affine=pam.affine,
                        render_callback=self.before_render,
                    )
                    self._sh_glyph_visualizations.append(sh3d)
                    self.UI_window.add(sh_name, sh3d.renderer)
        if sh_coeffs is not None:
            for idx, item in enumerate(sh_coeffs):
                # Support 3-tuple (coeffs, affine, name) or
                # 4-tuple (coeffs, affine, name, basis_type)
                if len(item) == 4:
                    coeffs, affine, path, basis_type = item
                else:
                    coeffs, affine, path = item
                    basis_type = "standard"
                fname = Path(path).name if path is not None else f"SH Glyphs {idx}"
                sh_name = f"SH Glyphs ({fname})"
                sh3d = SHGlyph3D(
                    sh_name,
                    coeffs,
                    affine=affine,
                    render_callback=self.before_render,
                    basis_type=basis_type,
                )
                self._sh_glyph_visualizations.append(sh3d)
                self.UI_window.add(sh_name, sh3d.renderer)
        if rois is not None:
            for idx, (roi, affine, path) in enumerate(rois):
                color = next(self._color_gen)
                fname = Path(path).name if path is not None else f"ROI {idx}"
                roi3d = ROI3D(
                    fname,
                    roi,
                    affine=affine,
                    color=color,
                    render_callback=self.before_render,
                )
                self._roi_visualizations.append(roi3d)
                self.UI_window.add(fname, roi3d.renderer)
        if surfaces is not None:
            for idx, (verts, faces, path) in enumerate(surfaces):
                color = next(self._color_gen)
                fname = Path(path).name if path is not None else f"Surface {idx}"
                surface3d = Surface(
                    fname,
                    verts,
                    faces,
                    color=color,
                    render_callback=self.before_render,
                )
                self._surface_visualizations.append(surface3d)
                self.UI_window.add(fname, surface3d.renderer)
        if tractograms is not None:
            if is_cluster:
                for idx, (sft, path) in enumerate(tractograms):
                    fname = Path(path).name if path is not None else f"Tractogram {idx}"
                    streamline3d = ClusterStreamline3D(
                        fname,
                        sft,
                        thr=20,
                        line_type="line" if is_light_version else "tube",
                        render_callback=self.before_render,
                        colormap=self._color_gen,
                    )
                    self._tractogram_visualizations.append(streamline3d)
                    self.UI_window.add(fname, streamline3d.renderer)
            else:
                for idx, (sft, path) in enumerate(tractograms):
                    color = next(self._color_gen)
                    fname = Path(path).name if path is not None else f"Tractogram {idx}"
                    streamline3d = Streamline3D(
                        fname,
                        sft,
                        line_type="line" if is_light_version else "tube",
                        color=color,
                        render_callback=self.before_render,
                    )
                    self._tractogram_visualizations.append(streamline3d)
                    self.UI_window.add(fname, streamline3d.renderer)
        self.active_image = None
        if self._image_visualizations:
            self._image_visualizations[-1].active = True
            self.active_image = self._image_visualizations[-1]
            self._arrange_image_actors()

        # Link SH glyph layers to the active image for slice syncing
        if self.active_image is not None:
            for sh_viz in self._sh_glyph_visualizations:
                sh_viz.set_image_ref(self.active_image)

        self.window._imgui.set_gui(self.draw_ui)
        self.before_render()
        self.window.start()

    def _refresh_actors(self):
        all_actors = [v.actor for v in self.visualizations]

        for actor in list(self.window.screens[0].scene.main_scene.children):
            if not isinstance(actor, Actor):
                continue
            if not any(a == actor for a in all_actors):
                self.window.screens[0].scene.main_scene.remove(actor)
        for a in all_actors:
            if a not in self.window.screens[0].scene.main_scene.children:
                self.window.screens[0].scene.main_scene.add(a)

    def _refresh_ui(self):
        for viz in self.visualizations:
            if viz.name not in self.UI_window.sections:
                self.visualizations.remove(viz)

    def _arrange_image_actors(self):
        for viz in self._image_visualizations:
            if viz.active:
                show_slices(
                    self.active_image.actor,
                    self.active_image.state,
                )
                self.active_image = viz
        show_slices(
            self.active_image.actor,
            self.active_image.state + (len(self._image_visualizations) * 0.1),
        )

    def draw_ui(self):
        self.UI_window.render()
        self.active_image and self._arrange_image_actors()

    def before_render(self):
        self._refresh_ui()
        self._refresh_actors()
        self.window.render()

    def handle_resize(self, size):
        self.size = size
        self.ui_size = (400, self.size[1])
        self.UI_window.size = (self.ui_size[0], size[1])
        self.window._screen_config = [
            (self.ui_size[0], 0, self.size[0] - self.ui_size[0], self.size[1])
        ]
        self.UI_window.size = (self.ui_size[0], size[1])
        self.window.render()

    def handle_key_events(self, event):
        for viz in self._tractogram_visualizations:
            if isinstance(viz, ClusterStreamline3D):
                viz.handle_key_events(event)

    @property
    def visualizations(self):
        return (
            self._image_visualizations
            + self._peak_visualizations
            + self._sh_glyph_visualizations
            + self._roi_visualizations
            + self._surface_visualizations
            + self._tractogram_visualizations
        )


def skyline(
    *,
    visualizer_type="standalone",
    images=None,
    peaks=None,
    rois=None,
    surfaces=None,
    tractograms=None,
    sh_coeffs=None,
    is_cluster=False,
):
    """Launch Skyline GUI.

    Parameters
    ----------
    visualizer_type : str, optional
        Type of visualizer to create. The options are:
        - "standalone": A standalone window with full interactivity.
        - "gui": A Qt-based GUI window.
        - "jupyter": An inline Jupyter notebook visualizer.
        - "stealth": An offscreen visualizer without GUI.
    images : list, optional
        List of (data, affine, name) tuples for volume images.
    peaks : list, optional
        List of (pam, name) tuples for peak visualizations.
    rois : list, optional
        List of (data, affine, name) tuples for ROIs.
    surfaces : list, optional
        List of (vertices, faces, name) tuples for surfaces.
    tractograms : list, optional
        List of (sft, name) tuples for tractograms.
    sh_coeffs : list, optional
        List of (coeffs_4d, affine, name) tuples for SH glyph slicers.
    is_cluster : bool, optional
        Whether to cluster the tractograms.
    """
    Skyline(
        visualizer_type=visualizer_type,
        images=images,
        peaks=peaks,
        rois=rois,
        surfaces=surfaces,
        tractograms=tractograms,
        sh_coeffs=sh_coeffs,
        is_cluster=is_cluster,
    )
