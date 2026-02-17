from fury.actor import set_group_opacity, show_slices, volume_slicer
from fury.lib import gfx
from imgui_bundle import icons_fontawesome_6, imgui
import numpy as np

from dipy.viz.skyline.UI.elements import (
    dropdown,
    render_group,
    segmented_switch,
    thin_slider,
    two_disk_slider,
)
from dipy.viz.skyline.UI.theme import THEME, WINDOW_THEME
from dipy.viz.skyline.render.renderer import Visualization


class Image3D(Visualization):
    def __init__(
        self,
        name,
        volume,
        *,
        affine=None,
        interpolation="linear",
        render_callback=None,
        opacity=100,
        rgb=False,
        value_percentiles=(2, 98),
        colormap="Gray",
    ):
        super().__init__(name, render_callback)
        self.dwi = volume
        self.affine = affine

        if (
            rgb
            and self.dwi.ndim == 4
            and (self.dwi.shape[3] != 3 and self.dwi.shape[3] != 4)
        ):
            ValueError(
                "When specifying rgb=True, the last dimension of the volume "
                "must be 3 (RGB) or 4 (RGBA)."
            )
        self.rgb = rgb

        self._has_directions = self.dwi.ndim == 4 and not rgb

        self._volume_idx = 0
        self.interpolation = interpolation or "linear"
        self._value_percentiles = value_percentiles
        self._colormap_options = ("Gray", "Inferno", "Magma", "Plasma", "Viridis")
        self.colormap = colormap

        self._create_slicer_actor()
        self.opacity = opacity

        # Per-axis visibility (children order: 0=X, 1=Y, 2=Z)
        self._axis_visible = [True, True, True]

        self._slicer.add_event_handler(self._pick_voxel, "pointer_down")

    def _pick_voxel(self, event):
        info = event.pick_info
        intensity_interpolated = np.asarray(info["rgba"].rgb).mean()
        intensity_raw = self.dwi[info["index"]]
        print(
            f"Voxel {info['index']}:"
            f"\nInterpolated intensity: {intensity_interpolated:.2f}"
            f"\nRaw intensity: {intensity_raw}"
        )

    def _create_slicer_actor(self):
        if self._has_directions:
            volume = self.dwi[..., self._volume_idx]
            self.value_range = self._value_range_from_percentile(volume)
            self._slicer = volume_slicer(
                volume,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="blend",
                depth_write=True,
            )
        else:
            self.value_range = self._value_range_from_percentile(self.dwi)
            self._slicer = volume_slicer(
                self.dwi,
                affine=self.affine,
                interpolation=self.interpolation,
                value_range=self.value_range,
                alpha_mode="blend",
                depth_write=True,
            )
        self._apply_colormap(self.colormap)
        self.bounds = self._slicer.get_bounding_box()
        self.state = np.mean(self.bounds, axis=0)
        show_slices(self._slicer, self.state)
        self.render()

    def _value_range_from_percentile(self, volume):
        p_low, p_high = self._value_percentiles
        vmin, vmax = np.percentile(volume, (p_low, p_high))
        return vmin, vmax

    def _apply_colormap(self, colormap):
        self.colormap = colormap
        if self.colormap.lower() == "gray":
            for actor in self._slicer.children:
                actor.material.map = None
        else:
            for actor in self._slicer.children:
                actor.material.map = getattr(gfx.cm, self.colormap.lower())

    @property
    def actor(self):
        return self._slicer

    # ----- hierarchical visibility ----------------------------------------

    def renderer(self, name, is_open):
        """Override to enforce hierarchical visibility.

        When the master header eye is toggled off, all axes are hidden
        regardless of per-axis toggles.  When toggled back on, the
        per-axis state is restored.
        """
        result = super().renderer(name, is_open)
        # After the base renderer sets self.actor.visible,
        # push per-axis state into children.
        self._apply_axis_visibility()
        return result

    def _apply_axis_visibility(self):
        """Set each slice child's visibility based on master + per-axis."""
        master_on = self._slicer.visible
        for i, child in enumerate(self._slicer.children):
            child.visible = master_on and self._axis_visible[i]

    # ----- eye toggle helper (reusable) -----------------------------------

    @staticmethod
    def _eye_toggle(label, is_on):
        """Draw a small eye-icon toggle."""
        imgui.push_id(label)
        icon = (
            icons_fontawesome_6.ICON_FA_EYE
            if is_on
            else icons_fontawesome_6.ICON_FA_EYE_SLASH
        )
        text_color = THEME["text_highlight"] if is_on else THEME["text"]
        icon_size = imgui.calc_text_size(icon)
        btn_h = icon_size.y + 4
        btn_w = icon_size.x + 4

        start = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()

        imgui.invisible_button(f"eye_{label}", (btn_w, btn_h))
        clicked = imgui.is_item_clicked()
        hovered = imgui.is_item_hovered()

        col = imgui.get_color_u32(text_color)
        if hovered:
            col = imgui.get_color_u32(WINDOW_THEME["title_active_color"])
        icon_pos = (start.x + 2, start.y + (btn_h - icon_size.y) * 0.5)
        draw_list.add_text(icon_pos, col, icon)

        new_state = (not is_on) if clicked else is_on
        imgui.pop_id()
        return clicked, new_state

    def render_widgets(self):
        changed, _pointer_released, new = thin_slider(
            "Opacity",
            self.opacity,
            0,
            100,
            value_type="int",
            text_format=".0f",
            value_unit="%",
            step=1,
        )
        if changed:
            self.opacity = new
            set_group_opacity(self._slicer, self.opacity / 100.0)

        imgui.spacing()

        # ---- Per-axis slice sliders with eye toggles ----
        axis_labels = ("X", "Y", "Z")
        slider_bounds = (
            (int(self.bounds[0][0] + 1), int(self.bounds[1][0] - 1)),
            (int(self.bounds[0][1] + 1), int(self.bounds[1][1] - 1)),
            (int(self.bounds[0][2] + 1), int(self.bounds[1][2] - 1)),
        )

        label_color = THEME["text"]
        imgui.text_colored(label_color, "Slice")
        imgui.spacing()

        for axis in range(3):
            label = axis_labels[axis]
            min_bound, max_bound = slider_bounds[axis]

            # Eye toggle + slider on the same row
            eye_clicked, self._axis_visible[axis] = self._eye_toggle(
                f"slice_{label}", self._axis_visible[axis]
            )
            if eye_clicked:
                self._apply_axis_visibility()

            imgui.same_line(0, 6)

            changed, _pointer_released, new_val = thin_slider(
                label,
                self.state[axis],
                min_bound,
                max_bound,
                value_type="float",
                text_format=".0f",
                step=1,
            )
            if changed:
                self.state[axis] = new_val
            show_slices(self._slicer, self.state)
            # Re-apply per-axis visibility after show_slices
            self._apply_axis_visibility()

        imgui.spacing()
        volume_for_range = (
            self.dwi[..., self._volume_idx] if self._has_directions else self.dwi
        )
        intensity_changed, new_percentiles = two_disk_slider(
            "Intensities",
            self._value_percentiles,
            0,
            100,
            text_format=".1f",
            step=1,
            min_gap=0.1,
            display_values=self.value_range,
        )
        if intensity_changed:
            self._value_percentiles = new_percentiles
            self.value_range = self._value_range_from_percentile(volume_for_range)
            for actor in self._slicer.children:
                actor.material.clim = self.value_range

        if self._has_directions:
            imgui.spacing()
            volume_changed, _pointer_released, new_idx = thin_slider(
                "Directions",
                self._volume_idx,
                0,
                self.dwi.shape[3] - 1,
                value_type="int",
                text_format=".0f",
                step=1,
            )
            if volume_changed:
                self._volume_idx = new_idx
                self._create_slicer_actor()

        imgui.spacing()
        colormap_changed, new_cmap = dropdown(
            "Colormap", self._colormap_options, self.colormap
        )
        if colormap_changed:
            self._apply_colormap(new_cmap)

        imgui.spacing()
        imgui.spacing()
        changed, new = segmented_switch(
            "Interpolation", ["Linear", "Nearest"], self.interpolation
        )
        if changed:
            self.interpolation = new
            for actor in self._slicer.children:
                actor.material.interpolation = self.interpolation

        imgui.spacing()
