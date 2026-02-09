"""
=====================================================
Visualize SH Glyphs with Hermite Billboard Rendering
=====================================================

Interactive SH glyph viewer with ImGui controls.

Three modes:
- ``realtime``: per-fragment SH evaluation
- ``lut``: precomputed LUT with Hermite interpolation
- ``slicer``: volume slicer with per-axis range controls
"""

from fury import window
from imgui_bundle import imgui
import numpy as np

from dipy.viz.sh_glyph import (
    descoteaux07_to_standard,
    sh_glyph_billboard,
    sh_glyph_billboard_lut,
    sh_slicer,
)


def _synthetic_coeffs(n_glyphs=64, l_max=4):
    n_coeffs = (l_max + 1) ** 2
    coeffs = np.zeros((n_glyphs, n_coeffs), dtype=np.float32)
    for i in range(n_glyphs):
        phase = i * 0.1
        coeffs[i, 0] = 0.5 + 0.2 * np.sin(phase)
        for big_l in range(1, l_max + 1):
            strength = 0.3 / big_l
            m = (i % (2 * big_l + 1)) - big_l
            idx = big_l * big_l + big_l + m
            if 0 <= idx < n_coeffs:
                coeffs[i, idx] = strength * np.sin(phase + big_l + m)
    return coeffs


def _grid_positions(n, spacing=2.5):
    g = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pos[i, 0] = (i % g - g / 2) * spacing
        pos[i, 1] = (i // g - g / 2) * spacing
    return pos


def demo_realtime(n_glyphs=64, l_max=4, scale=2.0):
    coeffs = _synthetic_coeffs(n_glyphs, l_max)
    centers = _grid_positions(n_glyphs)

    actor = sh_glyph_billboard(
        coeffs, centers=centers, l_max=l_max, scale=scale,
    )

    scene = window.Scene()
    scene.background = (0.05, 0.05, 0.1)
    scene.add(actor)

    state = {"scale": scale}

    def gui():
        if imgui.begin("SH Billboard — Real-time"):
            imgui.text(f"{n_glyphs} glyphs, L={l_max}")
            imgui.separator()
            _, state["scale"] = imgui.slider_float(
                "Scale", state["scale"], 0.1, 5.0, "%.2f",
            )
            actor.local.scale = (
                state["scale"], state["scale"], state["scale"],
            )
        imgui.end()

    sm = window.ShowManager(
        scene=scene,
        size=(1200, 800),
        title=f"Real-time SH ({n_glyphs} glyphs, L={l_max})",
        show_fps=True,
        imgui=True,
        imgui_draw_function=gui,
    )
    sm.start()


def demo_lut_hermite(n_glyphs=256, l_max=4, scale=2.0, lut_res=64):
    coeffs = _synthetic_coeffs(n_glyphs, l_max)
    centers = _grid_positions(n_glyphs)

    actors = sh_glyph_billboard_lut(
        coeffs,
        centers=centers,
        l_max=l_max,
        scale=scale,
        lut_res=lut_res,
        mapping_mode="octahedral",
        use_hermite=True,
        interpolation_mode=2,
    )

    scene = window.Scene()
    scene.background = (0.05, 0.05, 0.1)
    for a in actors:
        scene.add(a)

    state = {"scale": scale}

    def gui():
        if imgui.begin("SH Billboard — LUT + Hermite"):
            imgui.text(f"{n_glyphs} glyphs, L={l_max}, res={lut_res}")
            imgui.separator()
            _, state["scale"] = imgui.slider_float(
                "Scale", state["scale"], 0.1, 5.0, "%.2f",
            )
            for a in actors:
                a.local.scale = (
                    state["scale"], state["scale"], state["scale"],
                )
        imgui.end()

    sm = window.ShowManager(
        scene=scene,
        size=(1200, 800),
        title=f"LUT+Hermite ({n_glyphs} glyphs, L={l_max})",
        show_fps=True,
        imgui=True,
        imgui_draw_function=gui,
    )
    sm.start()


def demo_slicer(roi_size=10, scale=2.0, lut_res=64):
    try:
        from dipy.data import fetch_stanford_hardi, read_stanford_hardi
        from dipy.reconst.csdeconv import (
            ConstrainedSphericalDeconvModel,
            auto_response_ssst,
        )
        from dipy.segment.mask import median_otsu

        print("Fetching Stanford HARDI ...")
        fetch_stanford_hardi()
        img, gtab = read_stanford_hardi()
        data = img.get_fdata()
        affine = img.affine

        maskdata, mask = median_otsu(
            data, vol_idx=range(10, 50), median_radius=3, numpass=1,
        )

        mc = [np.where(mask)[i] for i in range(3)]
        cx, cy, cz = [int(np.mean(c)) for c in mc]
        h = roi_size // 2
        sl = tuple(
            slice(max(0, c - h), min(s, c + h))
            for c, s in zip((cx, cy, cz), data.shape[:3])
        )
        roi_data = data[sl + (slice(None),)]
        roi_mask = mask[sl]

        response, _ = auto_response_ssst(
            gtab, data, roi_radii=10, fa_thr=0.7,
        )
        csd_model = ConstrainedSphericalDeconvModel(
            gtab, response, sh_order_max=8,
        )
        csd_fit = csd_model.fit(roi_data, mask=roi_mask)

        coeffs_4d = descoteaux07_to_standard(
            csd_fit.shm_coeff.astype(np.float32), l_max=8,
        )
        max_val = np.abs(coeffs_4d[..., 0]).max()
        if max_val > 0:
            coeffs_4d = coeffs_4d / max_val * 0.5

        voxel_sizes = tuple(
            np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)),
        )
    except Exception as e:
        print(f"Could not load dMRI data ({e}), using synthetic volume.")
        shape = (roi_size, roi_size, roi_size)
        n_coeffs = 81
        coeffs_4d = np.random.randn(*shape, n_coeffs).astype(np.float32)
        coeffs_4d *= 0.1
        coeffs_4d[..., 0] = 0.5
        voxel_sizes = (1.0, 1.0, 1.0)

    slicer_obj = sh_slicer(
        coeffs_4d,
        voxel_sizes=voxel_sizes,
        scale=scale,
        lut_res=lut_res,
        use_hermite=True,
        mapping_mode="octahedral",
    )
    group = slicer_obj.build(axes="xyz")

    scene = window.Scene()
    scene.background = (0.05, 0.05, 0.1)
    scene.add(group)

    nx, ny, nz = coeffs_4d.shape[:3]

    slicer_obj.hide_all("x")
    slicer_obj.hide_all("y")
    slicer_obj.set_z_range(nz // 2, nz // 2 + 1)

    state = {
        "show_x": False, "x_min": 0, "x_max": nx,
        "show_y": False, "y_min": 0, "y_max": ny,
        "show_z": True, "z_min": nz // 2, "z_max": nz // 2 + 1,
        "cross_x": nx // 2, "cross_y": ny // 2, "cross_z": nz // 2,
    }

    def _apply():
        if state["show_x"]:
            slicer_obj.set_x_range(state["x_min"], state["x_max"])
        else:
            slicer_obj.hide_all("x")

        if state["show_y"]:
            slicer_obj.set_y_range(state["y_min"], state["y_max"])
        else:
            slicer_obj.hide_all("y")

        if state["show_z"]:
            slicer_obj.set_z_range(state["z_min"], state["z_max"])
        else:
            slicer_obj.hide_all("z")

    def gui():
        changed = False
        if imgui.begin("SH Slicer"):
            imgui.text(f"Volume: {nx} x {ny} x {nz}")
            imgui.separator()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.8, 0.2, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "X Axis", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()
                c, state["show_x"] = imgui.checkbox(
                    "Show X", state["show_x"],
                )
                changed = changed or c
                c, state["x_min"] = imgui.slider_int(
                    "X Min", state["x_min"], 0, nx - 1,
                )
                changed = changed or c
                c, state["x_max"] = imgui.slider_int(
                    "X Max", state["x_max"], 1, nx,
                )
                changed = changed or c
                if state["x_min"] >= state["x_max"]:
                    state["x_min"] = max(0, state["x_max"] - 1)
                if imgui.button("All X"):
                    state["x_min"] = 0
                    state["x_max"] = nx
                    state["show_x"] = True
                    changed = True
                imgui.same_line()
                if imgui.button("Mid X"):
                    mid = nx // 2
                    state["x_min"] = mid
                    state["x_max"] = mid + 1
                    state["show_x"] = True
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.2, 0.8, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "Y Axis", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()
                c, state["show_y"] = imgui.checkbox(
                    "Show Y", state["show_y"],
                )
                changed = changed or c
                c, state["y_min"] = imgui.slider_int(
                    "Y Min", state["y_min"], 0, ny - 1,
                )
                changed = changed or c
                c, state["y_max"] = imgui.slider_int(
                    "Y Max", state["y_max"], 1, ny,
                )
                changed = changed or c
                if state["y_min"] >= state["y_max"]:
                    state["y_min"] = max(0, state["y_max"] - 1)
                if imgui.button("All Y"):
                    state["y_min"] = 0
                    state["y_max"] = ny
                    state["show_y"] = True
                    changed = True
                imgui.same_line()
                if imgui.button("Mid Y"):
                    mid = ny // 2
                    state["y_min"] = mid
                    state["y_max"] = mid + 1
                    state["show_y"] = True
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.2, 0.2, 0.8, 0.8),
            )
            if imgui.collapsing_header(
                "Z Axis", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()
                c, state["show_z"] = imgui.checkbox(
                    "Show Z", state["show_z"],
                )
                changed = changed or c
                c, state["z_min"] = imgui.slider_int(
                    "Z Min", state["z_min"], 0, nz - 1,
                )
                changed = changed or c
                c, state["z_max"] = imgui.slider_int(
                    "Z Max", state["z_max"], 1, nz,
                )
                changed = changed or c
                if state["z_min"] >= state["z_max"]:
                    state["z_min"] = max(0, state["z_max"] - 1)
                if imgui.button("All Z"):
                    state["z_min"] = 0
                    state["z_max"] = nz
                    state["show_z"] = True
                    changed = True
                imgui.same_line()
                if imgui.button("Mid Z"):
                    mid = nz // 2
                    state["z_min"] = mid
                    state["z_max"] = mid + 1
                    state["show_z"] = True
                    changed = True
            else:
                imgui.pop_style_color()

            imgui.spacing()
            imgui.separator()

            imgui.push_style_color(
                imgui.Col_.header, imgui.ImVec4(0.8, 0.6, 0.2, 0.8),
            )
            if imgui.collapsing_header(
                "Crosshair", imgui.TreeNodeFlags_.default_open,
            ):
                imgui.pop_style_color()
                imgui.text_colored(
                    imgui.ImVec4(0.8, 0.8, 0.5, 1.0),
                    "Each slider sets other axes to their crosshair",
                )
                imgui.spacing()

                c, new_x = imgui.slider_int(
                    "##CrossX", state["cross_x"], 0, nx - 1,
                )
                if c:
                    state["cross_x"] = new_x
                    state["x_min"] = new_x
                    state["x_max"] = new_x + 1
                    state["show_x"] = True
                    state["y_min"] = state["cross_y"]
                    state["y_max"] = state["cross_y"] + 1
                    state["show_y"] = True
                    state["z_min"] = state["cross_z"]
                    state["z_max"] = state["cross_z"] + 1
                    state["show_z"] = True
                    changed = True

                c, new_y = imgui.slider_int(
                    "##CrossY", state["cross_y"], 0, ny - 1,
                )
                if c:
                    state["cross_y"] = new_y
                    state["y_min"] = new_y
                    state["y_max"] = new_y + 1
                    state["show_y"] = True
                    state["x_min"] = state["cross_x"]
                    state["x_max"] = state["cross_x"] + 1
                    state["show_x"] = True
                    state["z_min"] = state["cross_z"]
                    state["z_max"] = state["cross_z"] + 1
                    state["show_z"] = True
                    changed = True

                c, new_z = imgui.slider_int(
                    "##CrossZ", state["cross_z"], 0, nz - 1,
                )
                if c:
                    state["cross_z"] = new_z
                    state["z_min"] = new_z
                    state["z_max"] = new_z + 1
                    state["show_z"] = True
                    state["x_min"] = state["cross_x"]
                    state["x_max"] = state["cross_x"] + 1
                    state["show_x"] = True
                    state["y_min"] = state["cross_y"]
                    state["y_max"] = state["cross_y"] + 1
                    state["show_y"] = True
                    changed = True

                imgui.spacing()
                if imgui.button("Center All"):
                    cx, cy, cz = nx // 2, ny // 2, nz // 2
                    state["cross_x"] = cx
                    state["cross_y"] = cy
                    state["cross_z"] = cz
                    state["x_min"] = cx
                    state["x_max"] = cx + 1
                    state["y_min"] = cy
                    state["y_max"] = cy + 1
                    state["z_min"] = cz
                    state["z_max"] = cz + 1
                    state["show_x"] = True
                    state["show_y"] = True
                    state["show_z"] = True
                    changed = True
            else:
                imgui.pop_style_color()

        imgui.end()

        if changed:
            _apply()

    sm = window.ShowManager(
        scene=scene,
        size=(1200, 800),
        title="SH Slicer (Hermite)",
        show_fps=True,
        imgui=True,
        imgui_draw_function=gui,
    )
    sm.start()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="DIPY SH Billboard Demo")
    p.add_argument(
        "--mode",
        choices=["realtime", "lut", "slicer"],
        default="slicer",
        help="Rendering mode (default: slicer)",
    )
    p.add_argument("--n-glyphs", type=int, default=64)
    p.add_argument("--l-max", type=int, default=4)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--lut-res", type=int, default=64)
    p.add_argument("--roi-size", type=int, default=10)
    args = p.parse_args()

    if args.mode == "realtime":
        demo_realtime(args.n_glyphs, args.l_max, args.scale)
    elif args.mode == "lut":
        demo_lut_hermite(
            args.n_glyphs, args.l_max, args.scale, args.lut_res,
        )
    elif args.mode == "slicer":
        demo_slicer(args.roi_size, args.scale, args.lut_res)
