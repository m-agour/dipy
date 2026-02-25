#!/usr/bin/env python

import argparse
import os
import sys
import time

import numpy as np

BUILD = os.path.join(os.path.dirname(__file__), "build", "cp312")
if os.path.isdir(BUILD):
    sys.path.insert(0, BUILD)


def fit_csd_and_launch(data_dir, sh_order=8):
    import nibabel as nib

    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import (
        ConstrainedSphericalDeconvModel,
        auto_response_ssst,
    )
    from dipy.reconst.dti import TensorModel

    print("=" * 60)
    print("Fitting CSD on HCP data")
    print("=" * 60)

    print("\nLoading data...")
    t0 = time.time()
    img = nib.load(os.path.join(data_dir, "data.nii.gz"))
    data = img.get_fdata()
    affine = img.affine
    voxel_sizes = tuple(np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)))
    print(f"  Shape: {data.shape}")
    print(f"  Voxel sizes: {voxel_sizes}")

    bvals = np.loadtxt(os.path.join(data_dir, "bvals"))
    bvecs = np.loadtxt(os.path.join(data_dir, "bvecs"))
    gtab = gradient_table(bvals, bvecs)
    shells = sorted(set(int(round(b / 100) * 100) for b in bvals if b > 50))
    print(f"  Shells: {shells}")
    print(f"  Volumes: {len(bvals)}")

    print("\nLoading brain mask...")
    mask_img = nib.load(os.path.join(data_dir, "nodif_brain_mask.nii.gz"))
    mask = mask_img.get_fdata().astype(bool)
    print(f"  Mask voxels: {mask.sum():,}")

    print("\nCropping to brain bounding box...")
    coords = np.where(mask)
    slices = tuple(
        slice(c.min(), c.max() + 1) for c in coords
    )
    roi_data = data[slices + (slice(None),)]
    roi_mask = mask[slices]
    print(f"  Cropped shape: {roi_data.shape[:3]}")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nComputing auto response (ssst)...")
    t0 = time.time()
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    print(f"  Response ratio: {ratio:.3f}")
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nFitting CSD (L={sh_order})...")
    t0 = time.time()
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response, sh_order_max=sh_order
    )
    csd_fit = csd_model.fit(roi_data, mask=roi_mask)
    sh_desc = csd_fit.shm_coeff.astype(np.float32)
    print(f"  SH descoteaux shape: {sh_desc.shape}")
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\nConverting to standard SH basis...")
    n_std = (sh_order + 1) ** 2
    sh_std = np.zeros(sh_desc.shape[:-1] + (n_std,), dtype=np.float32)
    desc_idx = 0
    for l_val in range(0, sh_order + 1, 2):
        for m in range(-l_val, l_val + 1):
            std_idx = l_val * l_val + l_val + m
            sh_std[..., std_idx] = sh_desc[..., desc_idx]
            desc_idx += 1
    print(f"  Standard SH shape: {sh_std.shape}")

    print("\nFitting DTI for FA map...")
    t0 = time.time()
    dti_model = TensorModel(gtab)
    dti_fit = dti_model.fit(roi_data, mask=roi_mask)
    fa = np.clip(dti_fit.fa, 0, 1).astype(np.float32)
    fa[~roi_mask] = 0.0
    print(f"  FA range: [{fa[roi_mask].min():.3f}, {fa[roi_mask].max():.3f}]")
    print(f"  Done in {time.time() - t0:.1f}s")

    viz_affine = np.diag(list(voxel_sizes) + [1.0])

    print(f"\n  SH coeffs shape : {sh_std.shape}")
    print(f"  FA shape         : {fa.shape}")
    print(f"  Mask voxels      : {roi_mask.sum()}")
    print(f"  Voxel sizes      : {voxel_sizes}")
    print(f"  Affine:\n{viz_affine}")

    return sh_std, fa, roi_mask, viz_affine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sh-order", type=int, default=8)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "hermite", "hcp_data"),
    )
    args = parser.parse_args()

    sh_coeffs, fa, mask, affine = fit_csd_and_launch(
        args.data_dir, args.sh_order
    )

    print("\nLaunching Skyline …")
    from dipy.viz.skyline.app import skyline

    skyline(
        images=[(fa, affine, "FA (HCP)")],
        sh_coeffs=[(sh_coeffs, affine, "SH Glyphs (HCP)")],
    )


if __name__ == "__main__":
    main()
