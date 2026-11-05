#!/usr/bin/env python3
"""
Post-Processing Script for MAMA-MIA Challenge.
Implements Connected Component Analysis to clean up segmentations.

Strategies:
1. keep_largest: Keep ONLY the largest connected component. (Best for single tumors).
2. remove_small: Keep largest, plus any other components larger than `min_size` voxels.
3. clinical:
   - If 'multifocal_cancer' == Yes: Keep all components > min_size.
   - If 'multifocal_cancer' == No:  Keep largest. AND keep small components if they are within `dist_threshold` voxels of the largest.

Usage:
    python apply_postprocessing.py \
        --input predictions_1000ep_full_fold1 \
        --output predictions_1000ep_pp_clinical \
        --strategy clinical \
        --clinical clinical_and_imaging_info.xlsx \
        --dist_threshold 100
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

def get_connected_components(mask):
    """
    Returns sorted list of (label_id, size)
    Sorted by size descending.
    """
    # 26-connectivity (3x3x3)
    s = generate_binary_structure(3, 3)
    labeled_array, num_features = label(mask, structure=s)

    if num_features == 0:
        return labeled_array, []

    # Measure sizes
    sizes = np.bincount(labeled_array.ravel())
    # sizes[0] is background

    components = []
    for i in range(1, len(sizes)):
        if sizes[i] > 0:
            components.append((i, sizes[i]))

    # Sort by size descending
    components.sort(key=lambda x: x[1], reverse=True)
    return labeled_array, components

def apply_lcc(mask_data, strategy="keep_largest", min_size=50, is_multifocal=False, dist_threshold=100):
    """
    Applies LCC logic with distance checking.
    """
    if np.sum(mask_data) == 0:
        return mask_data

    labeled_map, components = get_connected_components(mask_data)

    if not components:
        return mask_data

    new_mask = np.zeros_like(mask_data)

    # Identify largest
    largest_id, largest_size = components[0]

    # Always keep largest
    new_mask[labeled_map == largest_id] = 1

    # If there are other components, checking them
    if len(components) > 1:

        # Pre-compute distance transform from the largest component IF we need it
        # We need it if strategy is clinical (unifocal) or distance-based
        # dist_map: 0 at largest_component, increasing euclidean distance elsewhere
        dist_map = None

        # Optimization: Only compute if we actually enter a branch that needs it
        need_dist = (strategy == "clinical" and not is_multifocal) or (strategy == "distance")

        if need_dist:
            # Invert mask: 0 inside object, 1 outside. EDT computes dist to nearest 0.
            # So we want dist to nearest LargestComponent voxel.
            # Grid where largest is 0, rest is 1.
            binary_largest = (labeled_map == largest_id)
            # edt computes distance to background (0). So we input inverted boolean.
            # We want distance TO the True pixels.
            # scipy.ndimage.distance_transform_edt(input):
            # "Calculates the distance from non-zero (i.e. non-background) points to the nearest background point."
            # Wait, commonly used: dist to nearest feature.
            # If we want dist TO component A. We set A=0, rest=1. EDT gives dist to A.
            input_grid = np.ones_like(labeled_map)
            input_grid[binary_largest] = 0
            dist_map = distance_transform_edt(input_grid)

        # Loop over smaller components
        for comp_id, size in components[1:]:
            keep_it = False

            # --- STRATEGY LOGIC ---

            if strategy == "remove_small":
                if size >= min_size:
                    keep_it = True

            elif strategy == "keep_largest":
                keep_it = False # Strict

            elif strategy == "clinical":
                if is_multifocal:
                    # Multifocal: Just check size
                    if size >= min_size:
                        keep_it = True
                else:
                    # Single Focal: Check distance!
                    # If close to main tumor, it's likely a fragmented part of the same tumor.
                    # Get mask of this small component
                    comp_mask = (labeled_map == comp_id)
                    # Find min distance to largest
                    # min_d = np.min(dist_map[comp_mask])
                    # Optimization: slice it out to avoid full array indexing
                    # but dist_map matches shape.
                    # using boolean index on large array is okay if sparse?
                    # actually scipy ndimage uses C backend, it's fast.
                    min_d = np.min(dist_map[comp_mask])

                    if min_d <= dist_threshold:
                        keep_it = True

            if keep_it:
                new_mask[labeled_map == comp_id] = 1

    return new_mask

def process_folder(input_dir, output_dir, strategy="keep_largest", clinical_info=None, min_size=50, dist_threshold=100):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

    # Load clinical if available
    multifocal_dict = {}
    if clinical_info:
        try:
            df = pd.read_excel(clinical_info, sheet_name='dataset_info')
        except:
            df = pd.read_excel(clinical_info)

        col = 'multifocal_cancer' if 'multifocal_cancer' in df.columns else None
        pid_col = 'patient_id' if 'patient_id' in df.columns else 'Case_ID'

        if col and pid_col in df.columns:
             print(f"Loaded clinical info. Found column '{col}'.")
             for idx, row in df.iterrows():
                 pid = str(row[pid_col])
                 val = row[col]
                 is_multi = False
                 if pd.notna(val):
                     s = str(val).lower()
                     if s in ['yes', 'true', '1', '1.0']:
                         is_multi = True
                 multifocal_dict[pid] = is_multi

    print(f"Processing {len(files)} files with strategy='{strategy}'...")
    if strategy == "clinical":
        print(f"  Unifocal logic: Keep components within {dist_threshold} voxels of largest.")
        print(f"  Multifocal logic: Keep all components > {min_size} voxels.")

    cnt_changed = 0

    for f in tqdm(files):
        cid = os.path.basename(f).replace(".nii.gz", "")

        nii = nib.load(f)
        data = nii.get_fdata().astype(np.uint8)
        affine = nii.affine
        header = nii.header

        original_sum = np.sum(data)

        # Modify strategy per case check ? No, apply_lcc handles "clinical" logic internally
        is_multi = multifocal_dict.get(cid, False)

        # Apply
        new_data = apply_lcc(data, strategy=strategy, min_size=min_size,
                             is_multifocal=is_multi, dist_threshold=dist_threshold)

        new_sum = np.sum(new_data)
        if new_sum != original_sum:
            cnt_changed += 1

        # Save
        new_img = nib.Nifti1Image(new_data, affine, header)
        nib.save(new_img, os.path.join(output_dir, f"{cid}.nii.gz"))

    print(f"Done. Modified {cnt_changed}/{len(files)} masks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input predictions folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--strategy", default="keep_largest", choices=["keep_largest", "remove_small", "clinical"], help="Strategy")
    parser.add_argument("--clinical", help="Path to clinical excel (required for 'clinical' strategy)")
    parser.add_argument("--min_size", type=int, default=50, help="Minimum voxel size for remove_small logic")
    parser.add_argument("--dist_threshold", type=float, default=100.0, help="Distance threshold in voxels for unifocal proximity check")
    args = parser.parse_args()

    process_folder(args.input, args.output, args.strategy, args.clinical, args.min_size, args.dist_threshold)
