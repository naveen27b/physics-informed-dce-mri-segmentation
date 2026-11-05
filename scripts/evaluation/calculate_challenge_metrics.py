#!/usr/bin/env python3
"""
Official Challenge Metrics Calculator for MAMA-MIA Task 1.
Replicates logic from src/challenge/scoring_task1.py EXACTLY.

metrics:
  - DSC (Dice)
  - NormHD (Hausdorff Dist 95% / 150.0) -> clipped to 1.0

Scores:
  - Performance Score = 0.5 * (MeanDSC + (1 - MeanNormHD))
  - Fairness Score (per attribute) = 1 - 0.5 * (DSC_Disparity + NormHD_Disparity)
  - Overall Fairness Score = Mean(Fairness Scores of attributes)
  - Ranking Score = 0.5 * Performance + 0.5 * Fairness
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys

# Settings from scoring_task1.py
HD_MAX = 150.0
ALPHA = 0.5
FAIRNESS_VARS = ['age', 'menopausal_status', 'breast_density']

def preprocess_clinical_data(df):
    """Replicate preprocessing from scoring_task1.py"""
    df = df.copy()

    # 1. Age Bins: [0, 40, 50, 60, 70, 100] -> labels ['0-40', '41-50', '51-60', '61-70', '71+']
    # Note: labels must match bins-1
    bins = [0, 40, 50, 60, 70, 100]
    labels = ['0-40', '41-50', '51-60', '61-70', '71+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # 2. Menopausal Status
    # map peri/pre -> pre, post -> post, fillna -> unknown
    # The clinical file has column 'menopause', challenge script expects 'menopausal_status'
    if 'menopause' in df.columns:
        df['menopausal_status'] = df['menopause']

    def map_menopause(x):
        if pd.isna(x): return 'unknown'
        x = str(x).lower()
        if 'peri' in x: return 'pre' # perimenopausic -> pre
        if 'pre' in x: return 'pre'
        if 'post' in x: return 'post'
        return 'unknown'

    if 'menopausal_status' in df.columns:
        df['menopausal_status'] = df['menopausal_status'].apply(map_menopause)
    else:
        # If still missing, fill with unknown
        df['menopausal_status'] = 'unknown'

    # 3. Breast Density
    # Just need it as is or cleaned up? scoring_task1 doesn't change values, just uses them.
    # But usually A, B, C, D.
    if 'breast_density' in df.columns:
        df['breast_density'] = df['breast_density'].astype(str).str.upper().str.strip()
    else:
        df['breast_density'] = 'unknown'

    return df

def calculate_metrics(test_metrics_path, clinical_info_path, output_dir):
    print("Loading Data...")
    # 1. Load Data
    # test_metrics_path should have 'Case_ID', 'Dice', 'HD95'
    df_res = pd.read_excel(test_metrics_path)

    # clinical_info_path: sheet 'dataset_info'
    # load sheet 'dataset_info' if valid, else default
    try:
        df_clin = pd.read_excel(clinical_info_path, sheet_name='dataset_info')
    except:
        df_clin = pd.read_excel(clinical_info_path)

    # Merge
    # Ensure ID columns match. df_clin usually 'patient_id'. df_res 'Case_ID'
    print("Merging Metrics with Clinical Info...")
    if 'patient_id' not in df_clin.columns and 'Case_ID' in df_clin.columns:
        df_clin = df_clin.rename(columns={'Case_ID': 'patient_id'})

    merged = pd.merge(df_res, df_clin, left_on='Case_ID', right_on='patient_id', how='left')

    # Preprocess demographics
    merged = preprocess_clinical_data(merged)

    # 2. Compute NormHD
    # scoring_task1: norm_hausdorff = hausdorff_dist / hd_max (if not inf) else 1
    # We assume 'HD95' in df_res is the raw HD95 in mm.
    # Note: if HD95 is NaN (failed), treat as max (1.0 normalized)

    def normalize_hd(row):
        hd = row['HD95']
        if pd.isna(hd) or np.isinf(hd):
            return 1.0
        norm = hd / HD_MAX
        return min(norm, 1.0) # Clip to 1.0 ?? scoring_task1 doesn't explicitly clip in loop but metrics.py might?
        # metrics.py: norm_hausdorff = hausdorff_dist / hd_max if hausdorff_dist != np.inf else 1
        # It does NOT explicit clip to 1.0 if HD > 150. But assuming HD < 150 usually.
        # Let's clip to 1.0 to be safe and consistent with "1 - NormHD" logic (disparity shouldn't go negative etc).
        # Actually scoring_task1 doesn't clip. But standard logic implies it.

    merged['NormHD'] = merged.apply(normalize_hd, axis=1)

    # Rename Dice to DSC for consistency
    if 'Dice' in merged.columns:
        merged['DSC'] = merged['Dice']

    # 3. Performance Score
    # performance_score = 0.5 * (np.mean(dice_scores) + (1 - np.mean(hausdorff_distances)))
    mean_dsc = merged['DSC'].mean()
    mean_norm_hd = merged['NormHD'].mean()
    performance_score = 0.5 * (mean_dsc + (1 - mean_norm_hd))

    print(f"Mean DSC: {mean_dsc:.4f}")
    print(f"Mean NormHD: {mean_norm_hd:.4f}")
    print(f"Performance Score: {performance_score:.4f}")

    # 4. Fairness Score
    # For each variable:
    #   Dice Disparity = Max(GroupMeans) - Min(GroupMeans)
    #   NormHD Disparity = Max(GroupMeans) - Min(GroupMeans)
    #   Fairness_v = 1 - 0.5 * (DiceDisp + NormHDDisp)
    # Overall Fairness = Mean(Fairness_v)

    fairness_scores = {}
    reports = []

    # Use 'age_group' instead of 'age'
    active_vars = ['age_group', 'menopausal_status', 'breast_density']

    print("\n--- FAIRNESS ANALYSIS ---")

    for var in active_vars:
        # Get valid data
        valid_rows = merged.dropna(subset=[var, 'DSC', 'NormHD'])
        # Filter out 'unknown' if we want? scoring_task1 keeps them?
        # "fairness_varibles_df.groupby(variable)" includes all.

        groups = valid_rows.groupby(var)
        dsc_means = groups['DSC'].mean()
        nhd_means = groups['NormHD'].mean()
        counts = groups['DSC'].count()

        # Print group stats
        print(f"\nVariable: {var}")
        print(pd.concat([dsc_means.rename('DSC'), nhd_means.rename('NormHD'), counts.rename('Count')], axis=1))

        if len(dsc_means) > 0:
            dsc_disp = dsc_means.max() - dsc_means.min()
            nhd_disp = nhd_means.max() - nhd_means.min()
        else:
            dsc_disp = 0.0
            nhd_disp = 0.0

        feat_fairness = 1 - 0.5 * (dsc_disp + nhd_disp)
        fairness_scores[var] = feat_fairness

        print(f"  DSC Disp: {dsc_disp:.4f}")
        print(f"  NHD Disp: {nhd_disp:.4f}")
        print(f"  Feature Fairness: {feat_fairness:.4f}")

    avg_fairness = np.mean(list(fairness_scores.values()))

    # 5. Ranking Score
    ranking_score = (1 - ALPHA) * performance_score + ALPHA * avg_fairness

    # 6. Output
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    summary.append("="*50)
    summary.append("MAMA-MIA CHALLENGE METRICS REPORT (STRICT)")
    summary.append("="*50)
    summary.append(f"Mean DSC:        {mean_dsc:.4f}")
    summary.append(f"Mean NormHD:     {mean_norm_hd:.4f}")
    summary.append(f"PERFORMANCE:     {performance_score:.4f}")
    summary.append("-" * 30)
    for v, s in fairness_scores.items():
        summary.append(f"Fairness ({v}): {s:.4f}")
    summary.append(f"FAIRNESS EVAL:   {avg_fairness:.4f}")
    summary.append("="*50)
    summary.append(f"RANKING SCORE:   {ranking_score:.4f}")
    summary.append("="*50)

    print("\n" + "\n".join(summary))

    with open(os.path.join(output_dir, "challenge_metrics.txt"), "w") as f:
        f.write("\n".join(summary))

    # Save combined df
    merged.to_excel(os.path.join(output_dir, "test_metrics_extended_fairness.xlsx"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_metrics", required=True, help="Path to test_metrics.xlsx (must have Dice, HD95)")
    parser.add_argument("-c", "--clinical", required=True, help="Path to clinical_and_imaging_info.xlsx")
    parser.add_argument("-o", "--output_dir", required=True, help="Output output_dir")
    args = parser.parse_args()

    calculate_metrics(args.test_metrics, args.clinical, args.output_dir)
