
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, jaccard_score

def compute_metrics(pred, gt, spacing):
    # Flatten for sklearn metrics (careful with memory on large volumes, but breast MRI should be okay)
    # Optimization: Calculate statistics on Flattened arrays directly

    p_flat = pred.flatten().astype(bool)
    g_flat = gt.flatten().astype(bool)

    # Basics
    tp = np.logical_and(p_flat, g_flat).sum()
    tn = np.logical_and(~p_flat, ~g_flat).sum()
    fp = np.logical_and(p_flat, ~g_flat).sum()
    fn = np.logical_and(~p_flat, g_flat).sum()

    # 1. Dice (F1)
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 1.0 if (fp+fn)==0 else 0.0

    # 2. IoU (Jaccard)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0 if (fp+fn)==0 else 0.0

    # 3. Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 # If no GT object, recall is 1? Or undefined. Usually 1.0 if pred is empty.

    # 4. Precision (Positive Predictive Value)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if tp==0 else 0.0 # If empty pred, precision 1.0?

    # 5. Accuracy (Pixel-wise)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 6. Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

    # 7. HD95
    hd95 = compute_hd95_impl(p_flat.reshape(pred.shape), g_flat.reshape(gt.shape), spacing)

    # 8. NormHD
    norm_hd = min(hd95 / 150.0, 1.0)

    # 9. AUC (Binary)
    # AUC on binary masks is essentially (Sensitivity + Specificity) / 2 for balanced classes,
    # but for imbalanced segmentation it's just trapezoidal area of a single point (TPR, FPR).
    # It's not very meaningful without probabilities, but we can compute it.
    try:
        # Use a simplified calculation to save memory/time vs sklearn.roc_auc_score on millions of pixels
        # AUC = 1 - (fp / (fp + tn)) * (1 - tp / (tp + fn)) / 2 ... wait, binary ROC is simple.
        # It's just (Sensitivity + Specificity) / 2 is Balanced Accuracy. AUC is exactly Balanced Accuracy for binary predictors.
        auc = (recall + specificity) / 2.0
    except:
        auc = 0.5

    return {
        'Dice': dice,
        'IoU': iou,
        'Recall': recall,
        'Precision': precision,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'AUC_Binary': auc,
        'HD95': hd95,
        'NormHD': norm_hd
    }

def compute_hd95_impl(p_bool, g_bool, spacing):
    if not p_bool.any() and not g_bool.any(): return 0.0
    if not p_bool.any() or not g_bool.any(): return 999.0

    struct = generate_binary_structure(3, 1)

    dist_p2g = 999.0
    dist_g2p = 999.0

    # Surface Pred
    surface_pred = p_bool ^ binary_erosion(p_bool, structure=struct)
    pts_pred = np.argwhere(surface_pred) * np.array(spacing)

    # Surface GT
    surface_gt = g_bool ^ binary_erosion(g_bool, structure=struct)
    pts_gt = np.argwhere(surface_gt) * np.array(spacing)

    if len(pts_pred) == 0 or len(pts_gt) == 0: return 999.0

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    d_p2g, _ = tree_gt.query(pts_pred, k=1)
    d_g2p, _ = tree_pred.query(pts_gt, k=1)

    all_d = np.concatenate([d_p2g, d_g2p])
    return np.percentile(all_d, 95)

def evaluate_case(case_id, pred_path, gt_path):
    try:
        if not os.path.exists(gt_path):
            return None

        p_nii = nib.load(pred_path)
        g_nii = nib.load(gt_path)

        spacing = p_nii.header.get_zooms()[:3]

        p = (p_nii.get_fdata() > 0).astype(np.uint8)
        g = (g_nii.get_fdata() > 0).astype(np.uint8)

        if p.shape != g.shape:
             return {'Case_ID': case_id, 'Error': f"Shape mismatch {p.shape} vs {g.shape}"}

        metrics = compute_metrics(p, g, spacing)
        metrics['Case_ID'] = case_id
        return metrics

    except Exception as e:
        return {'Case_ID': case_id, 'Error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    files = [f for f in os.listdir(args.pred) if f.endswith('.nii.gz')]
    tasks = []

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for f in files:
        case_id = f.replace('.nii.gz', '')
        # Handle suffixes like _seg or _postprocessed if present
        if "_seg" in case_id: case_id = case_id.replace("_seg", "")

        gt_f = os.path.join(args.gt, f"{case_id}.nii.gz")
        if not os.path.exists(gt_f):
             # Try direct match
             gt_f = os.path.join(args.gt, f)

        tasks.append([case_id, os.path.join(args.pred, f), gt_f])

    print(f"Evaluating {len(tasks)} cases with {args.workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(evaluate_case, t[0], t[1], t[2]): t[0] for t in tasks}

        for f in tqdm(as_completed(futures), total=len(tasks)):
            res = f.result()
            if res:
                results.append(res)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['Case_ID', 'Dice', 'IoU', 'Recall', 'Precision', 'Specificity', 'Accuracy', 'AUC_Binary', 'HD95', 'NormHD', 'Error']
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    print("="*30)
    print(f"Mean Dice:      {df['Dice'].mean():.4f}")
    print(f"Mean IoU:       {df['IoU'].mean():.4f}")
    print(f"Mean Recall:    {df['Recall'].mean():.4f}")
    print(f"Mean Precision: {df['Precision'].mean():.4f}")
    print(f"Mean HD95:      {df[df['HD95'] < 900]['HD95'].mean():.4f}")
    print("="*30)

    df.to_excel(args.out, index=False)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
