#!/usr/bin/env python3
"""
predict_600.py  –  Dataset 600 Inference Pipeline (self-contained)
====================================================================
Adapted from predict_506.py for the Dataset600_MAMAMIAPhysics model which uses:
  • Joint percentile scaling (pre-contrast percentiles applied to ALL volumes)
  • Posts NOT clipped to 1.0 (up to 10.0) to preserve enhancing-tumor signal
  • Different clip ranges:  RelEnh [0,10], Sub [0,5], Washout [0,5]
  • 3d_fullres_iso configuration (1mm isotropic spacing)

Supports:
  • Single patient folder  OR  batch directory of patient folders
  • ROI crop via JSON bounding-box  OR  bilateral breast-segmentation mask
  • Single-fold (default: fold_0) or multi-fold ensemble
  • Proximity / volume / keep_largest post-processing
  • Optional Dice evaluation against a ground-truth directory
  • Uses ALL available post-contrast images for washout calculation

Usage examples
--------------
# Single patient, JSON crop
python predict_600.py \\
    --input /path/to/images/CASE_002 \\
    --json_dir /path/to/roi_json

# Batch mode on YUN test set
python predict_600.py \\
    --input /path/to/case_folders \\
    --batch \\
    --json_dir /path/to/roi_json \\
    --gt_dir /path/to/ground_truths \\
    --output inference_predictions/external_predictions \\
    --cases CASE_002 CASE_012 CASE_017

# Bilateral breast-mask crop
python predict_600.py \\
    --input /path/to/images/CASE_002 \\
    --mode bilateral \\
    --seg_dir /path/to/breast_masks

# Override model or checkpoint
python predict_600.py \\
    --input /path/to/images \\
    --batch \\
    --model /path/to/custom_model \\
    --folds 0 \\
    --checkpoint checkpoint_final.pth
"""

import os, sys, json, glob, argparse, tempfile, shutil, time, csv
import numpy as np
import nibabel as nib
import nibabel.processing
import torch
import torch.multiprocessing as mp
import SimpleITK as sitk
from scipy.ndimage import (label, generate_binary_structure,
                           distance_transform_edt)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ─────────────────────────────  DEFAULTS  ────────────────────────────────────

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(THIS_DIR))  # dataset_506_1000ep_analysis

DEFAULT_MODEL = None
DEFAULT_FOLDS      = (0,)           # Only fold_0 trained so far
DEFAULT_CHECKPOINT = "checkpoint_best.pth"
DEFAULT_GT_DIR     = None           # No default GT for YUN
DEFAULT_JSON_DIR   = None
DEFAULT_IMAGE_DIR  = None

# DS600 preprocessing parameters (CRITICAL: must match create_dataset_600.py)
CLIP_PCTS          = (0.1, 99.9)
EPS                = 1e-5

# ─────────────────────────── preprocessing helpers ───────────────────────────
# These EXACTLY match create_dataset_600.py to ensure train/test consistency

def n4_bias(img: np.ndarray, skip: bool = False):
    """N4 bias field correction.  Returns (corrected, bias_field)."""
    if skip:
        print("  [info] N4 skipped (--skip_n4)")
        return img, np.ones_like(img)
    try:
        s = sitk.Cast(sitk.GetImageFromArray(img), sitk.sitkFloat32)
        mask  = sitk.OtsuThreshold(s, 0, 1, 200)
        corr  = sitk.N4BiasFieldCorrectionImageFilter()
        corr.SetMaximumNumberOfIterations([50, 50, 30])
        fixed = sitk.GetArrayFromImage(corr.Execute(s, mask))
        with np.errstate(divide='ignore', invalid='ignore'):
            field = fixed / (img + 1e-6)
            field = np.nan_to_num(field, nan=1.0)
        return fixed, field
    except Exception as e:
        print(f"  [warn] N4 failed ({e}), using uncorrected image")
        return img, np.ones_like(img)


def compute_features_predict(pre_n4: np.ndarray, posts_n4: list) -> list:
    """
    DS600 physics-informed channels (MUST match create_dataset_600.py exactly)
    """
    channels = []

    # 1. JOINT PERCENTILE SCALING (CRITICAL FIX)
    lower = np.percentile(pre_n4, CLIP_PCTS[0])
    upper = np.percentile(pre_n4, CLIP_PCTS[1])
    range_diff = max(upper - lower, 1e-5)

    # Pre is strictly clipped [0, 1]
    pre_norm = np.clip((pre_n4 - lower) / range_diff, 0.0, 1.0)

    # Posts are NOT clipped to 1.0.
    posts_norm = [np.clip((p - lower) / range_diff, 0.0, 10.0) for p in posts_n4]

    # Ch 0: Anatomy
    channels.append(pre_norm.astype(np.float32))

    pre_safe = np.maximum(pre_norm, 0.01) # Avoid div by zero

    if len(posts_norm) >= 1:
        post1 = posts_norm[0]
        # Ch 1: Relative Enhancement
        rel_enh = (post1 - pre_norm) / pre_safe
        rel_enh = np.clip(rel_enh, 0.0, 10.0)
        channels.append(rel_enh.astype(np.float32))

        # Ch 2: Subtraction
        sub1 = post1 - pre_norm
        sub1 = np.clip(sub1, 0.0, 5.0)
        channels.append(sub1.astype(np.float32))
    else:
        channels.append(np.zeros_like(pre_norm, dtype=np.float32))
        channels.append(np.zeros_like(pre_norm, dtype=np.float32))

    # Ch 3: Washout Slope (Positive)
    if len(posts_norm) > 1:
        stack = np.stack(posts_norm, axis=-1)
        peak_val = np.max(stack, axis=-1)
        peak_idx = np.argmax(stack, axis=-1)
        last_val = posts_norm[-1]

        last_idx = len(posts_norm) - 1
        time_diff = last_idx - peak_idx
        time_diff = np.maximum(time_diff, 1.0)

        slope = (peak_val - last_val) / time_diff
        slope = np.clip(slope, 0.0, 5.0)
        channels.append(slope.astype(np.float32))
    else:
        channels.append(np.zeros_like(pre_norm, dtype=np.float32))

    return channels


# ──────────────────────── ROI cropping: JSON path ────────────────────────────

def json_crop(pre: np.ndarray, posts: list, json_path: str):
    """Crop arrays using breast bounding-box coordinates from JSON."""
    with open(json_path) as f:
        c = json.load(f)["primary_lesion"]["breast_coordinates"]
    sx, ex = max(0, c["z_min"]-1), c["z_max"]
    sy, ey = max(0, c["y_min"]-1), c["y_max"]
    sz, ez = max(0, c["x_min"]-1), c["x_max"]
    def crop(a):
        return a[min(a.shape[0],sx):min(a.shape[0],ex),
                 min(a.shape[1],sy):min(a.shape[1],ey),
                 min(a.shape[2],sz):min(a.shape[2],ez)]
    return crop(pre), [crop(p) for p in posts], (sx, ex, sy, ey, sz, ez)


def json_restore(seg: np.ndarray, orig_shape: tuple, bounds: tuple) -> np.ndarray:
    """Restore cropped segmentation back to original image volume."""
    sx, ex, sy, ey, sz, ez = bounds
    sx, ex = min(orig_shape[0], sx), min(orig_shape[0], ex)
    sy, ey = min(orig_shape[1], sy), min(orig_shape[1], ey)
    sz, ez = min(orig_shape[2], sz), min(orig_shape[2], ez)
    dx, dy, dz = ex-sx, ey-sy, ez-sz
    seg = seg[:dx, :dy, :dz]
    full = np.zeros(orig_shape, dtype=seg.dtype)
    full[sx:sx+seg.shape[0], sy:sy+seg.shape[1], sz:sz+seg.shape[2]] = seg
    return full


# ──────────────────────── ROI cropping: breast mask ──────────────────────────

def get_bbox(mask_data: np.ndarray, label_val: int, margin: int = 0):
    """Get bounding box slices for a given label value in a breast mask."""
    bm = (mask_data == label_val)
    if not np.any(bm):
        return None
    coords = np.argwhere(bm)
    lo, hi = coords.min(0), coords.max(0) + 1
    lo = np.maximum(0, lo - margin)
    hi = np.minimum(np.array(mask_data.shape), hi + margin)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def resample_to(moving_nii, fixed_nii, order=0):
    return nibabel.processing.resample_from_to(moving_nii, fixed_nii, order=order)


# ──────────────────────────── post-processing ────────────────────────────────

def get_components(mask: np.ndarray):
    s = generate_binary_structure(3, 3)
    lbl, n = label(mask, structure=s)
    if n == 0:
        return lbl, []
    sizes = np.bincount(lbl.ravel())
    comps = [(i, sizes[i]) for i in range(1, len(sizes)) if sizes[i] > 0]
    return lbl, sorted(comps, key=lambda x: x[1], reverse=True)


def proximity_postprocess(mask: np.ndarray, min_size: int = 50,
                          dist_threshold: float = 100.0) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    lbl, comps = get_components(mask)
    if not comps:
        return mask
    out = np.zeros_like(mask)
    largest_id = comps[0][0]
    out[lbl == largest_id] = 1
    if len(comps) > 1:
        grid = np.ones_like(lbl, dtype=np.float32)
        grid[lbl == largest_id] = 0
        dist = distance_transform_edt(grid)
        for cid, sz in comps[1:]:
            if sz >= min_size:
                cmask = (lbl == cid)
                if dist[cmask].min() <= dist_threshold:
                    out[cmask] = 1
    return out


def volume_postprocess(mask: np.ndarray, min_vol_pct: float = 5.0) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    lbl, comps = get_components(mask)
    if not comps:
        return mask
    out = np.zeros_like(mask)
    largest_id, largest_sz = comps[0]
    out[lbl == largest_id] = 1
    threshold = (min_vol_pct / 100.0) * largest_sz
    for cid, sz in comps[1:]:
        if sz >= threshold:
            out[lbl == cid] = 1
    return out


def keep_largest(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    lbl, comps = get_components(mask)
    if not comps:
        return mask
    out = np.zeros_like(mask)
    out[lbl == comps[0][0]] = 1
    return out


def apply_postprocess(mask: np.ndarray, strategy: str,
                      min_size: int = 50, dist_threshold: float = 100.0,
                      min_vol_pct: float = 5.0):
    if strategy == "proximity":
        return proximity_postprocess(mask, min_size, dist_threshold)
    elif strategy == "volume":
        return volume_postprocess(mask, min_vol_pct)
    elif strategy == "keep_largest":
        return keep_largest(mask)
    else:   # "none"
        return mask


# ────────────────────────────── Dice metric ──────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.flatten().astype(float), gt.flatten().astype(float)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


# ─────────────────────────── core inference ──────────────────────────────────

def find_json(case_id: str, json_dir: str) -> str | None:
    if not json_dir:
        return None
    p = os.path.join(json_dir, f"{case_id}.json")
    return p if os.path.exists(p) else None


def build_predictor(model_dir: str, folds: tuple,
                    checkpoint: str, device: torch.device) -> nnUNetPredictor:
    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=True,
    )
    pred.initialize_from_trained_model_folder(
        model_dir, use_folds=folds, checkpoint_name=checkpoint)
    return pred


def run_nnunet_on_channels(predictor, channels: list,
                           affine, header, case_id: str,
                           prep_dir: str, pred_dir: str, args_ns=None):
    """Save channels as nnUNet input, run prediction, return raw mask array."""
    for i, ch in enumerate(channels):
        nib.save(nib.Nifti1Image(ch.astype(np.float32), affine, header),
                 os.path.join(prep_dir, f"{case_id}_{i:04d}.nii.gz"))

    save_probs = bool(args_ns and hasattr(args_ns, 'confidence')
                      and args_ns.confidence != 0.5)

    predictor.predict_from_files(
        prep_dir, pred_dir,
        save_probabilities=save_probs, overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    if save_probs:
        npz_path = os.path.join(pred_dir, f"{case_id}.npz")
        if not os.path.exists(npz_path):
            raise RuntimeError(f"nnUNet produced no .npz (expected {npz_path})")
        probs = np.load(npz_path)['probabilities']
        tumor_probs = probs[1]
        return (tumor_probs >= args_ns.confidence).astype(np.uint8)
    else:
        out = os.path.join(pred_dir, f"{case_id}.nii.gz")
        if not os.path.exists(out):
            raise RuntimeError(f"nnUNet produced no output (expected {out})")
        return nib.load(out).get_fdata().astype(np.uint8)


def _preprocess_600(pre: np.ndarray, posts: list, skip_n4: bool, skip_norm: bool):
    """
    Full DS600 preprocessing: N4 → bias propagation → joint percentile scaling → channels.
    Returns list of 4 channel arrays.
    """
    pre_n4, field = n4_bias(pre, skip=skip_n4)
    posts_n4 = [p * field for p in posts]

    if skip_norm:
        return [pre_n4.astype(np.float32)] + [p.astype(np.float32) for p in posts_n4] + [np.zeros_like(pre_n4, dtype=np.float32)] * max(0, 3 - len(posts_n4))

    return compute_features_predict(pre_n4, posts_n4)


def infer_single(args_ns, predictor, case_id: str,
                 patient_dir: str, output_dir: str) -> dict:
    """Full pipeline for one patient. Returns metrics dict."""
    t0 = time.time()
    pre_path = os.path.join(patient_dir, f"{case_id}_0000.nii.gz")

    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"_0000 not found: {pre_path}")

    nii_pre = nib.load(pre_path)
    affine, header = nii_pre.affine, nii_pre.header
    orig_shape = nii_pre.shape
    pre = nii_pre.get_fdata().astype(np.float32)

    # Load ALL available post-contrast images (_0001, _0002, ...)
    posts = []
    post_idx = 1
    while True:
        post_path = os.path.join(patient_dir, f"{case_id}_{post_idx:04d}.nii.gz")
        if not os.path.exists(post_path):
            break
        posts.append(nib.load(post_path).get_fdata().astype(np.float32))
        post_idx += 1

    if len(posts) == 0:
        raise FileNotFoundError("No post-contrast images found (_0001.nii.gz)")

    print(f"  Timepoints: Pre + {len(posts)} post-contrast images")
    if len(posts) == 1:
        print("  Washout   : single post → baseline zero fallback")

    prep_dir = tempfile.mkdtemp(prefix="600_prep_")
    pred_dir = tempfile.mkdtemp(prefix="600_pred_")
    mask_full = None

    try:
        mode = _detect_mode(args_ns, case_id, patient_dir)

        if mode == "json":
            json_path = find_json(case_id, args_ns.json_dir)
            crop_success = False

            if json_path:
                try:
                    print(f"  ROI mode  : JSON crop  ({json_path})")
                    pre_crop, posts_crop, bounds = json_crop(pre, posts, json_path)
                    print(f"  Cropped   : {orig_shape} -> {pre_crop.shape}")

                    channels = _preprocess_600(pre_crop, posts_crop,
                                               args_ns.skip_n4, args_ns.skip_norm)
                    raw_mask = run_nnunet_on_channels(
                        predictor, channels, affine, header,
                        case_id, prep_dir, pred_dir, args_ns)
                    raw_mask = apply_postprocess(
                        raw_mask, args_ns.postprocess,
                        args_ns.min_size, args_ns.dist_threshold,
                        args_ns.min_vol_pct)
                    mask_full = json_restore(raw_mask, orig_shape, bounds)
                    crop_success = True
                except Exception as e:
                    print(f"  [warn] JSON crop failed: {e}")
                    print(f"  [warn] Falling back to full-image mode")
            else:
                print(f"  [warn] JSON not found for {case_id}")
                print(f"  [warn] Falling back to full-image mode")

            if not crop_success:
                channels = _preprocess_600(pre, posts,
                                           args_ns.skip_n4, args_ns.skip_norm)
                raw_mask = run_nnunet_on_channels(
                    predictor, channels, affine, header,
                    case_id, prep_dir, pred_dir, args_ns)
                mask_full = apply_postprocess(
                    raw_mask, args_ns.postprocess,
                    args_ns.min_size, args_ns.dist_threshold,
                    args_ns.min_vol_pct)

        elif mode == "bilateral":
            seg_path = _resolve_seg(args_ns, case_id, patient_dir)
            if not seg_path:
                raise FileNotFoundError(f"Seg mask not found for {case_id}")
            print(f"  ROI mode  : bilateral  ({seg_path})")

            seg_nii = nib.load(seg_path)
            if seg_nii.shape != nii_pre.shape or \
               not np.allclose(seg_nii.affine, affine, atol=1e-3):
                print("  Resampling seg mask to image space...")
                seg_nii = resample_to(seg_nii, nii_pre, order=0)
            seg_data = seg_nii.get_fdata().astype(np.uint8)

            # N4 on full image first, then crop per-breast
            pre_n4, field = n4_bias(pre, skip=args_ns.skip_n4)
            posts_n4 = [p * field for p in posts]

            mask_full = np.zeros(orig_shape, dtype=np.uint8)
            for side, lval in [("Left",  args_ns.label_left),
                               ("Right", args_ns.label_right)]:
                bbox = get_bbox(seg_data, lval, margin=args_ns.bbox_margin)
                if bbox is None:
                    print(f"  {side} breast label {lval} not found – skipping")
                    continue
                print(f"  {side} breast bbox: {bbox}")

                pre_c = pre_n4[bbox]
                post_c = [p[bbox] for p in posts_n4]

                if args_ns.skip_norm:
                    ch = [pre_c.astype(np.float32)] + [pc.astype(np.float32) for pc in post_c] + [np.zeros_like(pre_c, dtype=np.float32)] * max(0, 3 - len(post_c))
                else:
                    ch = compute_features_predict(pre_c, post_c)

                spacing = tuple(float(x) for x in header.get_zooms()[:3])
                roi_pred = predictor.predict_single_npy_array(
                    np.stack(ch, axis=0).astype(np.float32),
                    {"spacing": spacing}, None, None,
                    save_or_return_probabilities=False,
                )
                roi_pred = apply_postprocess(
                    roi_pred.astype(np.uint8), args_ns.postprocess,
                    args_ns.min_size, args_ns.dist_threshold)
                cur = mask_full[bbox]
                mask_full[bbox] = np.maximum(cur, roi_pred.astype(np.uint8))

            mask_full = mask_full * (seg_data > 0).astype(np.uint8)

        else:  # mode == "full"
            print("  ROI mode  : full image (no crop)")
            channels = _preprocess_600(pre, posts,
                                       args_ns.skip_n4, args_ns.skip_norm)
            raw_mask = run_nnunet_on_channels(
                predictor, channels, affine, header,
                case_id, prep_dir, pred_dir, args_ns)
            mask_full = apply_postprocess(
                raw_mask, args_ns.postprocess,
                args_ns.min_size, args_ns.dist_threshold,
                args_ns.min_vol_pct)

        # ── save prediction ──────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        pred_out = os.path.join(output_dir, f"{case_id}.nii.gz")
        nib.save(nib.Nifti1Image(mask_full, affine, header), pred_out)
        elapsed = time.time() - t0
        print(f"  Saved     : {pred_out}  ({mask_full.sum()} voxels, {elapsed:.1f}s)")

        # ── optional Dice evaluation ─────────────────────────────────────
        ds = None
        gt_vox = None
        if args_ns.gt_dir:
            gt_path = os.path.join(args_ns.gt_dir, f"{case_id}.nii.gz")
            if os.path.exists(gt_path):
                gt_mask = (nib.load(gt_path).get_fdata() > 0.5).astype(np.uint8)
                gt_vox = int(gt_mask.sum())
                ds = dice_score(mask_full, gt_mask)
                quality = ("EXCELLENT" if ds >= 0.8 else
                           "GOOD"      if ds >= 0.7 else
                           "FAIR"      if ds >= 0.6 else "POOR")
                print(f"  Dice      : {ds:.4f}  ({quality})  | "
                      f"GT {gt_vox} vs Pred {mask_full.sum()}")
            else:
                print(f"  Dice      : skipped (no GT at {gt_path})")

        return {"case_id": case_id, "pred_file": pred_out,
                "pred_voxels": int(mask_full.sum()), "gt_voxels": gt_vox,
                "dice": ds, "elapsed_s": round(elapsed, 1)}

    finally:
        shutil.rmtree(prep_dir, ignore_errors=True)
        shutil.rmtree(pred_dir, ignore_errors=True)


# ─────────────────── mode + seg-mask resolution helpers ──────────────────────

def _detect_mode(args_ns, case_id: str, patient_dir: str) -> str:
    """Decide crop mode: json > bilateral > full."""
    if args_ns.mode:
        return args_ns.mode
    if args_ns.json_dir and find_json(case_id, args_ns.json_dir):
        return "json"
    if _resolve_seg(args_ns, case_id, patient_dir):
        return "bilateral"
    return "full"


def _resolve_seg(args_ns, case_id: str, patient_dir: str) -> str | None:
    """Find breast seg mask from --breast_divided_mask, --seg_dir, or patient folder."""
    if hasattr(args_ns, 'breast_divided_mask') and args_ns.breast_divided_mask \
       and os.path.exists(args_ns.breast_divided_mask):
        return args_ns.breast_divided_mask
    if hasattr(args_ns, 'seg_dir') and args_ns.seg_dir:
        for name in [f"{case_id}.nii.gz", f"{case_id}_breast.nii.gz",
                     f"{case_id}_seg.nii.gz", f"{case_id}_breast_mask.nii.gz"]:
            p = os.path.join(args_ns.seg_dir, name)
            if os.path.exists(p):
                return p
    for name in [f"{case_id}_breast.nii.gz", f"{case_id}_seg.nii.gz",
                 f"{case_id}_breast_mask.nii.gz"]:
        p = os.path.join(patient_dir, name)
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────── main ────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Dataset 600 Inference Pipeline (physics-informed, iso-spacing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input / output ---
    p.add_argument("--input", required=True,
                   help="Patient folder (single) OR parent folder (batch with --batch)")
    p.add_argument("--output", default="inference_predictions/predictions_600",
                   help="Output directory for segmentation masks")
    p.add_argument("--batch", action="store_true",
                   help="Treat --input as parent dir; process every sub-folder")
    p.add_argument("--case_id", default=None,
                   help="Override case ID (single mode only)")
    p.add_argument("--cases", nargs="+", default=None,
                   help="Explicit list of case IDs to process (batch mode)")

    # --- Model ---
    p.add_argument("--model", default=DEFAULT_MODEL, required=True,
                   help="Path to nnUNetTrainer model folder")
    p.add_argument("--folds", nargs="+", type=int, default=list(DEFAULT_FOLDS),
                   help="Folds to ensemble")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="checkpoint_best.pth or checkpoint_final.pth")

    # --- ROI mode ---
    p.add_argument("--mode", default=None, choices=["json", "bilateral", "full"],
                   help="Force ROI mode (default: auto-detect json > bilateral > full)")
    p.add_argument("--json_dir", default=DEFAULT_JSON_DIR,
                   help="Directory containing per-case .json bounding-box files")
    p.add_argument("--breast_divided_mask", default=None,
                   help="Single breast seg NIfTI for bilateral processing")
    p.add_argument("--seg_dir", default=None,
                   help="Directory with per-case breast seg masks (bilateral batch)")
    p.add_argument("--label_left",  type=int, default=1,
                   help="Label value for left breast in seg mask")
    p.add_argument("--label_right", type=int, default=2,
                   help="Label value for right breast in seg mask")
    p.add_argument("--bbox_margin", type=int, default=0,
                   help="Extra voxel margin around bilateral breast bboxes")

    # --- Post-processing ---
    p.add_argument("--postprocess", default="volume",
                   choices=["proximity", "volume", "keep_largest", "none"],
                   help="Post-processing strategy")
    p.add_argument("--min_size",       type=int,   default=50,
                   help="Min voxel size for secondary components (proximity)")
    p.add_argument("--dist_threshold", type=float, default=100.0,
                   help="Max distance (voxels) for proximity inclusion")
    p.add_argument("--min_vol_pct",    type=float, default=5.0,
                   help="Min volume %% of largest blob to keep (volume mode)")
    p.add_argument("--confidence",     type=float, default=0.5,
                   help="Tumor probability threshold [0-1]")

    # --- Preprocessing overrides ---
    p.add_argument("--skip_n4", action="store_true",
                   help="Skip N4 bias field correction")
    p.add_argument("--skip_norm", action="store_true",
                   help="Skip percentile normalization")


    # --- Evaluation ---
    p.add_argument("--gt_dir", default=DEFAULT_GT_DIR,
                   help="Ground-truth directory for Dice evaluation ('' to disable)")

    # --- Device ---
    p.add_argument("--device", default="cuda",
                   help="cuda / cuda:0 / cuda:1 / cpu")

    # --- Batch behaviour ---
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true",
                   help="Skip cases whose output mask already exists")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                   help="Process cases even if output exists")
    p.set_defaults(skip_existing=False)

    return p.parse_args()


def main():
    args = parse_args()

    # ── validate model ───────────────────────────────────────────────────
    if not os.path.isdir(args.model):
        print(f"ERROR: Model folder not found:\n  {args.model}")
        sys.exit(1)

    # ── disable GT if empty string ───────────────────────────────────────
    if args.gt_dir == "" or args.gt_dir == "None":
        args.gt_dir = None
    if args.gt_dir and not os.path.isdir(args.gt_dir):
        print(f"[warn] GT dir not found, Dice disabled: {args.gt_dir}")
        args.gt_dir = None

    # ── build case list ──────────────────────────────────────────────────
    if args.batch:
        all_subdirs = sorted([
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d))
        ])
        if args.cases:
            # Filter to only requested cases
            case_ids = [c for c in args.cases if c in all_subdirs]
            if len(case_ids) < len(args.cases):
                missing = set(args.cases) - set(case_ids)
                print(f"[warn] Cases not found in input dir: {missing}")
        else:
            case_ids = all_subdirs
        patient_dirs = [os.path.join(args.input, c) for c in case_ids]
    else:
        patient_dirs = [args.input]
        case_ids = [args.case_id or os.path.basename(args.input.rstrip("/"))]

    folds = tuple(args.folds)

    print(f"\n{'='*65}")
    print(f"  Dataset 600 Inference Pipeline")
    print(f"{'='*65}")
    print(f"  Cases      : {len(case_ids)}")
    print(f"  Model      : {args.model}")
    print(f"  Folds      : {folds}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Config     : 3d_fullres_iso (1mm isotropic)")
    print(f"  Preproc    : DS600 (joint norm, clips [0,10]/[0,5]/[0,5])")
    print(f"  PostProc   : {args.postprocess}")
    print(f"  JSON dir   : {args.json_dir or 'none'}")
    print(f"  Breast mask: {args.seg_dir or args.breast_divided_mask or 'none'}")
    print(f"  GT dir     : {args.gt_dir or 'none (Dice disabled)'}")
    print(f"  Output     : {args.output}")
    print(f"{'='*65}\n")

    # ── initialize predictor once ────────────────────────────────────────
    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device_str = "cpu"

    if ":" in device_str:
        dev_type, dev_idx = device_str.split(":")
        device = torch.device(dev_type, int(dev_idx))
    else:
        device = torch.device(device_str)
    print(f"  Device: {device}\n")

    predictor = build_predictor(args.model, folds, args.checkpoint, device)

    # ── process cases ────────────────────────────────────────────────────
    all_results = []
    failed = []

    for case_id, patient_dir in zip(case_ids, patient_dirs):
        print(f"\n[{len(all_results)+len(failed)+1}/{len(case_ids)}] {case_id}")
        print(f"  Folder: {patient_dir}")

        pred_out = os.path.join(args.output, f"{case_id}.nii.gz")
        if args.skip_existing and os.path.exists(pred_out):
            print(f"  Skipping (prediction exists): {pred_out}")
            continue

        try:
            result = infer_single(args, predictor, case_id,
                                  patient_dir, args.output)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed.append({"case_id": case_id, "error": str(e)})

    # ── summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DONE  —  {len(all_results)} succeeded, {len(failed)} failed")
    if all_results:
        dice_vals = [r["dice"] for r in all_results if r["dice"] is not None]
        if dice_vals:
            mean_d = sum(dice_vals) / len(dice_vals)
            print(f"  Dice  —  mean {mean_d:.4f}  "
                  f"min {min(dice_vals):.4f}  max {max(dice_vals):.4f}  "
                  f"(n={len(dice_vals)})")
    if failed:
        print(f"\n  Failed cases:")
        for f in failed:
            print(f"    {f['case_id']}: {f['error']}")
    print(f"  Output: {os.path.abspath(args.output)}")
    print(f"{'='*65}")

    # ── save CSV summary ─────────────────────────────────────────────────
    if all_results:
        csv_path = os.path.join(args.output, "results_summary.csv")
        os.makedirs(args.output, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Summary CSV: {csv_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
