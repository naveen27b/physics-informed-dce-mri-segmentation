import os
import json
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob

# Set from command-line arguments in main(). Kept module-level because worker
# processes read them when ProcessPoolExecutor starts.
SOURCE_ALIGNED_DIR = None
SOURCE_LABELS = None
SOURCE_JSONS = None
NNUNET_RAW = None
DATASET_ID = "Dataset600_MAMAMIAPhysics"

# Parameters
MAX_WORKERS = 32
CLIP_PERCENTILES = (0.1, 99.9)

def get_aligned_files(subject_id):
    subj_dir = os.path.join(SOURCE_ALIGNED_DIR, subject_id)
    if not os.path.exists(subj_dir):
        return None, None

    files = sorted(glob.glob(os.path.join(subj_dir, "*.nii.gz")))
    anchor_path = None
    post_paths = []

    for f in files:
        if f.endswith(f"{subject_id}_0000.nii.gz"):
            anchor_path = f
        else:
            post_paths.append(f)

    if not anchor_path:
        return None, None
    return anchor_path, post_paths

def crop_arrays_to_json(arrays_dict, json_path):
    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)

        if 'primary_lesion' not in meta or 'breast_coordinates' not in meta['primary_lesion']:
            return None, "No Coordinates"

        coords = meta['primary_lesion']['breast_coordinates']
        x_min, x_max = coords['x_min'], coords['x_max']
        y_min, y_max = coords['y_min'], coords['y_max']
        z_min, z_max = coords['z_min'], coords['z_max']

        start_x, end_x = max(0, z_min - 1), z_max
        start_y, end_y = max(0, y_min - 1), y_max
        start_z, end_z = max(0, x_min - 1), x_max

        def do_crop(arr):
            sx, ex = min(arr.shape[0], start_x), min(arr.shape[0], end_x)
            sy, ey = min(arr.shape[1], start_y), min(arr.shape[1], end_y)
            sz, ez = min(arr.shape[2], start_z), min(arr.shape[2], end_z)
            return arr[sx:ex, sy:ey, sz:ez]

        arrays_dict['pre'] = do_crop(arrays_dict['pre'])
        arrays_dict['posts'] = [do_crop(p) for p in arrays_dict['posts']]
        if arrays_dict['label'] is not None:
             arrays_dict['label'] = do_crop(arrays_dict['label'])

        return arrays_dict, None
    except Exception as e:
        return None, f"Crop Error: {e}"

def n4_bias_correction_numpy(image_numpy):
    img_sitk = sitk.GetImageFromArray(image_numpy)
    img_sitk = sitk.Cast(img_sitk, sitk.sitkFloat32)
    mask_sitk = sitk.OtsuThreshold(img_sitk, 0, 1, 200)
    try:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30])
        corrected_sitk = corrector.Execute(img_sitk, mask_sitk)
        corrected_numpy = sitk.GetArrayFromImage(corrected_sitk)

        with np.errstate(divide='ignore', invalid='ignore'):
             field = corrected_numpy / (image_numpy + 1e-6)
             field = np.nan_to_num(field, nan=1.0)
        return corrected_numpy, field
    except:
        return image_numpy, np.ones_like(image_numpy)

def compute_features(pre_n4, posts_n4):
    channels = []

    # 1. JOINT PERCENTILE SCALING (CRITICAL FIX)
    lower = np.percentile(pre_n4, CLIP_PERCENTILES[0])
    upper = np.percentile(pre_n4, CLIP_PERCENTILES[1])
    range_diff = max(upper - lower, 1e-5)

    # Pre is strictly clipped [0, 1]
    pre_norm = np.clip((pre_n4 - lower) / range_diff, 0.0, 1.0)

    # Posts are NOT clipped to 1.0. If we clip to 1.0, we flatten the enhancing tumor and erase the kinetics!
    # We clip the bottom to 0.0 and leave the top loose (up to 10.0 for safety against wild outliers)
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

def process_case(args):
    subject_id, out_tr, out_ts, lbl_tr, lbl_ts, is_train = args
    anchor_path, post_paths = get_aligned_files(subject_id)
    json_path = os.path.join(SOURCE_JSONS, f"{subject_id}.json")

    if not anchor_path or not os.path.exists(json_path):
        return f"SKIP {subject_id}: Missing Images or JSON"

    label_path = os.path.join(SOURCE_LABELS, f"{subject_id}.nii.gz")
    if is_train and not os.path.exists(label_path):
        return f"SKIP {subject_id}: Missing Label"

    try:
        nii = nib.load(anchor_path)
        affine, header = nii.affine, nii.header
        pre_dat = nii.get_fdata().astype(np.float32)
        post_dats = [nib.load(p).get_fdata().astype(np.float32) for p in post_paths]

        label_dat = None
        if os.path.exists(label_path):
            label_dat = nib.load(label_path).get_fdata().astype(np.uint8)

        data_dict = {'pre': pre_dat, 'posts': post_dats, 'label': label_dat}
        cropped_dict, err = crop_arrays_to_json(data_dict, json_path)

        if not cropped_dict:
            return f"FAIL {subject_id}: {err}"

        pre_crop, posts_crop, label_crop = cropped_dict['pre'], cropped_dict['posts'], cropped_dict['label']

        # N4 Bias Correction and Propagation
        pre_n4, field = n4_bias_correction_numpy(pre_crop)
        posts_n4 = [p * field for p in posts_crop]

        channels = compute_features(pre_n4, posts_n4)

        base_out = os.path.join(out_tr if is_train else out_ts, subject_id)
        label_out = os.path.join(lbl_tr if is_train else lbl_ts, f"{subject_id}.nii.gz")

        for i, ch in enumerate(channels):
             nib.save(nib.Nifti1Image(ch, affine, header), f"{base_out}_{i:04d}.nii.gz")

        if label_crop is not None:
            nib.save(nib.Nifti1Image(label_crop, affine, header), label_out)

        return None
    except Exception as e:
        return f"ERROR {subject_id}: {e}"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the four-channel Dataset600 nnU-Net dataset."
    )
    parser.add_argument("--source-aligned", required=True,
                        help="Root containing one registered-image folder per case")
    parser.add_argument("--source-labels", required=True,
                        help="Directory containing CASE_ID.nii.gz training labels")
    parser.add_argument("--source-jsons", required=True,
                        help="Directory containing CASE_ID.json ROI metadata")
    parser.add_argument("--split-csv", required=True,
                        help="CSV with train_split and test_split columns")
    parser.add_argument("--nnunet-raw", required=True,
                        help="Destination nnUNet_raw directory")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def main():
    global SOURCE_ALIGNED_DIR, SOURCE_LABELS, SOURCE_JSONS
    global NNUNET_RAW, DATASET_ID, MAX_WORKERS

    args = parse_args()
    SOURCE_ALIGNED_DIR = os.path.abspath(args.source_aligned)
    SOURCE_LABELS = os.path.abspath(args.source_labels)
    SOURCE_JSONS = os.path.abspath(args.source_jsons)
    NNUNET_RAW = os.path.abspath(args.nnunet_raw)
    DATASET_ID = args.dataset_id
    MAX_WORKERS = args.workers

    for d in ["imagesTr", "labelsTr", "imagesTs"]:
        os.makedirs(os.path.join(NNUNET_RAW, DATASET_ID, d), exist_ok=True)

    # Load Official Splits
    train_list = []
    test_list = []

    import pandas as pd
    df = pd.read_csv(args.split_csv)
    required_columns = {"train_split", "test_split"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing_columns)}")
    train_list = df["train_split"].dropna().astype(str).unique().tolist()
    test_list = df["test_split"].dropna().astype(str).unique().tolist()
    overlap = set(train_list).intersection(test_list)
    if overlap:
        raise ValueError(f"Subjects occur in both partitions: {sorted(overlap)[:5]}")
    if not train_list or not test_list:
        raise ValueError("Both train_split and test_split must contain cases")
    print(f"Loaded official splits: {len(train_list)} train, {len(test_list)} test")

    d_root = os.path.join(NNUNET_RAW, DATASET_ID)
    imagesTr, labelsTr = os.path.join(d_root, "imagesTr"), os.path.join(d_root, "labelsTr")
    imagesTs, labelsTs = os.path.join(d_root, "imagesTs"), os.path.join(d_root, "labelsTs")
    os.makedirs(labelsTs, exist_ok=True)

    tasks = [(s, imagesTr, imagesTs, labelsTr, labelsTs, True) for s in train_list]
    tasks += [(s, imagesTr, imagesTs, labelsTr, labelsTs, False) for s in test_list]

    print(f"Launching {MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(process_case, t): t[0] for t in tasks}
        for f in tqdm(as_completed(futures), total=len(tasks)):
            res = f.result()
            if res: print(res)

    meta = {
        "channel_names": {
            "0": "Anatomy_Harmonized",
            "1": "Rel_Enhancement",
            "2": "Subtraction_Post1_Pre",
            "3": "Washout_Slope_Pos"
        },
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(glob.glob(os.path.join(labelsTr, "*.nii.gz"))),
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(d_root, "dataset.json"), 'w') as f:
        json.dump(meta, f, indent=4)
    print("Dataset 600 Generation Complete.")

if __name__ == "__main__":
    main()
