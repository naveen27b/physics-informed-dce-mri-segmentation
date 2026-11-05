#!/usr/bin/env python3
"""
predict_600_multigpu.py – Multi-GPU wrapper for predict_600.py
===============================================================
Splits patients across multiple GPUs and runs predict_600.py --batch in
parallel.  Each GPU loads the model ONCE and processes its share of patients.
Automatically skips patients whose predictions already exist in --output.

Usage
-----
# Run on default GPUs (4,5,6,7) — skips already-predicted patients
python src/inference/predict_600_multigpu.py \\
    --input datasets/YUN_Dataset_nnunet/images \\
    --gt_dir datasets/YUN_Dataset_nnunet/ground_truths \\
    --output inference_predictions/YUN_predictions_600

# Custom GPUs
python src/inference/predict_600_multigpu.py \\
    --input datasets/YUN_Dataset_nnunet/images \\
    --gt_dir datasets/YUN_Dataset_nnunet/ground_truths \\
    --output inference_predictions/YUN_predictions_600 \\
    --gpus 0 1 2 3

# Specific cases only
python src/inference/predict_600_multigpu.py \\
    --input datasets/YUN_Dataset_nnunet/images \\
    --cases YUN_002 YUN_012 YUN_017 \\
    --output inference_predictions/YUN_predictions_600 \\
    --gpus 5

# Force re-prediction (no skipping)
python src/inference/predict_600_multigpu.py \\
    --input datasets/YUN_Dataset_nnunet/images \\
    --output inference_predictions/YUN_predictions_600 \\
    --no_skip
"""

import os, sys, subprocess, argparse, time, shutil, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICT_SCRIPT = os.path.join(THIS_DIR, "predict_600.py")

DEFAULT_GPUS = [4, 5, 6, 7]


def get_all_cases(input_dir: str) -> list[str]:
    """List all patient subdirectories."""
    return sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])


def get_pending_cases(all_cases: list[str], output_dir: str, skip: bool) -> list[str]:
    """Filter out cases that already have predictions."""
    if not skip:
        return all_cases

    pending = []
    skipped = []
    for case_id in all_cases:
        pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
        if os.path.exists(pred_path):
            skipped.append(case_id)
        else:
            pending.append(case_id)

    if skipped:
        print(f"\n  Skipping {len(skipped)} already-predicted cases:")
        for s in skipped:
            print(f"    ✓ {s}")
    return pending


def make_symlink_dir(input_dir: str, case_ids: list[str]) -> str:
    """Create a temp directory with symlinks to the assigned patient folders."""
    tmp = tempfile.mkdtemp(prefix="multigpu_600_split_")
    for case_id in case_ids:
        src = os.path.join(input_dir, case_id)
        dst = os.path.join(tmp, case_id)
        os.symlink(os.path.abspath(src), dst)
    return tmp


def run_gpu_worker(gpu_id: int, case_ids: list[str], input_dir: str,
                   output_dir: str, extra_args: list[str]) -> dict:
    """
    Create a symlinked input dir for this GPU's cases, then run
    predict_600.py --batch ONCE.  Model loads once, processes all cases.
    """
    if not case_ids:
        return {"gpu": gpu_id, "cases": [], "succeeded": [], "failed": []}

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"gpu_{gpu_id}.log")

    def log(msg: str):
        stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(f"  [GPU {gpu_id}] {stamped}", flush=True)
        with open(log_path, "a") as lf:
            lf.write(stamped + "\n")

    tmp_dir = None
    try:
        tmp_dir = make_symlink_dir(input_dir, case_ids)

        python_bin = sys.executable

        cmd = [
            python_bin, PREDICT_SCRIPT,
            "--input", tmp_dir,
            "--batch",
            "--output", output_dir,
            "--device", f"cuda:{gpu_id}",
        ] + extra_args

        log(f"Launching batch: {len(case_ids)} cases {case_ids}")
        t0 = time.time()

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)

        proc.wait()
        elapsed = time.time() - t0

        succeeded = []
        failed = []
        for case_id in case_ids:
            if os.path.exists(os.path.join(output_dir, f"{case_id}.nii.gz")):
                succeeded.append(case_id)
            else:
                failed.append(case_id)

        status = "✓" if not failed else "⚠"
        log(f"{status} Batch done: {len(succeeded)}/{len(case_ids)} succeeded ({elapsed:.1f}s)")

        return {"gpu": gpu_id, "cases": case_ids,
                "succeeded": succeeded, "failed": failed}

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def split_cases(cases: list[str], n_gpus: int) -> list[list[str]]:
    """Split cases as evenly as possible across GPUs."""
    chunks = [[] for _ in range(n_gpus)]
    for i, case in enumerate(cases):
        chunks[i % n_gpus].append(case)
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU wrapper for predict_600.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Parent directory of patient folders (batch mode)")
    parser.add_argument("--output", default="inference_predictions/predictions_600",
                        help="Output directory for predictions")
    parser.add_argument("--gpus", nargs="+", type=int, default=DEFAULT_GPUS,
                        help=f"GPU IDs to use (default: {DEFAULT_GPUS})")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Explicit list of case IDs to process (default: all)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-prediction even if output already exists")

    # All remaining args are passed through to predict_600.py
    args, passthrough = parser.parse_known_args()

    # ── collect and filter cases ─────────────────────────────────────────
    all_cases = get_all_cases(args.input)
    if not all_cases:
        print(f"ERROR: No patient subdirectories found in {args.input}")
        sys.exit(1)

    # Filter to specific cases if requested
    if args.cases:
        all_cases = [c for c in all_cases if c in args.cases]
        if not all_cases:
            print(f"ERROR: None of the requested cases found in {args.input}")
            sys.exit(1)

    pending = get_pending_cases(all_cases, args.output, skip=not args.no_skip)

    print(f"\n{'='*65}")
    print(f"  Multi-GPU Inference Wrapper (Dataset 600)")
    print(f"{'='*65}")
    print(f"  Total patients : {len(all_cases)}")
    print(f"  Already done   : {len(all_cases) - len(pending)}")
    print(f"  To predict     : {len(pending)}")
    print(f"  GPUs           : {args.gpus}")
    print(f"  Output         : {args.output}")
    print(f"  Extra args     : {passthrough}")
    print(f"{'='*65}\n")

    if not pending:
        print("  Nothing to do — all cases already predicted!")
        print("  Use --no_skip to force re-prediction.")
        return

    # ── split across GPUs ────────────────────────────────────────────────
    n_gpus = len(args.gpus)
    chunks = split_cases(pending, n_gpus)

    for gpu_id, chunk in zip(args.gpus, chunks):
        print(f"  GPU {gpu_id}: {len(chunk)} cases  {chunk}")

    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, "logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n  📁 Per-GPU logs:")
    for gpu_id in args.gpus:
        print(f"     tail -f {os.path.abspath(log_dir)}/gpu_{gpu_id}.log")

    # ── launch parallel workers ──────────────────────────────────────────
    t_start = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {}
        for gpu_id, chunk in zip(args.gpus, chunks):
            if chunk:
                fut = executor.submit(
                    run_gpu_worker, gpu_id, chunk,
                    args.input, args.output, passthrough
                )
                futures[fut] = gpu_id

        for fut in as_completed(futures):
            result = fut.result()
            all_results.append(result)

    # ── summary ──────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    total_ok = sum(len(r["succeeded"]) for r in all_results)
    total_fail = sum(len(r["failed"]) for r in all_results)

    print(f"\n{'='*65}")
    print(f"  MULTI-GPU INFERENCE COMPLETE (Dataset 600)")
    print(f"{'='*65}")
    print(f"  Succeeded  : {total_ok}")
    print(f"  Failed     : {total_fail}")
    print(f"  Total time : {total_time:.1f}s")
    print(f"  Output     : {os.path.abspath(args.output)}")

    if total_fail > 0:
        print(f"\n  Failed cases:")
        for r in all_results:
            for c in r["failed"]:
                print(f"    GPU {r['gpu']}: {c}")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
