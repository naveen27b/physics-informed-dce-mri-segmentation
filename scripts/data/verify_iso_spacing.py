#!/usr/bin/env python3
"""Verify target spacing recorded in nnU-Net preprocessed case properties."""

import argparse
import glob
import os
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl_dir")
    parser.add_argument("--expected", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.pkl_dir, "*.pkl")))
    if not files:
        raise SystemExit(f"No .pkl files found in {args.pkl_dir}")

    spacings = []
    failures = []
    for path in files:
        with open(path, "rb") as stream:
            properties = pickle.load(stream)
        spacing = properties.get("spacing", properties.get("spacing_of_data"))
        if spacing is None:
            failures.append((os.path.basename(path), "missing spacing"))
            continue
        spacing = np.asarray(spacing, dtype=float)
        spacings.append(spacing)
        if not np.allclose(spacing, args.expected, atol=1e-3):
            failures.append((os.path.basename(path), spacing.tolist()))

    print(f"Checked {len(files)} files; {len(failures)} deviations")
    if spacings:
        values = np.asarray(spacings)
        print(f"Min:  {values.min(axis=0)}")
        print(f"Max:  {values.max(axis=0)}")
        print(f"Mean: {values.mean(axis=0)}")
    for name, spacing in failures[:20]:
        print(f"Deviation: {name}: {spacing}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
