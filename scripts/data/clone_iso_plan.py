#!/usr/bin/env python3
"""Clone a verified nnU-Net residual-encoder plan for a new dataset name."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_plan")
    parser.add_argument("target_plan")
    parser.add_argument("--dataset-name", default="Dataset600_MAMAMIAPhysics")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.source_plan, encoding="utf-8") as source:
        plan = json.load(source)

    plan["dataset_name"] = args.dataset_name
    configuration = plan["configurations"]["3d_fullres_iso"]
    architecture = configuration["architecture"]["network_class_name"]
    spacing = configuration["spacing"]

    os.makedirs(os.path.dirname(os.path.abspath(args.target_plan)), exist_ok=True)
    with open(args.target_plan, "w", encoding="utf-8") as target:
        json.dump(plan, target, indent=2)
        target.write("\n")

    print(f"Saved: {args.target_plan}")
    print(f"Architecture: {architecture.split('.')[-1]}")
    print(f"Spacing: {spacing}")


if __name__ == "__main__":
    main()
