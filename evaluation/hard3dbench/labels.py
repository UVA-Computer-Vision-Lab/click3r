#!/usr/bin/env python3
"""Label every Hard3DBench 2-view pair with its failure categories; print Table 7.

A pair can have several labels (paper Table 5):

  large_viewpoint      relative rotation > 60 degrees
  moderate_viewpoint   30 < relative rotation <= 60 degrees
  low_overlap          overlap < 0.20
  large_baseline       baseline / median depth > 0.3
  extreme_depth_range  max depth / min depth > 50
  other                none of the above

Relative rotation is the angle of R1^T R2 and the baseline the distance between the
two camera centres, both from the ground-truth cameras in float64. Overlap is the
fraction of valid view-1 depth pixels (every 7th row and column of the 224 x 224
input) that project inside view 2. Depth statistics use pixels with depth > 0.01:
the median depth is the mean of the two views' medians, the depth range the larger
of the two views' max/min. They are taken from the 224 x 224 crop of the DUSt3R
loaders for In-Distribution and from the full-resolution depth for
Out-of-Distribution, which reproduces the shipped labels. A pair whose depth gives
no valid statistic is an error, not an unlabeled pair.

  python evaluation/hard3dbench/labels.py --data-root /path/to/processed_data \\
      --co3d-root /path/to/table1_data --ood-root /path/to/hard3dbench_ood \\
      --output outputs/categories.json

--table7 prints paper Table 7 from the 2-view result files of a Table 2 results
directory and the labels in click3r/configs/hard3dbench/categories.json: mAA@30 per
category of DUSt3R, DUSt3R (retrained), Click3R w/o prompts and Click3R (8 prompts)
over all In- and Out-of-Distribution pairs. Unlike Table 2, a failed pose stays in
the denominator and counts as a miss.

  python evaluation/hard3dbench/labels.py --table7 outputs/table2
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in ("third_party/segment-anything", "third_party/dust3r", ""):
    sys.path.insert(0, str(REPO_ROOT / path))

import numpy as np
from tqdm import tqdm

from click3r.metrics import calculate_auc_np
from evaluation.hard3dbench.common import SPLIT_NAMES, load_list, result_path
from evaluation.hard3dbench.data import (
    indist_folders,
    load_pair,
    ood_cameras,
    read_ood_depth,
)

CATEGORIES_FILE = REPO_ROOT / "click3r/configs/hard3dbench/categories.json"
CATEGORIES = (
    "large_viewpoint",
    "moderate_viewpoint",
    "low_overlap",
    "large_baseline",
    "extreme_depth_range",
    "other",
)
THRESHOLDS = {
    "large_viewpoint": 60.0,
    "moderate_viewpoint": 30.0,
    "low_overlap": 0.2,
    "large_baseline": 0.3,
    "extreme_depth_range": 50.0,
}
# Table 7: (row, model, number of prompts in the result file name) and columns.
TABLE7_ROWS = (
    ("DUSt3R", "dust3r_pretrained", None),
    ("DUSt3R (retrained)", "dust3r_retrained", None),
    ("Click3R w/o prompts", "click3r", 0),
    ("Click3R (8 prompts)", "click3r", 8),
)
TABLE7_COLUMNS = {
    "large_viewpoint": "Lg. viewpoint",
    "moderate_viewpoint": "Mod. viewpoint",
    "low_overlap": "Low overlap",
    "large_baseline": "Lg. baseline",
    "extreme_depth_range": "Ext. depth range",
    "all": "All",
}
GAIN = ("Click3R (8 prompts)", "DUSt3R (retrained)")


def rotation_angle_deg(c2w1, c2w2):
    """Angle of the relative rotation R1^T R2 in degrees (float64)."""
    relative = (
        np.asarray(c2w1, np.float64)[:3, :3].T @ np.asarray(c2w2, np.float64)[:3, :3]
    )
    cos = np.clip((np.clip(np.trace(relative), -1.0, 3.0) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def overlap_ratio(depth1, K1, c2w1, K2, c2w2, step=7):
    """Fraction of valid view-1 depth pixels (every `step`-th) inside view 2.

    Depth and intrinsics are those of the 224 x 224 model inputs. Fewer than five
    valid pixels count as no overlap.
    """
    height, width = depth1.shape
    depth = np.asarray(depth1)[::step, ::step]
    K = np.asarray(K1, dtype=np.float64).copy()
    K[0, :] *= depth.shape[1] / width
    K[1, :] *= depth.shape[0] / height
    u, v = np.meshgrid(
        np.arange(depth.shape[1], dtype=np.float64),
        np.arange(depth.shape[0], dtype=np.float64),
    )
    valid = depth > 0.01
    if valid.sum() < 5:
        return 0.0
    z = depth[valid].astype(np.float64)
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    world = np.asarray(c2w1, np.float64) @ np.stack([x, y, z, np.ones_like(z)])
    cam2 = np.linalg.inv(np.asarray(c2w2, np.float64)) @ world
    front = cam2[2] > 0.01
    if not front.any():
        return 0.0
    K2 = np.asarray(K2, dtype=np.float64)
    points = cam2[:3, front]
    u2 = K2[0, 0] * points[0] / points[2] + K2[0, 2]
    v2 = K2[1, 1] * points[1] / points[2] + K2[1, 2]
    inside = (u2 >= 0) & (u2 < width) & (v2 >= 0) & (v2 < height)
    return float(inside.sum() / valid.sum())


def depth_stats(depth, name):
    """Median and max/min of the depth values above 0.01."""
    if not np.isfinite(depth).all():
        raise ValueError(f"{name}: non-finite ground-truth depth")
    valid = depth[depth > 0.01]
    if valid.size == 0:
        raise ValueError(f"{name}: no ground-truth depth above 0.01")
    return float(np.median(valid)), float(valid.max() / valid.min())


def features(c2w1, c2w2, depth1, depth2, overlap, name):
    """Geometric features of one pair; every value must be finite."""
    c2w1, c2w2 = np.asarray(c2w1, np.float64), np.asarray(c2w2, np.float64)
    baseline = float(np.linalg.norm(c2w2[:3, 3] - c2w1[:3, 3]))
    (median1, range1), (median2, range2) = (
        depth_stats(depth1, f"{name} view 1"),
        depth_stats(depth2, f"{name} view 2"),
    )
    median = float(np.mean([median1, median2]))
    values = {
        "relative_rotation_deg": rotation_angle_deg(c2w1, c2w2),
        "overlap_ratio": overlap,
        "baseline": baseline,
        "median_depth": median,
        "baseline_depth_ratio": baseline / median,
        "depth_range_ratio": max(range1, range2),
    }
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f"{name}: non-finite feature {values}")
    return values


def pair_features(sample, split, folders, ood_root):
    """Features of one pair. Overlap uses the 224 x 224 inputs; depth statistics the
    In-Distribution loader crop or the full-resolution Out-of-Distribution depth."""
    pair = load_pair(sample, split, folders, ood_root)
    depths = pair["depth1"], pair["depth2"]
    if split == "ood":
        cameras = ood_cameras(str(ood_root), sample["dataset"])
        records = [cameras[f"{sample['scene']}/{sample[f'frame{i}']}"] for i in (1, 2)]
        depths = [
            read_ood_depth(Path(ood_root) / r["depth"], sample["dataset"], r["size"])
            for r in records
        ]
    overlap = overlap_ratio(
        pair["depth1"], pair["K1"], pair["c2w1"], pair["K2"], pair["c2w2"]
    )
    return features(pair["c2w1"], pair["c2w2"], *depths, overlap, sample["sample_id"])


def assign(values):
    """Categories of one pair, in the order of CATEGORIES."""
    labels = []
    rotation = values["relative_rotation_deg"]
    if rotation > THRESHOLDS["large_viewpoint"]:
        labels.append("large_viewpoint")
    elif rotation > THRESHOLDS["moderate_viewpoint"]:
        labels.append("moderate_viewpoint")
    if values["overlap_ratio"] < THRESHOLDS["low_overlap"]:
        labels.append("low_overlap")
    if values["baseline_depth_ratio"] > THRESHOLDS["large_baseline"]:
        labels.append("large_baseline")
    if values["depth_range_ratio"] > THRESHOLDS["extreme_depth_range"]:
        labels.append("extreme_depth_range")
    return labels or ["other"]


def counts(records):
    tally = Counter(c for r in records for c in r["categories"])
    return {c: tally[c] for c in CATEGORIES}


def build(splits, folders, ood_root):
    records, per_split = [], {}
    for split in splits:
        split_records = [
            {
                "sample_id": sample["sample_id"],
                "categories": assign(pair_features(sample, split, folders, ood_root)),
            }
            for sample in tqdm(load_list("two_view", split)[0], desc=f"labels {split}")
        ]
        per_split[SPLIT_NAMES[split]] = {
            "num_samples": len(split_records),
            "counts": counts(split_records),
        }
        records += split_records
    metadata = {
        "benchmark": "Hard3DBench",
        "splits": per_split,
        "generated_by": "evaluation/hard3dbench/labels.py",
    }
    return {"metadata": metadata, "samples": records}


def maa30_with_misses(samples):
    """mAA@30 (%) over all samples; a failed (null) pose counts as a miss."""
    errors = np.array([(s["rre"], s["rte"]) for s in samples], dtype=np.float64)
    errors[~np.isfinite(errors).all(axis=1)] = np.inf
    return float(calculate_auc_np(errors[:, 0], errors[:, 1], max_threshold=30) * 100)


def table7(results_dir):
    """Paper Table 7 (Markdown) from the 2-view result files in `results_dir`."""
    labels = {
        r["sample_id"]: r["categories"]
        for r in json.loads(CATEGORIES_FILE.read_text())["samples"]
    }
    listed = {
        split: [s["sample_id"] for s in load_list("two_view", split)[0]]
        for split in SPLIT_NAMES
    }
    maa = {}
    for row, model, num_prompts in TABLE7_ROWS:
        samples = []
        for split in SPLIT_NAMES:
            path = result_path(results_dir, "two_view", split, model, num_prompts)
            if not path.is_file():
                raise SystemExit(f"Table 7 needs {path}")
            found = json.loads(path.read_text())["samples"]
            if [s["sample_id"] for s in found] != listed[split]:
                raise SystemExit(f"{path}: expected every listed pair once, in order")
            samples += found
        maa[row] = {
            column: maa30_with_misses(
                [
                    s
                    for s in samples
                    if column == "all" or column in labels[s["sample_id"]]
                ]
            )
            for column in TABLE7_COLUMNS
        }
    pairs = Counter(c for ids in listed.values() for s in ids for c in labels[s])
    pairs["all"] = sum(len(ids) for ids in listed.values())
    lines = [
        "## 2-view mAA@30 (%) by failure category",
        "",
        "| Method | " + " | ".join(TABLE7_COLUMNS.values()) + " |",
        "|---|" + "---:|" * len(TABLE7_COLUMNS),
    ]
    for row, values in maa.items():
        cells = [f"{values[c]:.2f}" for c in TABLE7_COLUMNS]
        if row == GAIN[0]:
            gain = [values[c] - maa[GAIN[1]][c] for c in TABLE7_COLUMNS]
            cells = [f"{cell} ({g:+.2f})" for cell, g in zip(cells, gain, strict=True)]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    lines += [
        "| Pairs | " + " | ".join(str(pairs[c]) for c in TABLE7_COLUMNS) + " |",
        "",
        "Categories from categories.json; a pair can be in several. Every listed In- "
        "and Out-of-Distribution pair is in the denominator and a failed pose counts "
        "as a miss.",
        f"In parentheses: difference to {GAIN[1]}.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split", nargs="+", choices=list(SPLIT_NAMES), default=list(SPLIT_NAMES)
    )
    parser.add_argument(
        "--data-root",
        help="processed DUSt3R data: BlendedMVS, Habitat and WildRGB-D (In-Distribution)",
    )
    parser.add_argument(
        "--co3d-root",
        help="Table 1 data root holding the held-out CO3D test sequences "
        "(In-Distribution CO3D)",
    )
    parser.add_argument("--ood-root", help="prepared Out-of-Distribution data")
    parser.add_argument(
        "--output", type=Path, help="write the labels to this JSON file"
    )
    parser.add_argument(
        "--table7",
        type=Path,
        metavar="RESULTS",
        help="print Table 7 from the 2-view result files in this Table 2 results "
        "directory and the shipped labels; needs no data",
    )
    args = parser.parse_args()
    if args.table7:
        if args.output:
            parser.error("--table7 runs on its own")
        print(table7(args.table7))
        return
    if "indist" in args.split and not (args.data_root and args.co3d_root):
        parser.error("--split indist needs --data-root and --co3d-root")
    if "ood" in args.split and not args.ood_root:
        parser.error("--split ood needs --ood-root")
    if not args.output:
        parser.error("give --output or --table7")
    folders = None
    if "indist" in args.split:
        folders = indist_folders(args.data_root, args.co3d_root)
    labels = build(args.split, folders, args.ood_root)
    for name, split in labels["metadata"]["splits"].items():
        print(f"{name}: {split['num_samples']} pairs, {split['counts']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(labels, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
