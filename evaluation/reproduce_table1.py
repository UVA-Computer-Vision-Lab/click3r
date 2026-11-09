"""Run the Table 1 protocol on the held-out CO3D, ScanNet++ and ARKitScenes sets."""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party/dust3r"),
    str(ROOT / "third_party/segment-anything"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/click3r.pth")
    p.add_argument(
        "--dust3r-pretrained",
        type=Path,
        default=ROOT / "checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    )
    p.add_argument(
        "--dust3r-retrained",
        type=Path,
        default=ROOT / "checkpoints/dust3r/retrained.pth",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--all-methods",
        action="store_true",
        help="Include MASt3R, VGGT and Pi3X; see docs/eval.md",
    )
    p.add_argument("--click3r-only", action="store_true")
    p.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Evaluate a prefix of each frozen 1000-pair sample",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume-from", type=Path)
    p.add_argument(
        "--sampling-manifest-dir",
        type=Path,
        default=ROOT / "click3r/configs/table1/samples",
        help="Frozen Table 1 sample lists; use --allow-new-sampling for new datasets",
    )
    p.add_argument(
        "--allow-new-sampling",
        action="store_true",
        help="Allow different dataset lengths (smoke fixtures or a new protocol)",
    )
    args = p.parse_args()
    from evaluation.summarize_table1 import ROWS, DATASETS

    curope = ROOT / "third_party/dust3r/croco/models/curope"
    if args.all_methods and not any(curope.glob("curope*.so")):
        p.error(
            "Pi3X needs the compiled CroCo RoPE extension. Build it first:\n"
            f"  cd {curope.relative_to(ROOT)} && python setup.py build_ext --inplace\n"
            "See docs/eval.md."
        )
    for path in [args.checkpoint] + (
        [] if args.click3r_only else [args.dust3r_pretrained, args.dust3r_retrained]
    ):
        if not path.is_file():
            p.error(f"Checkpoint not found: {path}")
    if args.all_methods and args.click3r_only:
        p.error("--all-methods and --click3r-only are mutually exclusive")
    if args.max_pairs < 0:
        p.error("--max-pairs must be nonnegative")
    if args.resume_from:
        for ds in DATASETS:
            manifest = args.resume_from / f"{ds}_sampling_224.json"
            if not manifest.is_file():
                p.error(f"Resume requires a sampling manifest: {manifest}")
            sampling = json.loads(manifest.read_text())
            count = min(1000, args.max_pairs) if args.max_pairs else 1000
            if (
                sampling["seed"] != 777
                or len(sampling["evaluated_indices"]) != count
                or sampling["canonical_order"] != list(DATASETS)
            ):
                p.error(f"Resume sampling differs from this run: {manifest}")
    command = [
        str(ROOT / "evaluation/benchmark.py"),
        "--data_root",
        str(args.data_root),
        "--click3r_checkpoint",
        str(args.checkpoint),
        "--dust3r_pretrained",
        str(args.dust3r_pretrained),
        "--dust3r_checkpoint",
        str(args.dust3r_retrained),
        "--output_dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--datasets",
        *DATASETS,
        "--num_samples",
        "1000",
        "--seed",
        "777",
        "--num_prompts",
        "1",
        "--sampling_budgets",
        "5",
        "--grid_cell_size",
        "14",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--max_pairs",
        str(args.max_pairs),
        "--highres_resolution",
        "518",
        "--skip_vggt_native",
        "--skip_pi3x_native",
    ]
    if not args.all_methods:
        command += [
            "--skip_mast3r",
            "--skip_vggt",
            "--skip_pi3x",
            "--skip_vggt_highres",
            "--skip_pi3x_highres",
        ]
    if args.click3r_only:
        command += ["--skip_dust3r_pretrained", "--skip_dust3r_trained"]
    if not args.allow_new_sampling:
        command += ["--sampling-manifest-dir", str(args.sampling_manifest_dir)]
    if args.resume_from:
        command += ["--resume_from", str(args.resume_from)]
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")
    expected_count = min(1000, args.max_pairs) if args.max_pairs else 1000
    required_models = {"click3r"}
    if not args.click3r_only:
        required_models |= {"dust3r_pretrained", "dust3r_trained"}
    failures = []
    for _, model, config in ROWS:
        if not args.all_methods and model not in required_models:
            continue
        for dataset in DATASETS:
            path = args.output_dir / f"{dataset}_{model}_{config}_results.json"
            if not path.is_file():
                failures.append(f"Missing result: {path.name}")
            else:
                result = json.loads(path.read_text())
                if result.get("num_samples") != expected_count:
                    failures.append(f"Incomplete result: {path.name}")
                if any(
                    not isinstance(result.get(key), (int, float))
                    or not math.isfinite(result[key])
                    for key in ("median_loss", "mAA_30")
                ):
                    failures.append(f"Invalid metrics: {path.name}")
    if failures:
        raise SystemExit("Table 1 run incomplete:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
