"""Create the Table 1 layout using median pointmap loss and mAA@30."""

import argparse
import json
import math
from pathlib import Path

# Column order and the shared-RNG sampling order of the frozen manifests.
DATASETS = ("co3d", "scannetpp", "arkitscenes")
ROWS = [
    ("VGGT (518)", "vggt_518", "no_prompts"),
    ("Pi3X (518)", "pi3x_518", "no_prompts"),
    ("VGGT (224)", "vggt", "no_prompts"),
    ("Pi3X (224)", "pi3x", "no_prompts"),
    ("DUSt3R pretrained", "dust3r_pretrained", "no_prompts"),
    ("MASt3R (512 weights, 224 inputs)", "mast3r", "no_prompts"),
    ("DUSt3R retrained", "dust3r_trained", "no_prompts"),
    ("Click3R without prompts", "click3r", "no_prompts"),
    ("Click3R random oracle (K=1, 5 trials)", "click3r", "1p_best5"),
    ("Click3R grid oracle", "click3r", "grid_oracle_1p_pose"),
]


def summarize(directory):
    rows = []
    for label, model, config in ROWS:
        metrics = {}
        for ds in DATASETS:
            path = directory / f"{ds}_{model}_{config}_results.json"
            if path.is_file():
                result = json.loads(path.read_text())
                metrics[ds] = {
                    key: result[key] for key in ("median_loss", "mAA_30", "num_samples")
                }
        if metrics:
            rows.append(
                {"label": label, "model": model, "config": config, "datasets": metrics}
            )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--maa-tolerance", type=float, default=0.1)
    p.add_argument("--loss-tolerance", type=float, default=0.005)
    args = p.parse_args()
    rows = summarize(args.results)
    if not rows:
        p.error("No Table 1 result files found")
    text = []
    for metric, title, digits in [
        ("median_loss", "Median pointmap error", 3),
        ("mAA_30", "mAA@30 (%)", 2),
    ]:
        text += [
            f"## {title}",
            "",
            "| Method | " + " | ".join(DATASETS) + " |",
            "|---|" + "---:|" * len(DATASETS),
        ]
        for row in rows:
            values = [
                f"{row['datasets'][ds][metric]:.{digits}f}"
                if ds in row["datasets"]
                else "—"
                for ds in DATASETS
            ]
            text.append("| " + row["label"] + " | " + " | ".join(values) + " |")
        text.append("")
    print("\n".join(text))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "table1.json").write_text(
            json.dumps({"rows": rows}, indent=2) + "\n"
        )
        (args.output / "table1.md").write_text("\n".join(text))
    if args.compare:
        expected = {
            (r["model"], r["config"]): r
            for r in json.loads(args.compare.read_text())["rows"]
        }
        failures = []
        actual = {(r["model"], r["config"]): r for r in rows}
        for key, row in expected.items():
            for ds, values in row["datasets"].items():
                candidate = actual.get(key, {}).get("datasets", {}).get(ds)
                if candidate is None:
                    failures.append(f"{key}/{ds}: missing result")
                    continue
                for metric, tolerance in [
                    ("median_loss", args.loss_tolerance),
                    ("mAA_30", args.maa_tolerance),
                ]:
                    if (
                        not math.isfinite(candidate[metric])
                        or abs(candidate[metric] - values[metric]) > tolerance
                    ):
                        failures.append(
                            f"{key}/{ds}/{metric}: {candidate[metric]} vs {values[metric]}"
                        )
                if candidate["num_samples"] != values["num_samples"]:
                    failures.append(f"{key}/{ds}: sample count differs")
        if failures:
            raise SystemExit("Comparison failed:\n" + "\n".join(failures))
        print("All expected rows match within the requested tolerances.")


if __name__ == "__main__":
    main()
