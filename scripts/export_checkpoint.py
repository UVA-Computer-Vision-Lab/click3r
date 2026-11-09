"""Export model weights and metadata from a trusted local training checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--training-config",
        type=Path,
        help="Published recipe specifying the effective training objective",
    )
    args = p.parse_args()
    ckpt = torch.load(args.source, map_location="cpu", mmap=True, weights_only=False)
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model"].items()
    }
    training_args = dict(vars(ckpt["args"]))
    # Some training checkpoints store these options under SAM-prefixed names.
    for name, previous in {
        "overlap_matching": "sam_use_overlap_matching",
        "evaluation_prompt_count": "sam_points_per_view",
    }.items():
        training_args.setdefault(name, training_args.get(previous))
    metadata = {
        k: training_args.get(k)
        for k in [
            "model",
            "epochs",
            "batch_size",
            "accum_iter",
            "world_size",
            "seed",
            "train_criterion",
            "test_criterion",
            "lr",
            "min_lr",
            "warmup_epochs",
            "weight_decay",
            "amp",
            "overlap_matching",
            "evaluation_prompt_count",
        ]
    }
    metadata.update(
        epoch=ckpt["epoch"],
        format="click3r-model-only-v1",
        source_filename=args.source.name,
    )
    if args.training_config:
        recipe = json.loads(args.training_config.read_text())
        metadata["train_criterion"] = recipe["train_criterion"]
        metadata["training_config"] = args.training_config.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "metadata": metadata}, args.output)
    digest = hashlib.sha256()
    with args.output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata["sha256"] = digest.hexdigest()
    metadata["bytes"] = args.output.stat().st_size
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
