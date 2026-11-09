"""Portable launcher for the saved Click3R checkpoint recipe."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "third_party/dust3r"),
    str(ROOT / "third_party/segment-anything"),
]

DATASETS = [
    ("BlendedMVS", "blendedmvs", "train", "val", 16, False),
    ("MegaDepth", "megadepth", "train", "val", 16, False),
    ("Co3d", "co3d", "train", "test", 16, True),
    ("StaticThings3D", "staticthings3d", None, None, 256, True),
    ("Waymo", "waymo", None, None, 16, False),
    ("ARKitScenes", "arkitscenes", "train", "test", 256, False),
    ("WildRGBD", "wildrgbd", "train", "test", 16, True),
    ("Habitat", "habitat", "train", "val", 16, False),
    ("ScanNetpp", "scannetpp", "train", None, 16, False),
]


def dataset_string(data_root, validation=False, smoke=False):
    result = []
    for cls, name, train_split, test_split, crop, mask in DATASETS:
        if smoke and name != "co3d":
            continue
        if validation and test_split is None:
            continue
        path = data_root / f"{name}_processed_for_dust3r"
        if not path.is_dir():
            raise FileNotFoundError(f"Missing dataset: {path}")
        split = test_split if validation else train_split
        params = [f"ROOT={str(path)!r}", "resolution=224"]
        if split:
            params.append(f"split={split!r}")
        if validation:
            params.append("seed=777")
        else:
            params += [f"aug_crop={crop}", "transform=ColorJitter"]
            if mask:
                params.append("mask_bg='rand'")
        if cls == "Habitat":
            params.insert(0, "10000" if validation else "100000")
        count = 8 if smoke else (1000 if validation else 100000)
        result.append(f"{count} @ {cls}({', '.join(params)})")
    return " + ".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "click3r/configs/train_click3r.json"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=ROOT / "checkpoints/croco/CroCo_V2_ViTLarge_BaseDecoder.pth",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum-iter", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for portability/debugging",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="One epoch, two batches, CO3D only"
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=1,
        help="Epoch limit for a smoke/resume check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print resolved arguments",
    )
    opts = parser.parse_args()
    if opts.smoke_epochs < 1:
        parser.error("--smoke-epochs must be positive")
    from click3r.training import get_args_parser, train

    cfg = json.loads(opts.config.read_text())
    cfg.update(
        train_dataset=dataset_string(opts.data_root, smoke=opts.smoke),
        test_dataset=dataset_string(opts.data_root, validation=True, smoke=opts.smoke),
        pretrained=str(opts.pretrained.resolve()),
        output_dir=str(opts.output_dir.resolve()),
    )
    for flag in ("batch_size", "accum_iter", "num_workers"):
        value = getattr(opts, flag)
        if value is not None:
            cfg[flag] = value
    if opts.smoke:
        cfg.update(
            epochs=opts.smoke_epochs,
            warmup_epochs=0,
            batch_size=1,
            num_workers=0,
            max_train_steps=2,
            print_freq=1,
            compile=False,
        )
    if opts.no_compile:
        cfg["compile"] = False
    if not opts.pretrained.is_file():
        raise FileNotFoundError(opts.pretrained)
    args = get_args_parser().parse_args(["--train_dataset", cfg["train_dataset"]])
    unknown = set(cfg) - set(vars(args))
    if unknown:
        raise ValueError(f"Unknown config options: {sorted(unknown)}")
    vars(args).update(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg["effective_batch_size"] = args.batch_size * args.accum_iter * world_size
    cfg["lr_schedule"] = "linear warmup followed by cosine decay"
    if opts.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    (opts.output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n"
    )
    try:
        train(args)
    finally:
        import torch.distributed as distributed

        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    main()
