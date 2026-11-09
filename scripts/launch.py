"""One-command environment setup, checkpoint acquisition, and launch."""

import argparse
import os
from pathlib import Path
import sys

from download_checkpoints import RELEASE_CHECKPOINTS, URLS, acquire
from setup_environment import ROOT, setup

CHECKSUM = RELEASE_CHECKPOINTS["click3r"]["sha256"]


def parser_for(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Use the current Python environment instead of .venv",
    )
    if mode == "demo":
        parser.add_argument(
            "--checkpoint-source",
            default=os.environ.get("CLICK3R_CHECKPOINT_SOURCE")
            or RELEASE_CHECKPOINTS["click3r"]["source"],
            help="Click3R download URL or local exported checkpoint",
        )
        parser.add_argument(
            "--checkpoint-sha256",
            default=CHECKSUM,
            help="Checksum of the supplied release checkpoint",
        )
        parser.add_argument(
            "--model", type=Path, default=ROOT / "checkpoints/click3r.pth"
        )
        parser.add_argument(
            "--dust3r-model",
            type=Path,
            default=ROOT / "checkpoints" / URLS["dust3r"][1],
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo")
        parser.add_argument("--server-name", default="127.0.0.1")
        parser.add_argument(
            "--server-port",
            type=int,
            help="Default: the first free port from 7860",
        )
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--examples", type=Path, default=ROOT / "assets/examples/manifest.json"
        )
    else:
        parser.add_argument("--data-root", type=Path, required=True)
        parser.add_argument(
            "--output-dir", type=Path, default=ROOT / "outputs/training"
        )
        parser.add_argument(
            "--gpus", type=int, help="Processes on this machine (default: 4; smoke: 1)"
        )
        parser.add_argument(
            "--pretrained", type=Path, default=ROOT / "checkpoints" / URLS["croco"][1]
        )
        parser.add_argument(
            "--pretrained-source",
            default=URLS["croco"][0],
            help="Initialization URL or local checkpoint",
        )
        parser.add_argument(
            "--config", type=Path, default=ROOT / "click3r/configs/train_click3r.json"
        )
        parser.add_argument(
            "--smoke",
            action="store_true",
            help="Two training batches and bounded validation on CO3D",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate paths and print the training recipe",
        )
        parser.add_argument("--no-compile", action="store_true")
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--accum-iter", type=int)
        parser.add_argument("--num-workers", type=int)
        parser.add_argument("--smoke-epochs", type=int, default=1)
    return parser


def main(mode=None):
    mode = mode or sys.argv.pop(1)
    parser = parser_for(mode)
    args = parser.parse_args()
    # Resolve paths against the caller's directory; the launch below runs from ROOT.
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    # Validate local inputs before installing packages or downloading GBs.
    if mode == "demo":
        if not args.model.is_file() and not args.checkpoint_source:
            parser.error(
                "Supply --checkpoint-source URL_OR_FILE (or CLICK3R_CHECKPOINT_SOURCE)."
            )
        if not args.examples.is_file():
            parser.error(f"Example manifest not found: {args.examples}")
    else:
        if not args.data_root.is_dir():
            parser.error(f"Processed data directory not found: {args.data_root}")
        if args.gpus is not None and args.gpus < 1:
            parser.error("--gpus must be positive")
    python = sys.executable if args.skip_setup else str(setup())
    if mode == "demo":
        # Each checkout uses its own cache on machines with a shared /tmp.
        if "GRADIO_TEMP_DIR" not in os.environ:
            cache = args.output.resolve() / ".gradio"
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["GRADIO_TEMP_DIR"] = str(cache)
        if args.checkpoint_source:
            acquire(args.checkpoint_source, args.model, args.checkpoint_sha256)
        if not args.dust3r_model.is_file():
            acquire(URLS["dust3r"][0], args.dust3r_model)
        command = [
            python,
            "-m",
            "click3r.demo",
            "--model",
            str(args.model),
            "--dust3r-model",
            str(args.dust3r_model),
            "--device",
            args.device,
            "--output",
            str(args.output),
            "--server-name",
            args.server_name,
            "--examples",
            str(args.examples),
        ]
        if args.server_port is not None:
            command += ["--server-port", str(args.server_port)]
        command += [f"--{name}" for name in ("share", "debug") if getattr(args, name)]
    else:
        if not args.pretrained.is_file():
            acquire(args.pretrained_source, args.pretrained)
        gpus = args.gpus or (1 if args.smoke else 4)
        command = [python]
        if not args.dry_run and gpus > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpus}",
            ]
        command += [str(ROOT / "click3r/train.py")]
        for name in (
            "data_root",
            "output_dir",
            "pretrained",
            "config",
            "batch_size",
            "accum_iter",
            "num_workers",
            "smoke_epochs",
        ):
            value = getattr(args, name)
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
        command += [
            "--" + name.replace("_", "-")
            for name in ("smoke", "dry_run", "no_compile")
            if getattr(args, name)
        ]
        if args.dry_run:
            os.environ["WORLD_SIZE"] = str(gpus)
    print("Launching " + mode + ".", flush=True)
    os.chdir(ROOT)
    os.execv(python, command)


if __name__ == "__main__":
    main()
