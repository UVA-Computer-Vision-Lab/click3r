"""Prepare one Hard3DBench Out-of-Distribution dataset from its original download.

The script writes the images and depth maps of every frame in ``two_view_ood.json``
and ``five_view_ood.json`` under ``--out`` at the paths the lists use, and their
cameras to ``<out>/<dataset>/cameras.json``. It then checks the result against
``ood_checksums.json``, the checksums of the data the paper evaluated. Each
``ood_<dataset>.py`` module names the ``--raw`` input it reads.

    python scripts/hard3dbench/prepare_ood.py --dataset dtu \\
        --raw /path/to/dtu_test_mvsnet.zip --out /path/to/hard3dbench_ood
    python scripts/hard3dbench/prepare_ood.py --dataset dtu --check-only \\
        --out /path/to/hard3dbench_ood
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ood_7scenes
import ood_dtu
import ood_eth3d
import ood_nrgbd
import ood_scannetv2
from PIL import Image

CONFIGS = Path(__file__).resolve().parents[2] / "click3r/configs/hard3dbench"
CHECKSUMS = CONFIGS / "ood_checksums.json"
MODULES = {
    "7scenes": ood_7scenes,
    "dtu": ood_dtu,
    "eth3d": ood_eth3d,
    "nrgbd": ood_nrgbd,
    "scannetv2": ood_scannetv2,
}
SHOWN = 10  # file names listed per problem


def needed_frames(dataset):
    """Return {scene: sorted frames} over both Out-of-Distribution lists."""
    frames = {}
    for name in ("two_view_ood.json", "five_view_ood.json"):
        for sample in json.loads((CONFIGS / name).read_text())["samples"]:
            if sample["dataset"] == dataset:
                names = sample.get("frames") or [sample["frame1"], sample["frame2"]]
                frames.setdefault(sample["scene"], set()).update(names)
    return {scene: sorted(names) for scene, names in sorted(frames.items())}


def canonical(value):
    """Round floats to six decimals and drop -0.0, so cameras.json is stable."""
    if isinstance(value, float):
        return round(value, 6) + 0.0
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def camera_bytes(cameras):
    """Canonical cameras.json bytes: sorted keys, rounded floats, no NaN."""
    text = json.dumps(canonical(cameras), sort_keys=True, indent=1, allow_nan=False)
    return (text + "\n").encode()


def decoded_sha256(path):
    """sha256 of the decoded image or depth array, by the rule in ood_checksums.json."""
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.parent.parent.name == "ground_truth_depth":
        # ETH3D depth is a raw float32 dump named .JPG; the --out path is not looked at.
        array = np.fromfile(path, dtype="<f4")
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare(dataset, raw, out):
    """Run the dataset module on the listed frames and write cameras.json."""
    result = MODULES[dataset].prepare(raw, out, needed_frames(dataset))
    (out / dataset / "cameras.json").write_bytes(camera_bytes(result))
    print(f"{dataset}: wrote {len(result['frames'])} frames and {dataset}/cameras.json")


def check(dataset, out):
    """Compare the files and cameras.json under ``out`` with ood_checksums.json."""
    expected = json.loads(CHECKSUMS.read_text())[dataset]
    files = expected["files"]
    problems = []
    cameras = out / dataset / "cameras.json"
    if not cameras.is_file():
        problems.append(f"{dataset}/cameras.json is missing")
    elif hashlib.sha256(cameras.read_bytes()).hexdigest() != expected["cameras_sha256"]:
        problems.append(f"{dataset}/cameras.json differs from the paper's cameras")
    present = [rel for rel in sorted(files) if (out / rel).is_file()]
    missing = sorted(set(files) - set(present))
    with ThreadPoolExecutor(8) as pool:
        digests = list(pool.map(decoded_sha256, [out / rel for rel in present]))
    wrong = [
        rel for rel, sha in zip(present, digests, strict=True) if sha != files[rel]
    ]
    labels = {"are missing": missing, "differ from the paper's data": wrong}
    for label, rels in labels.items():
        if rels:
            listed = "".join(f"\n  {rel}" for rel in rels[:SHOWN])
            more = f"\n  ... and {len(rels) - SHOWN} more" if len(rels) > SHOWN else ""
            problems.append(f"{len(rels)} of {len(files)} files {label}:{listed}{more}")
    for problem in problems:
        print(f"{dataset}: {problem}")
    if problems:
        raise SystemExit(f"{dataset}: FAILED, the data differs from {CHECKSUMS.name}")
    print(f"{dataset}: OK, cameras.json and all {len(files)} files match the paper")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(MODULES))
    parser.add_argument("--raw", type=Path, help="the downloaded dataset")
    parser.add_argument("--out", type=Path, required=True, help="the OOD data root")
    parser.add_argument(
        "--check-only", action="store_true", help="only check an existing --out"
    )
    args = parser.parse_args()
    if not args.check_only:
        if args.raw is None:
            parser.error("--raw is required unless --check-only is given")
        prepare(args.dataset, args.raw, args.out)
    check(args.dataset, args.out)


if __name__ == "__main__":
    main()
