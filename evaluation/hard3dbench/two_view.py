#!/usr/bin/env python3
"""Hard3DBench 2-view evaluation (Table 2).

Click3R and DUSt3R: each pair is predicted once without prompts (Click3R w/o
prompts, DUSt3R rows) and, for Click3R, once per requested K with up to K
ground-truth correspondence prompts. A pair with no valid prompt keeps its
prompt-free prediction. Cameras come from PnP on each predicted pointmap; a failed
PnP is written as null.

Baselines (mast3r_512, vggt_224, vggt_518, pi3_224, pi3_518) load their pinned
Hugging Face weights and predict one pair per forward at their own input size
(evaluation/hard3dbench/baselines.py). MASt3R cameras come from PnP as for DUSt3R;
VGGT and Pi3 predict their cameras.

Poses of In-Distribution pairs are scored on the DUSt3R dataset views; their
pointmap error is scored, as in the paper, on a second prediction from views
cropped like Out-of-Distribution ones (the resized and center-cropped whole frame,
depth without object mask).

  python evaluation/hard3dbench/two_view.py --split ood --model click3r \\
      --checkpoint checkpoints/click3r.pth --num-prompts 0 1 4 8 \\
      --ood-root /path/to/hard3dbench_ood --results-dir outputs/table2

  python evaluation/hard3dbench/two_view.py --split indist --model dust3r_pretrained \\
      --checkpoint checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth \\
      --data-root /path/to/processed_data --co3d-root /path/to/table1_data \\
      --results-dir outputs/table2

  python evaluation/hard3dbench/two_view.py --split ood --model vggt_518 \\
      --ood-root /path/to/hard3dbench_ood --results-dir outputs/table2

Results go to <results-dir>/two_view_<split>/<model>_k<K>.json for Click3R and
<model>.json for the other models; the Table 2 cells of each file are printed at
the end. Complete files are skipped when the command is rerun.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/dust3r"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "third_party/segment-anything")
)

import torch
from tqdm import tqdm

from evaluation.hard3dbench import baselines, common
from evaluation.hard3dbench.data import indist_folders, load_pair

MODELS = common.DUST3R_MODELS + tuple(baselines.BASELINES)


def result_row(pair, pts1, pts2, rre, rte):
    """Pose errors (null if not finite) and the pointmap error of one prediction."""
    gt = tuple(pair[key] for key in ("depth1", "depth2", "c2w1", "c2w2", "K1", "K2"))
    return {
        "rre": common.finite_or_none(rre),
        "rte": common.finite_or_none(rte),
        "pointmap_error": common.pointmap_error(pts1, pts2, *gt),
    }


def score(prediction, pair, pnp_keys=("K1", "K2")):
    """Score a pointmap prediction; cameras from PnP with the pair's ``pnp_keys`` K."""
    pts1, pts2, conf1, conf2 = prediction
    cameras = (pair["c2w1"], pair["c2w2"], *(pair[key] for key in pnp_keys))
    rre, rte = common.relative_pose_error(pts1, pts2, conf1, conf2, *cameras)
    return result_row(pair, pts1, pts2, rre, rte)


def evaluate_pair(model, pair, ks, precision):
    """Per-K result rows of one pair; the prompt-free forward is shared."""
    no_prompts = score(
        common.predict(model, pair["img1"], pair["img2"], precision), pair
    )
    gt = tuple(pair[key] for key in ("depth1", "depth2", "c2w1", "c2w2", "K1", "K2"))
    rows = {}
    for k in ks:
        if k == 0:
            rows[k] = no_prompts
            continue
        coords = common.sample_prompts(*gt, k)
        if coords is None:
            rows[k] = dict(no_prompts, num_prompts_used=0)
            continue
        prompts = common.prompt_inputs(*coords, k, *pair["depth1"].shape)
        prediction = common.predict(
            model, pair["img1"], pair["img2"], precision, prompts
        )
        rows[k] = dict(score(prediction, pair), num_prompts_used=len(coords[0]))
    return rows


def evaluate_baseline(key, model, pair, precision):
    """Result row of a Table 2 baseline on one pair loaded with center_crop."""
    if baselines.BASELINES[key].family == "mast3r":
        prediction = common.predict(model, pair["img1"], pair["img2"], precision)
        return score(prediction, pair, pnp_keys=("pnp_K1", "pnp_K2"))
    images = torch.cat([pair["img1"], pair["img2"]])
    prediction = baselines.predict_views(key, model, images, precision, depth_head=True)
    cameras = (pair["c2w1"], pair["c2w2"])
    rre, rte = baselines.camera_pose_error(prediction["cam2world"], *cameras)
    return result_row(pair, *prediction["pts3d"], rre, rte)


def evaluate_sample(key, model, sample, split, ks, precision, roots):
    """Per-K result rows of one sample; roots as returned by `data_roots`.

    In-Distribution pointmap errors come from a second prediction on whole-frame
    views; poses always come from the first.
    """

    def rows(whole_frame):
        baseline = baselines.BASELINES.get(key)
        if baseline is None:
            pair = load_pair(sample, split, *roots, whole_frame=whole_frame)
            return evaluate_pair(model, pair, ks, precision)
        size = baseline.resolution
        pair = load_pair(
            sample, split, *roots, size, center_crop=True, whole_frame=whole_frame
        )
        return {0: evaluate_baseline(key, model, pair, precision)}

    result = rows(whole_frame=False)
    if split == "indist":
        pointmaps = rows(whole_frame=True)
        result = {
            k: dict(row, pointmap_error=pointmaps[k]["pointmap_error"])
            for k, row in result.items()
        }
    return result


def parse_args(argv=None, description=__doc__):
    """Command line shared by two_view.py and five_view.py."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", required=True, choices=list(common.SPLIT_NAMES))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--checkpoint",
        help="Click3R/DUSt3R weights (baselines use their pinned Hugging Face weights)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        nargs="+",
        default=[0],
        help="K values; Click3R only for K > 0 (default: 0)",
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
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="first N samples of the list")
    args = parser.parse_args(argv)
    if args.split == "indist" and not (args.data_root and args.co3d_root):
        parser.error("--split indist needs --data-root and --co3d-root")
    if args.split == "ood" and not args.ood_root:
        parser.error("--split ood needs --ood-root")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if min(args.num_prompts) < 0:
        parser.error("--num-prompts must be non-negative")
    if args.model != "click3r" and args.num_prompts != [0]:
        parser.error(f"{args.model} has no prompt input; use --num-prompts 0")
    if args.model in baselines.BASELINES and args.checkpoint:
        parser.error(f"{args.model} loads its pinned weights; drop --checkpoint")
    if args.model not in baselines.BASELINES and not args.checkpoint:
        parser.error(f"{args.model} needs --checkpoint")
    return args


def data_roots(args):
    """(In-Distribution dataset folders, Out-of-Distribution root) of a command."""
    if args.split == "indist":
        return indist_folders(args.data_root, args.co3d_root), None
    return None, args.ood_root


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    samples, list_name, list_sha256 = common.load_list("two_view", args.split)
    samples = samples[: args.limit]
    sample_ids = [sample["sample_id"] for sample in samples]
    baseline = baselines.BASELINES.get(args.model)
    if baseline is None:
        precision = common.precision_for(device)
        weights = {
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_sha256": common.sha256(args.checkpoint),
        }
    else:
        precision = baselines.precision_for(args.model, device)
        weights = baselines.weights_metadata(args.model)
    run = dict(weights, precision=precision, device=str(device))
    paths, pending = {}, {}
    for k in sorted(set(args.num_prompts)):
        suffix = k if args.model == "click3r" else None
        path = common.result_path(
            args.results_dir, "two_view", args.split, args.model, suffix
        )
        paths[k] = path
        meta = common.run_metadata(
            "two_view", args.split, args.model, k, list_name, list_sha256, **run
        )
        if common.is_complete(path, meta, sample_ids):
            print(f"{path}: complete, skipped")
            continue
        pending[k] = (path, meta, [])

    if pending:
        if baseline is None:
            model = common.load_model(args.model, args.checkpoint, device)
        else:
            model = baselines.load_baseline(args.model, device)
        roots = data_roots(args)
        for sample in tqdm(samples, desc=f"{args.model} {args.split}"):
            rows = evaluate_sample(
                args.model, model, sample, args.split, list(pending), precision, roots
            )
            identity = {"sample_id": sample["sample_id"], "dataset": sample["dataset"]}
            for k, (_, _, samples_k) in pending.items():
                samples_k.append({**identity, **rows[k]})
        for path, meta, rows in pending.values():
            common.write_results(path, meta, rows)
            print(f"{path}: written")
    for k, path in paths.items():
        print(common.table2_line(path, "two_view", args.split, args.model, k))


if __name__ == "__main__":
    main()
