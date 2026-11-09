#!/usr/bin/env python3
"""Hard3DBench 5-view evaluation (Table 2).

Click3R, DUSt3R and MASt3R predict the 20 ordered pairs of each tuple, then DUSt3R's
global alignment recovers the five cameras; VGGT and Pi3 predict the five views in
one forward. Protocol and result format: docs/eval.md.

  python evaluation/hard3dbench/five_view.py --split ood --model click3r \\
      --checkpoint checkpoints/click3r.pth --num-prompts 0 1 4 8 \\
      --ood-root /path/to/hard3dbench_ood --results-dir outputs/table2

  python evaluation/hard3dbench/five_view.py --split indist --model pi3_518 \\
      --data-root /path/to/processed_data --co3d-root /path/to/table1_data \\
      --results-dir outputs/table2

Results go to <results-dir>/five_view_<split>/<model>_k<K>.json for Click3R and
<model>.json for the other models; the Table 2 cells of each file are printed at
the end. Complete files are skipped when the command is rerun.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/dust3r"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "third_party/segment-anything")
)

import numpy as np
import torch
from tqdm import tqdm

from click3r.metrics import rotation_angle_single, translation_angle_single
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from evaluation.hard3dbench import baselines, common, data, two_view

NUM_VIEWS = 5
VIEW_PAIRS = tuple((i, j) for i in range(NUM_VIEWS) for j in range(i + 1, NUM_VIEWS))
ALIGNMENT = {"init": "msp", "niter": 300, "schedule": "cosine", "lr": 0.01}


# ─── inputs ──────────────────────────────────────────────────────────────────


def frame_stem(dataset, frame):
    """File stem (data.INDIST_FILES) of an In-Distribution frame of the 5-view list.

    The 5-view list names frames as the DUSt3R loaders do: CO3D and WildRGB-D
    frames keep their .jpg suffix and Habitat views count from 0, while the
    file names count Habitat views from 1.
    """
    if dataset in ("Co3d_test", "WildRGBD_test"):
        return frame.removesuffix(".jpg")
    if dataset == "Habitat_val":
        key, view = frame.rsplit("_", 1)
        return f"{key}_{int(view) + 1}"
    return frame


def load_tuple(sample, split, size, folders=None, ood_root=None):
    """The views of one 5-view sample: img (1, 3, size, size), depth, K, c2w.

    In-Distribution frames are DUSt3R dataset views (data.load_indist_view, read
    from `folders`), as in the paper's 5-view runs. Out-of-Distribution frames use
    DUSt3R's load-and-crop at that size.
    """
    dataset, scene = sample["dataset"], sample["scene"]
    if split == "indist":
        return [
            data.load_indist_view(
                folders, dataset, scene, frame_stem(dataset, frame), size
            )
            for frame in sample["frames"]
        ]
    return [
        data.load_ood_view(ood_root, dataset, scene, frame, size)
        for frame in sample["frames"]
    ]


# ─── predictions ─────────────────────────────────────────────────────────────


def pairwise_predictions(model, views):
    """Predict the 20 ordered pairs of the complete graph without prompts.

    DUSt3R's inference runs four pairs per forward and disables autocast, so this
    pass is float32 for every precision.
    """
    device = next(model.parameters()).device
    images = [
        {
            "img": view["img"].to(device),
            "true_shape": torch.tensor([view["img"].shape[-2:]], dtype=torch.int32),
            "idx": index,
            "instance": str(index),
        }
        for index, view in enumerate(views)
    ]
    pairs = make_pairs(images, scene_graph="complete", symmetrize=True)
    return pairs, inference(pairs, model, device, batch_size=4, verbose=False)


@torch.no_grad()
def add_prompts(model, pairs, output, views, num_prompts, precision):
    """Replace each pair's prediction in `output` by one with up to K prompts.

    A pair without a valid ground-truth prompt keeps its prompt-free prediction.
    Returns the number of prompted pairs.
    """
    device = next(model.parameters()).device
    prompted = 0
    for index, (view1, view2) in enumerate(pairs):
        first, second = views[int(view1["instance"])], views[int(view2["instance"])]
        coords = common.sample_prompts(
            first["depth"],
            second["depth"],
            first["c2w"],
            second["c2w"],
            first["K"],
            second["K"],
            num_prompts,
        )
        if coords is None:
            continue
        prompts = common.prompt_inputs(*coords, num_prompts, *first["depth"].shape)
        inputs = [
            dict(view, sparse_prompts={k: v.to(device) for k, v in sparse.items()})
            for view, sparse in zip((view1, view2), prompts, strict=True)
        ]
        with common.autocast(device, precision):
            predictions = model(*inputs)
        for name, prediction in zip(("pred1", "pred2"), predictions, strict=True):
            for key, value in prediction.items():
                if key in output[name]:
                    output[name][key][index] = value[0].cpu()
        prompted += 1
    return prompted


def align(output, device):
    """DUSt3R global alignment; cameras and pointmaps in the first camera's frame.

    The msp initialization overwrites every randomly initialized parameter, so
    the result does not depend on the random seed.
    """
    aligner = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False
    )
    aligner.compute_global_alignment(**ALIGNMENT)
    cam2world = aligner.get_im_poses().detach().cpu().numpy()
    first = np.linalg.inv(cam2world[0])
    points = []
    for world in aligner.get_pts3d():
        world = world.detach().cpu().numpy()
        local = (first[:3, :3] @ world.reshape(-1, 3).T).T + first[:3, 3]
        points.append(local.reshape(world.shape).astype(np.float32))
    return cam2world, points


def evaluate_aligned(model, views, ks, precision):
    """Per-K rows of one tuple for Click3R, DUSt3R or MASt3R.

    The prompt-free pairwise pass is shared by all K; each K is aligned separately.
    """
    device = next(model.parameters()).device
    pairs, output = pairwise_predictions(model, views)
    rows = {}
    for k in ks:
        current = copy.deepcopy(output)
        prompted = add_prompts(model, pairs, current, views, k, precision) if k else 0
        cam2world, points = align(current, device)
        rows[k] = score(cam2world, points, views)
        if k:
            rows[k]["num_prompted_pairs"] = prompted
    return rows


def evaluate_cameras(key, model, views, precision):
    """Row of one tuple for VGGT or Pi3 (one forward over the five views)."""
    images = torch.cat([view["img"] for view in views])
    prediction = baselines.predict_views(
        key, model, images, precision, depth_head=False
    )
    return score(prediction["cam2world"], list(prediction["pts3d"]), views)


# ─── metrics ─────────────────────────────────────────────────────────────────


def pair_pose_errors(cam2world, gt_c2w):
    """RRE and RTE (degrees) of the relative poses of the 10 view pairs (i < j)."""
    rre, rte = [], []
    for i, j in VIEW_PAIRS:
        pred_rot = cam2world[i, :3, :3].T @ cam2world[j, :3, :3]
        gt_rot = gt_c2w[i, :3, :3].T @ gt_c2w[j, :3, :3]
        pred_t = cam2world[i, :3, :3].T @ (cam2world[j, :3, 3] - cam2world[i, :3, 3])
        gt_t = gt_c2w[i, :3, :3].T @ (gt_c2w[j, :3, 3] - gt_c2w[i, :3, 3])
        rotation = rotation_angle_single(
            torch.from_numpy(pred_rot).float(), torch.from_numpy(gt_rot).float()
        )
        translation = translation_angle_single(
            torch.from_numpy(pred_t).float(),
            torch.from_numpy(gt_t).float(),
            handle_ambiguity=True,
        )
        rre.append(common.finite_or_none(float(rotation.item())))
        rte.append(common.finite_or_none(float(translation.item())))
    return rre, rte


def gt_pointmaps(views):
    """Ground-truth pointmaps of all views in the first camera's frame (float32)."""
    first = np.linalg.inv(views[0]["c2w"].astype(np.float64))
    points, valid = [], []
    for view in views:
        depth = view["depth"].astype(np.float64)
        camera = common.camera_points(depth, view["K"].astype(np.float64))
        c2w = view["c2w"].astype(np.float64)
        world = (c2w[:3, :3] @ camera.reshape(-1, 3).T).T + c2w[:3, 3]
        local = (first[:3, :3] @ world.T).T + first[:3, 3]
        points.append(local.reshape(camera.shape).astype(np.float32))
        valid.append(depth > 1e-3)
    return points, valid


def score(cam2world, points, views):
    gt_c2w = np.stack([view["c2w"] for view in views])
    rre, rte = pair_pose_errors(cam2world, gt_c2w)
    gt, valid = gt_pointmaps(views)
    return {
        "pair_rre": rre,
        "pair_rte": rte,
        "pointmap_error": common.pointmap_error_views(points, gt, valid),
    }


# ─── command line ────────────────────────────────────────────────────────────


def main(argv=None):
    args = two_view.parse_args(argv, __doc__)
    device = torch.device(args.device)
    samples, list_name, list_sha256 = common.load_list("five_view", args.split)
    samples = samples[: args.limit]
    sample_ids = [sample["sample_id"] for sample in samples]
    baseline = baselines.BASELINES.get(args.model)
    camera_model = baseline is not None and baseline.family in baselines.CAMERA_FAMILIES
    if baseline is None:
        size = common.IMAGE_SIZE
        weights = {
            "checkpoint": Path(args.checkpoint).name,
            "checkpoint_sha256": common.sha256(args.checkpoint),
        }
    else:
        size = baseline.resolution
        weights = baselines.weights_metadata(args.model)
    precision = (
        baselines.precision_for(args.model, device)
        if camera_model
        else common.precision_for(device)
    )
    paths, pending = {}, {}
    for k in sorted(set(args.num_prompts)):
        suffix = k if args.model == "click3r" else None
        path = common.result_path(
            args.results_dir, "five_view", args.split, args.model, suffix
        )
        paths[k] = path
        run = dict(
            weights,
            precision=precision if camera_model or k else "float32",
            device=str(device),
        )
        meta = common.run_metadata(
            "five_view", args.split, args.model, k, list_name, list_sha256, **run
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
        roots = two_view.data_roots(args)
        for sample in tqdm(samples, desc=f"{args.model} {args.split}"):
            views = load_tuple(sample, args.split, size, *roots)
            if camera_model:
                rows = {0: evaluate_cameras(args.model, model, views, precision)}
            else:
                rows = evaluate_aligned(model, views, list(pending), precision)
            identity = {"sample_id": sample["sample_id"], "dataset": sample["dataset"]}
            for k, (_, _, samples_k) in pending.items():
                samples_k.append({**identity, **rows[k]})
        for path, meta, rows in pending.values():
            common.write_results(path, meta, rows)
            print(f"{path}: written")
    for k, path in paths.items():
        print(common.table2_line(path, "five_view", args.split, args.model, k))


if __name__ == "__main__":
    main()
