#!/usr/bin/env python3
# Portions adapted from DUSt3R (https://github.com/naver/dust3r): finalize_view()
# from dust3r/datasets/base/base_stereo_view_dataset.py, and the
# load_{co3d,scannetpp,arkitscenes,megadepth}_sequence() loaders from the
# _get_views() methods in dust3r/datasets/{co3d,scannetpp,arkitscenes,megadepth}.py.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified by the Click3R authors.
"""Evaluate Click3R pairwise edges in 3-5 view DUSt3R global alignment.

Click3R predicts each pair (edge) separately. DUSt3R's PointCloudOptimizer
performs multi-view alignment over the complete/swin edge graph.
"""

import argparse
import copy
import json
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "dust3r"))
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "segment-anything"))

from click3r.model import Click3R  # noqa: F401  # CLI model expression
from click3r.inference import inference as click3r_inference
from click3r.metrics import (
    calculate_auc_np,
    compute_pose_metrics_from_dust3r,
    rotation_angle_single,
    translation_angle_single,
)
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.losses import L21, Regr3D_ScaleShiftInv
from dust3r.datasets.arkitscenes import ARKitScenes
from dust3r.datasets.co3d import Co3d
from dust3r.datasets.megadepth import MegaDepth
from dust3r.datasets.scannetpp import ScanNetpp
from dust3r.datasets.base.base_stereo_view_dataset import transpose_to_landscape
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.image_pairs import make_pairs
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.utils.image import imread_cv2

from evaluation.benchmark import (
    MAX_P,
    _format_prompts_from_corrs,
    assign_to_pixel_grid,
    attach_random_prompts,
    compute_per_sample_loss,
    get_all_valid_correspondences,
)
from evaluation.baselines.loading import load_dust3r_model, load_click3r_model


MODEL_SPEC = (
    "Click3R(pos_embed='RoPE100',"
    " patch_embed_cls='ManyAR_PatchEmbed',"
    " img_size=(224, 224), head_type='linear',"
    " output_mode='pts3d', depth_mode=('exp', -float('inf'), float('inf')),"
    " conf_mode=('exp', 1, float('inf')),"
    " enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,"
    " dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/processed")
    p.add_argument(
        "--dust3r_pretrained",
        default="checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    )
    p.add_argument("--dust3r_checkpoint", required=True)
    p.add_argument("--click3r_checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["co3d", "scannetpp", "megadepth", "arkitscenes"],
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=[
            "dust3r_pretrained",
            "dust3r_trained",
            "click3r_no_prompts",
            "click3r_1p_best5",
            "click3r_4p_best5",
        ],
    )
    p.add_argument("--num_views", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--num_scenes", type=int, default=20)
    p.add_argument("--scene_graph", default="complete", choices=["complete", "swin"])
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--niter", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--best5_budget", type=int, default=5)
    p.add_argument("--oracle_cell_size", type=int, default=14)
    p.add_argument("--oracle_max_cells", type=int, default=8)
    p.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write results_partial.json every N completed scenes.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _to_batched_view(view, idx):
    """Match DUSt3R demo-style view dicts, plus GT fields for prompts/metrics."""
    height, width = view["true_shape"]
    out = {
        "img": view["img"].unsqueeze(0),
        "true_shape": torch.tensor([[height, width]], dtype=torch.int32),
        "idx": torch.tensor([idx], dtype=torch.int64),
        "instance": str(idx),
        "dataset": view["dataset"],
        "label": view["label"],
        "camera_pose": torch.from_numpy(view["camera_pose"]).float().unsqueeze(0),
        "camera_intrinsics": torch.from_numpy(view["camera_intrinsics"])
        .float()
        .unsqueeze(0),
        "pts3d": torch.from_numpy(view["pts3d"]).float().unsqueeze(0),
        "valid_mask": torch.from_numpy(view["valid_mask"]).bool().unsqueeze(0),
    }
    return out


def finalize_view(raw, local_idx):
    view = {
        "img": ImgNorm(raw["img"]),
        "true_shape": np.int32((raw["img"].size[1], raw["img"].size[0])),
        "depthmap": raw["depthmap"].astype(np.float32),
        "camera_pose": raw["camera_pose"].astype(np.float32),
        "camera_intrinsics": raw["camera_intrinsics"].astype(np.float32),
        "dataset": raw["dataset"],
        "label": raw["label"],
        "instance": raw["instance"],
    }
    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
    view["pts3d"] = pts3d
    view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)
    transpose_to_landscape(view)
    return _to_batched_view(view, local_idx)


def load_co3d_sequence(dataset, scene_key, frame_positions, resolution, rng):
    obj, instance = scene_key
    image_pool = dataset.scenes[scene_key]
    views = []

    for local_idx, pool_pos in enumerate(frame_positions):
        frame_id = image_pool[int(pool_pos)]
        metadata = np.load(dataset._get_metadatapath(obj, instance, frame_id))
        camera_pose = metadata["camera_pose"].astype(np.float32)
        intrinsics = metadata["camera_intrinsics"].astype(np.float32)

        image = imread_cv2(dataset._get_impath(obj, instance, frame_id))
        depthmap = dataset._read_depthmap(
            dataset._get_depthpath(obj, instance, frame_id), metadata
        )

        if dataset.mask_bg:
            mask = imread_cv2(
                dataset._get_maskpath(obj, instance, frame_id), cv2.IMREAD_UNCHANGED
            ).astype(np.float32)
            depthmap *= (mask / 255.0) > 0.1

        image, depthmap, intrinsics = dataset._crop_resize_if_necessary(
            image,
            depthmap,
            intrinsics,
            (resolution, resolution),
            rng=rng,
            info=dataset._get_impath(obj, instance, frame_id),
        )
        if (depthmap > 0).sum() < 16:
            raise ValueError("too few valid depth pixels")

        raw = {
            "img": image,
            "depthmap": depthmap,
            "camera_pose": camera_pose,
            "camera_intrinsics": intrinsics,
            "dataset": dataset.dataset_label,
            "label": osp.join(obj, instance),
            "instance": f"frame{frame_id:06d}",
        }
        views.append(finalize_view(raw, local_idx))

    return views


def select_positions(pool_len, num_views):
    return np.linspace(0, pool_len - 1, num_views).round().astype(int)


def load_scannetpp_sequence(dataset, scene_id, frame_positions, resolution, rng):
    image_indices = np.where(dataset.sceneids == scene_id)[0]
    selected = image_indices[frame_positions]
    views = []
    for local_idx, view_idx in enumerate(selected):
        scene_dir = osp.join(dataset.ROOT, dataset.scenes[scene_id])
        basename = dataset.images[view_idx]
        image = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
        depthmap = imread_cv2(
            osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED
        )
        depthmap = depthmap.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0
        intrinsics = dataset.intrinsics[view_idx].copy()
        image, depthmap, intrinsics = dataset._crop_resize_if_necessary(
            image,
            depthmap,
            intrinsics,
            (resolution, resolution),
            rng=rng,
            info=int(view_idx),
        )
        raw = {
            "img": image,
            "depthmap": depthmap,
            "camera_pose": dataset.trajectories[view_idx],
            "camera_intrinsics": intrinsics,
            "dataset": "ScanNet++",
            "label": dataset.scenes[scene_id] + "_" + basename,
            "instance": f"{int(view_idx)}",
        }
        views.append(finalize_view(raw, local_idx))
    return views


def load_arkitscenes_sequence(dataset, scene_id, frame_positions, resolution, rng):
    image_indices = np.where(dataset.sceneids == scene_id)[0]
    selected = image_indices[frame_positions]
    views = []
    for local_idx, view_idx in enumerate(selected):
        scene_dir = osp.join(dataset.ROOT, dataset.split, dataset.scenes[scene_id])
        basename = dataset.images[view_idx]
        image = imread_cv2(
            osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
        )
        depthmap = imread_cv2(
            osp.join(scene_dir, "lowres_depth", basename), cv2.IMREAD_UNCHANGED
        )
        depthmap = depthmap.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0
        intrinsics = dataset.intrinsics[view_idx].copy()
        image, depthmap, intrinsics = dataset._crop_resize_if_necessary(
            image,
            depthmap,
            intrinsics,
            (resolution, resolution),
            rng=rng,
            info=int(view_idx),
        )
        raw = {
            "img": image,
            "depthmap": depthmap,
            "camera_pose": dataset.trajectories[view_idx],
            "camera_intrinsics": intrinsics,
            "dataset": "arkitscenes",
            "label": dataset.scenes[scene_id] + "_" + basename,
            "instance": f"{int(view_idx)}",
        }
        views.append(finalize_view(raw, local_idx))
    return views


def megadepth_image_ids(dataset, scene_id):
    pairs = dataset.pairs[dataset.pairs["scene_id"] == scene_id]
    if len(pairs) == 0:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate([pairs["im1_id"], pairs["im2_id"]])).astype(
        np.int64
    )


def load_megadepth_sequence(dataset, scene_id, frame_positions, resolution, rng):
    image_ids = megadepth_image_ids(dataset, scene_id)
    selected = image_ids[frame_positions]
    scene, subscene = dataset.all_scenes[scene_id].split()
    seq_path = osp.join(dataset.ROOT, scene, subscene)
    views = []
    for local_idx, im_id in enumerate(selected):
        img = dataset.all_images[im_id]
        image = imread_cv2(osp.join(seq_path, img + ".jpg"))
        depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
        camera_params = np.load(osp.join(seq_path, img + ".npz"))
        intrinsics = np.float32(camera_params["intrinsics"])
        camera_pose = np.float32(camera_params["cam2world"])
        image, depthmap, intrinsics = dataset._crop_resize_if_necessary(
            image,
            depthmap,
            intrinsics,
            (resolution, resolution),
            rng=rng,
            info=(seq_path, img),
        )
        raw = {
            "img": image,
            "depthmap": depthmap.astype(np.float32),
            "camera_pose": camera_pose,
            "camera_intrinsics": intrinsics,
            "dataset": "MegaDepth",
            "label": osp.relpath(seq_path, dataset.ROOT),
            "instance": img,
        }
        views.append(finalize_view(raw, local_idx))
    return views


def build_dataset(name, args):
    roots = {
        "co3d": "co3d_processed_for_dust3r",
        "scannetpp": "scannetpp_processed_for_dust3r",
        "megadepth": "megadepth_processed_for_dust3r",
        "arkitscenes": "arkitscenes_processed_for_dust3r",
    }
    root = osp.join(args.data_root, roots[name])
    if name == "co3d":
        dataset = Co3d(
            mask_bg=True,
            split="test",
            ROOT=root,
            resolution=args.resolution,
            aug_crop=False,
            seed=args.seed,
        )
        scene_keys = list(dataset.scene_list)
    elif name == "scannetpp":
        dataset = ScanNetpp(
            split="train",
            ROOT=root,
            resolution=args.resolution,
            aug_crop=False,
            seed=args.seed,
        )
        scene_keys = sorted(np.unique(dataset.sceneids).astype(int).tolist())
    elif name == "arkitscenes":
        dataset = ARKitScenes(
            split="test",
            ROOT=root,
            resolution=args.resolution,
            aug_crop=False,
            seed=args.seed,
        )
        scene_keys = sorted(np.unique(dataset.sceneids).astype(int).tolist())
    elif name == "megadepth":
        dataset = MegaDepth(
            split="val",
            ROOT=root,
            resolution=args.resolution,
            aug_crop=False,
            seed=args.seed,
        )
        scene_keys = sorted(np.unique(dataset.pairs["scene_id"]).astype(int).tolist())
    else:
        raise ValueError(f"unknown dataset {name}")
    return dataset, scene_keys


def scene_pool_len(dataset_name, dataset, scene_key):
    if dataset_name == "co3d":
        return len(dataset.scenes[scene_key])
    if dataset_name in ("scannetpp", "arkitscenes"):
        return int((dataset.sceneids == scene_key).sum())
    if dataset_name == "megadepth":
        return len(megadepth_image_ids(dataset, scene_key))
    raise ValueError(dataset_name)


def load_sequence(dataset_name, dataset, scene_key, frame_positions, resolution, rng):
    if dataset_name == "co3d":
        return load_co3d_sequence(dataset, scene_key, frame_positions, resolution, rng)
    if dataset_name == "scannetpp":
        return load_scannetpp_sequence(
            dataset, scene_key, frame_positions, resolution, rng
        )
    if dataset_name == "arkitscenes":
        return load_arkitscenes_sequence(
            dataset, scene_key, frame_positions, resolution, rng
        )
    if dataset_name == "megadepth":
        return load_megadepth_sequence(
            dataset, scene_key, frame_positions, resolution, rng
        )
    raise ValueError(dataset_name)


def serialize_scene_key(scene_key):
    if isinstance(scene_key, tuple):
        return list(scene_key)
    if isinstance(scene_key, np.integer):
        return int(scene_key)
    return scene_key


def clone_view(v):
    out = {}
    for k, val in v.items():
        if isinstance(val, torch.Tensor):
            out[k] = val.clone()
        elif isinstance(val, np.ndarray):
            out[k] = val.copy()
        else:
            out[k] = val
    return out


def clone_pairs(pairs):
    return [(clone_view(a), clone_view(b)) for a, b in pairs]


def add_one_prompt_per_pair(pairs, seed):
    for edge_idx, pair in enumerate(pairs):
        attach_random_prompts(pair, num_prompts=1, seed=seed + edge_idx * 1009)


def _deep_clone_pair(pair):
    return copy.deepcopy(pair)


def _score_prompt_candidates(model, candidates, args):
    output = click3r_inference(
        candidates, model, args.device, batch_size=args.batch_size, verbose=False
    )
    criterion = Regr3D_ScaleShiftInv(L21, gt_scale=True)
    losses = []
    for idx in range(len(candidates)):
        losses.append(
            compute_per_sample_loss(
                criterion,
                output["pred1"],
                output["pred2"],
                output["view1"],
                output["view2"],
                idx,
            )
        )
    return losses


def _score_prompt_candidates_by_pose(model, candidates, args):
    output = click3r_inference(
        candidates, model, args.device, batch_size=args.batch_size, verbose=False
    )
    scores = []
    for idx in range(len(candidates)):
        try:
            r, t = compute_pose_metrics_from_dust3r(
                output["view1"],
                output["view2"],
                output["pred1"],
                output["pred2"],
                pair_idx=idx,
            )
            scores.append(max(float(r), float(t)))
        except Exception:
            scores.append(float("inf"))
    return scores


def _copy_sparse_prompts(src_pair, dst_pair):
    for src, dst in zip(src_pair, dst_pair, strict=False):
        if "sparse_prompts" in src:
            dst["sparse_prompts"] = {
                k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in src["sparse_prompts"].items()
            }


def _set_empty_prompts(pair):
    for image_id, view in enumerate(pair):
        h, w = view["true_shape"][0].tolist()
        view["sparse_prompts"] = {
            "coords": torch.zeros(1, MAX_P, 2),
            "image_ids": torch.full((1, MAX_P), image_id, dtype=torch.long),
            "pair_ids": torch.arange(MAX_P, dtype=torch.long).unsqueeze(0),
            "shape": torch.tensor([[h, w]], dtype=torch.long),
            "valid_mask": torch.zeros(1, MAX_P, dtype=torch.bool),
        }


def select_best_random_prompts_for_pairs(model, pairs, args, num_prompts):
    """For each pair, try args.best5_budget random sets of num_prompts prompts.

    Keeps the set with the lowest pairwise regression loss.
    """
    for edge_idx, pair in enumerate(pairs):
        candidates = []
        for k in range(args.best5_budget):
            cand = _deep_clone_pair(pair)
            attach_random_prompts(
                cand,
                num_prompts=num_prompts,
                seed=args.seed + edge_idx * 1009 + k * 9176,
            )
            assert (
                int(cand[0]["sparse_prompts"]["valid_mask"].sum().item()) <= num_prompts
            )
            candidates.append(cand)
        losses = _score_prompt_candidates(model, candidates, args)
        best = int(np.argmin(losses))
        _copy_sparse_prompts(candidates[best], pair)


def select_grid_oracle_prompts_for_pairs(model, pairs, args):
    prompt_dev = torch.device("cpu")
    rng = np.random.default_rng(args.seed)
    for edge_idx, pair in enumerate(pairs):
        _set_empty_prompts(pair)
        c1, c2, shapes = get_all_valid_correspondences(pair[0], pair[1], 0, prompt_dev)
        if c1 is None:
            continue
        H1, W1, _, _ = shapes
        cell_to_indices, _, _ = assign_to_pixel_grid(c1, H1, W1, args.oracle_cell_size)
        cell_ids = sorted(cell_to_indices)
        if args.oracle_max_cells and len(cell_ids) > args.oracle_max_cells:
            chosen = rng.choice(
                len(cell_ids), size=args.oracle_max_cells, replace=False
            )
            cell_ids = [cell_ids[int(i)] for i in sorted(chosen)]

        candidates = []
        for cell_rank, cell_id in enumerate(cell_ids):
            idxs = cell_to_indices[cell_id]
            corr_idx = idxs[(args.seed + edge_idx + cell_rank) % len(idxs)]
            cand = _deep_clone_pair(pair)
            prompts_v1, prompts_v2 = _format_prompts_from_corrs(
                c1[corr_idx : corr_idx + 1],
                c2[corr_idx : corr_idx + 1],
                shapes,
                prompt_dev,
                args.seed + edge_idx * 1009 + cell_rank,
            )
            cand[0]["sparse_prompts"] = prompts_v1
            cand[1]["sparse_prompts"] = prompts_v2
            candidates.append(cand)

        if not candidates:
            continue
        pose_scores = _score_prompt_candidates_by_pose(model, candidates, args)
        best = int(np.argmin(pose_scores))
        _copy_sparse_prompts(candidates[best], pair)


def run_global_alignment(model, views, config, args):
    pairs = make_pairs(
        views, scene_graph=args.scene_graph, prefilter=None, symmetrize=True
    )
    pairs = clone_pairs(pairs)
    if config == "click3r_1p_random":
        add_one_prompt_per_pair(pairs, args.seed)
    elif config == "click3r_4p_random":
        for edge_idx, pair in enumerate(pairs):
            attach_random_prompts(pair, num_prompts=4, seed=args.seed + edge_idx * 1009)
    elif config == "click3r_8p_random":
        for edge_idx, pair in enumerate(pairs):
            attach_random_prompts(pair, num_prompts=8, seed=args.seed + edge_idx * 1009)
    elif config == "click3r_1p_best5":
        select_best_random_prompts_for_pairs(model, pairs, args, num_prompts=1)
    elif config == "click3r_4p_best5":
        select_best_random_prompts_for_pairs(model, pairs, args, num_prompts=4)
    elif config == "click3r_grid_oracle":
        select_grid_oracle_prompts_for_pairs(model, pairs, args)
    elif config in ("dust3r_pretrained", "dust3r_trained", "click3r_no_prompts"):
        for v1, v2 in pairs:
            v1.pop("sparse_prompts", None)
            v2.pop("sparse_prompts", None)
    else:
        raise ValueError(config)

    output = click3r_inference(
        pairs, model, args.device, batch_size=args.batch_size, verbose=False
    )
    scene = global_aligner(
        output,
        device=args.device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=False,
    )
    loss = scene.compute_global_alignment(
        init="mst", niter=args.niter, schedule="linear", lr=0.01
    )
    pred = scene.get_im_poses().detach().cpu()
    return pred, float(loss if not isinstance(loss, list) else loss[-1])


def pose_errors_all_pairs(pred_c2w, gt_c2w):
    r_errors, t_errors = [], []
    for i in range(len(gt_c2w)):
        for j in range(i + 1, len(gt_c2w)):
            gt_rel = torch.inverse(gt_c2w[i]) @ gt_c2w[j]
            pred_rel = torch.inverse(pred_c2w[i]) @ pred_c2w[j]
            r_errors.append(
                float(rotation_angle_single(gt_rel[:3, :3], pred_rel[:3, :3]))
            )
            t_errors.append(
                float(translation_angle_single(gt_rel[:3, 3], pred_rel[:3, 3]))
            )
    return r_errors, t_errors


def summarize(r_errors, t_errors):
    out = {"num_pairs": len(r_errors)}
    if not r_errors:
        for th in (5, 15, 30):
            out[f"RRA_{th}"] = None
            out[f"RTA_{th}"] = None
            out[f"mAA_{th}"] = None
        out["R_median"] = None
        out["T_median"] = None
        return out
    for th in (5, 15, 30):
        r = np.asarray(r_errors)
        t = np.asarray(t_errors)
        out[f"RRA_{th}"] = float((r < th).mean() * 100)
        out[f"RTA_{th}"] = float((t < th).mean() * 100)
        out[f"mAA_{th}"] = float(calculate_auc_np(r, t, max_threshold=th) * 100)
    out["R_median"] = float(np.median(r_errors))
    out["T_median"] = float(np.median(t_errors))
    return out


def build_summary(accum, done):
    return {
        ds: {
            str(nv): {
                cfg: summarize(vals["r"], vals["t"])
                | {
                    "mean_global_align_loss": float(np.mean(vals["loss"]))
                    if vals["loss"]
                    else None,
                    "num_scenes": done[ds][nv],
                }
                for cfg, vals in cfgs.items()
            }
            for nv, cfgs in views.items()
        }
        for ds, views in accum.items()
    }


def write_results(path, args, summary, records, start, final=False):
    payload = {
        "args": vars(args),
        "semantics": {
            "click3r_1p_random": "one random GT correspondence prompt per directed edge",
            "click3r_4p_random": "four random GT correspondence prompts per directed edge",
            "click3r_8p_random": "eight random GT correspondence prompts per directed edge",
            "click3r_1p_best5": (
                "budget independent attempts; each attempt uses exactly one random "
                "GT correspondence prompt; select the lowest pairwise regression loss"
            ),
            "click3r_4p_best5": (
                "budget independent attempts; each attempt uses exactly four random "
                "GT correspondence prompts; select the lowest pairwise regression loss"
            ),
            "click3r_grid_oracle": (
                "one GT correspondence per populated grid cell, capped by oracle_max_cells; "
                "select the candidate with the lowest pairwise pose error"
            ),
            "datasets": "normal eval datasets: co3d, scannetpp, megadepth, arkitscenes",
        },
        "elapsed_sec": time.time() - start,
        "final": final,
        "summary": summary,
        "records": records,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _load_resume_state(output_dir, configs, datasets, num_views):
    """Load results_partial.json and rebuild accum/records/done for resumption."""
    partial_path = osp.join(output_dir, "results_partial.json")
    if not osp.exists(partial_path):
        return None, None, None, set()

    with open(partial_path) as f:
        payload = json.load(f)

    records = payload.get("records", [])
    if not records:
        return None, None, None, set()

    # Rebuild accum and done from records
    accum = {
        ds: {
            nv: {cfg: {"r": [], "t": [], "loss": []} for cfg in configs}
            for nv in num_views
        }
        for ds in datasets
    }
    done = {ds: {nv: 0 for nv in num_views} for ds in datasets}

    # Track which (dataset, nv, scene) combos are already completed
    completed_scenes = set()
    for rec in records:
        ds = rec["dataset"]
        nv = rec["num_views"]
        cfg = rec["config"]
        scene = rec["scene"]
        # Use str(scene) directly — matches str(serialize_scene_key(scene_key)) in main loop
        completed_scenes.add((ds, nv, str(scene)))
        if ds in accum and nv in accum[ds] and cfg in accum[ds][nv]:
            accum[ds][nv][cfg]["r"].extend(rec["r_errors"])
            accum[ds][nv][cfg]["t"].extend(rec["t_errors"])
            accum[ds][nv][cfg]["loss"].append(rec["loss"])

    # Count completed scenes per (dataset, nv) — a scene is done when all configs are present
    for ds in datasets:
        for nv in num_views:
            scene_cfgs = {}
            for rec in records:
                if rec["dataset"] == ds and rec["num_views"] == nv:
                    scene_str = str(rec["scene"])
                    scene_cfgs.setdefault(scene_str, set()).add(rec["config"])
            done[ds][nv] = sum(
                1 for s, cfgs in scene_cfgs.items() if len(cfgs) >= len(configs)
            )

    print(f"[resume] Loaded {len(records)} records from {partial_path}", flush=True)
    for ds in datasets:
        for nv in num_views:
            if done[ds][nv] > 0:
                print(f"  {ds} nv={nv}: {done[ds][nv]} scenes already done", flush=True)

    return accum, records, done, completed_scenes


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)

    rng = np.random.default_rng(args.seed)

    model_instance = eval(MODEL_SPEC)
    dust3r_pretrained_model = load_dust3r_model(
        args.dust3r_pretrained, model_instance, device
    )
    dust3r_model = load_dust3r_model(args.dust3r_checkpoint, eval(MODEL_SPEC), device)
    click3r_model = load_click3r_model(
        args.click3r_checkpoint, eval(MODEL_SPEC), device
    )

    all_configs = {
        "dust3r_pretrained": dust3r_pretrained_model,
        "dust3r_trained": dust3r_model,
        "click3r_no_prompts": click3r_model,
        "click3r_1p_random": click3r_model,
        "click3r_4p_random": click3r_model,
        "click3r_8p_random": click3r_model,
        "click3r_1p_best5": click3r_model,
        "click3r_4p_best5": click3r_model,
        "click3r_grid_oracle": click3r_model,
    }
    configs = {cfg: all_configs[cfg] for cfg in args.configs}

    # --- Resume from partial results if available ---
    resumed_accum, resumed_records, resumed_done, completed_scenes = _load_resume_state(
        args.output_dir, configs, args.datasets, args.num_views
    )
    if resumed_accum is not None:
        accum = resumed_accum
        records = resumed_records
        done = resumed_done
    else:
        accum = {
            ds: {
                nv: {cfg: {"r": [], "t": [], "loss": []} for cfg in configs}
                for nv in args.num_views
            }
            for ds in args.datasets
        }
        records = []
        done = {ds: {nv: 0 for nv in args.num_views} for ds in args.datasets}
        completed_scenes = set()

    start = time.time()

    for dataset_name in args.datasets:
        dataset, scene_keys = build_dataset(dataset_name, args)
        print(
            f"[dataset] {dataset_name}: {len(scene_keys)} candidate scenes", flush=True
        )

        for scene_key in scene_keys:
            if all(done[dataset_name][nv] >= args.num_scenes for nv in args.num_views):
                break
            pool_len = scene_pool_len(dataset_name, dataset, scene_key)
            if pool_len < max(args.num_views):
                continue

            for nv in args.num_views:
                if done[dataset_name][nv] >= args.num_scenes:
                    continue
                # Skip scenes already completed during a previous run
                scene_id_str = str(serialize_scene_key(scene_key))
                if (dataset_name, nv, scene_id_str) in completed_scenes:
                    continue
                positions = select_positions(pool_len, nv)
                try:
                    views = load_sequence(
                        dataset_name,
                        dataset,
                        scene_key,
                        positions,
                        args.resolution,
                        rng,
                    )
                    gt = [v["camera_pose"][0].cpu() for v in views]
                except Exception as exc:
                    print(
                        f"[skip] dataset={dataset_name} scene={scene_key} nv={nv}: {exc}",
                        flush=True,
                    )
                    continue

                print(
                    f"[scene] dataset={dataset_name} nv={nv} "
                    f"{done[dataset_name][nv] + 1}/{args.num_scenes} scene={scene_key}",
                    flush=True,
                )
                for cfg, model in configs.items():
                    try:
                        pred, loss = run_global_alignment(model, views, cfg, args)
                        r, t = pose_errors_all_pairs(pred, gt)
                        accum[dataset_name][nv][cfg]["r"].extend(r)
                        accum[dataset_name][nv][cfg]["t"].extend(t)
                        accum[dataset_name][nv][cfg]["loss"].append(loss)
                        rec = {
                            "dataset": dataset_name,
                            "num_views": nv,
                            "scene": serialize_scene_key(scene_key),
                            "config": cfg,
                            "loss": loss,
                            "r_errors": r,
                            "t_errors": t,
                        }
                        records.append(rec)
                        m = summarize(
                            accum[dataset_name][nv][cfg]["r"],
                            accum[dataset_name][nv][cfg]["t"],
                        )
                        print(
                            f"  {cfg}: loss={loss:.4f} mAA30={m['mAA_30']:.2f} "
                            f"RRA15={m['RRA_15']:.1f} RTA15={m['RTA_15']:.1f}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"  [fail] {cfg}: {type(exc).__name__}: {exc}", flush=True
                        )
                done[dataset_name][nv] += 1
                if (
                    args.save_every > 0
                    and done[dataset_name][nv] % args.save_every == 0
                ):
                    partial_summary = build_summary(accum, done)
                    write_results(
                        osp.join(args.output_dir, "results_partial.json"),
                        args,
                        partial_summary,
                        records,
                        start,
                        final=False,
                    )

    summary = build_summary(accum, done)
    write_results(
        osp.join(args.output_dir, "results.json"),
        args,
        summary,
        records,
        start,
        final=True,
    )
    print(json.dumps(summary, indent=2), flush=True)
    failures = []
    for dataset in args.datasets:
        for num_views in args.num_views:
            for config in configs:
                result = summary[dataset][str(num_views)][config]
                expected_pairs = args.num_scenes * num_views * (num_views - 1) // 2
                if result["num_pairs"] != expected_pairs:
                    failures.append(
                        f"{dataset}/{num_views}/{config}: incomplete scene metrics"
                    )
                elif not all(
                    np.isfinite(result[key])
                    for key in ("mAA_30", "mean_global_align_loss")
                ):
                    failures.append(
                        f"{dataset}/{num_views}/{config}: nonfinite metrics"
                    )
    if failures:
        raise SystemExit("Multi-view run incomplete:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
