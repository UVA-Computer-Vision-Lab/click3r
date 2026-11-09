#!/usr/bin/env python3
"""Standard-benchmark loss and pose evaluation for Table 1.

CO3D, ScanNet++ and ARKitScenes share one frozen sampling protocol.
Use evaluation/reproduce_table1.py for the paper's configurations and ordering.
"""

import argparse
import copy
import json
import logging
import os
import os.path as osp
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ─── paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "dust3r"))
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "segment-anything"))
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "mast3r"))

# ─── imports ─────────────────────────────────────────────────────────────────
from click3r.model import Click3R  # noqa: F401  # CLI model expression
from click3r.inference import make_batch_symmetric
from click3r.metrics import compute_pose_metrics_from_dust3r, calculate_auc_np
from dust3r.losses import Regr3D_ScaleShiftInv, L21

from evaluation.baselines.loading import (
    load_dust3r_model,
    load_click3r_model,
    load_mast3r_model,
    cleanup_model,
    restore_curope_class_patch,
)

from math import inf  # noqa: F401  # CLI model expression
from PIL import Image
import torchvision.transforms as tvf

# ─── constants ───────────────────────────────────────────────────────────────
DATA_ROOT_DEFAULT = "data/processed"
MAX_P = 12  # Max prompt slots (matches training)

DATASET_ROOTS = {
    "co3d": "co3d_processed_for_dust3r",
    "scannetpp": "scannetpp_processed_for_dust3r",
    "arkitscenes": "arkitscenes_processed_for_dust3r",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════


def get_args_parser():
    p = argparse.ArgumentParser("Click3R standard benchmark evaluation")

    # Data paths
    p.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)

    # Model checkpoints
    p.add_argument(
        "--model",
        type=str,
        default="Click3R(pos_embed='RoPE100',"
        " patch_embed_cls='ManyAR_PatchEmbed',"
        " img_size=(224, 224), head_type='linear',"
        " output_mode='pts3d', depth_mode=('exp', -inf, inf),"
        " conf_mode=('exp', 1, inf),"
        " enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,"
        " dec_embed_dim=768, dec_depth=12, dec_num_heads=12)",
    )
    p.add_argument("--dust3r_pretrained", type=str, required=True)
    p.add_argument("--dust3r_checkpoint", type=str, required=True)
    p.add_argument("--click3r_checkpoint", type=str, required=True)
    p.add_argument(
        "--mast3r_model",
        type=str,
        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    )

    # Evaluation config
    p.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Evaluate a prefix after sampling the full subset",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Samples per dataset (default: 1000)",
    )
    p.add_argument(
        "--num_prompts",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Prompt counts to test (default: 1 8)",
    )
    p.add_argument(
        "--sampling_budgets",
        type=int,
        nargs="+",
        default=[1, 5, 50],
        help="Oracle budgets (default: 1 5 50)",
    )
    p.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["co3d", "scannetpp", "arkitscenes"],
        help="Datasets to evaluate",
    )
    p.add_argument("--skip_mast3r", action="store_true", help="Skip MASt3R baseline")
    p.add_argument(
        "--skip_dust3r_pretrained",
        action="store_true",
        help="Skip DUSt3R pretrained baseline",
    )
    p.add_argument(
        "--skip_dust3r_trained",
        action="store_true",
        help="Skip DUSt3R trained baseline",
    )
    p.add_argument(
        "--skip_click3r",
        action="store_true",
        help="Skip Click3R phases (no-prompts + oracle)",
    )
    p.add_argument(
        "--skip_vggt", action="store_true", help="Skip VGGT baseline (224px)"
    )
    p.add_argument(
        "--skip_pi3x", action="store_true", help="Skip Pi3X baseline (224px)"
    )
    p.add_argument(
        "--skip_vggt_highres",
        action="store_true",
        help="Skip VGGT high-resolution baseline",
    )
    p.add_argument(
        "--skip_pi3x_highres",
        action="store_true",
        help="Skip Pi3X high-resolution baseline",
    )
    p.add_argument(
        "--wrapper_batch_size",
        type=int,
        default=4,
        help="Batch size for VGGT/Pi3X at 224px (smaller due to model size)",
    )
    p.add_argument(
        "--highres_resolution",
        type=int,
        default=518,
        help="Input resolution for VGGT/Pi3X (must be 14-aligned)",
    )
    p.add_argument(
        "--highres_batch_size",
        type=int,
        default=4,
        help="Batch size for high-resolution baselines (default: 4)",
    )
    p.add_argument(
        "--skip_vggt_native",
        action="store_true",
        help="Skip VGGT native aspect-ratio eval",
    )
    p.add_argument(
        "--skip_pi3x_native",
        action="store_true",
        help="Skip Pi3X native aspect-ratio eval",
    )
    p.add_argument(
        "--skip_grid_oracle",
        action="store_true",
        help="Skip Click3R grid-oracle evaluation",
    )
    p.add_argument(
        "--grid_cell_size",
        type=int,
        default=14,
        help="Grid cell size in pixels for grid-oracle (default: 14)",
    )
    p.add_argument(
        "--native_longside",
        type=int,
        default=224,
        help="Longside target for native aspect ratio eval (default: 224)",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from a previous run directory. Skips any (dataset, model, config) "
        "that already has a result JSON file in that directory. "
        "Existing results are copied to the new output_dir.",
    )

    # Batching & parallelism
    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for no-prompt and budget=1 evals (default: 64)",
    )
    p.add_argument(
        "--num_workers", type=int, default=4, help="Data loading workers (default: 4)"
    )

    # Runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument(
        "--sampling-manifest-dir",
        type=str,
        help="Use frozen Table 1 dataset indices and verify source dataset lengths",
    )
    p.add_argument("--output_dir", type=str, required=True)

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset building
# ═══════════════════════════════════════════════════════════════════════════════


def build_datasets(args, resolution=224):
    """Build test datasets, each subsampled to args.num_samples."""
    from dust3r.datasets.co3d import Co3d
    from dust3r.datasets.scannetpp import ScanNetpp
    from dust3r.datasets.arkitscenes import ARKitScenes

    # One RNG is consumed across datasets in order, so the sampled indices depend
    # on the contents and order of args.datasets. Reproducing a run needs the
    # same list (saved as "datasets" in config.json).
    rng = np.random.default_rng(args.seed)
    datasets = {}

    builders = {
        "co3d": lambda root: Co3d(
            mask_bg=True,
            split="test",
            ROOT=root,
            resolution=resolution,
            aug_crop=False,
            seed=args.seed,
        ),
        "scannetpp": lambda root: ScanNetpp(
            split="train",
            ROOT=root,
            resolution=resolution,
            aug_crop=False,
            seed=args.seed,
        ),
        "arkitscenes": lambda root: ARKitScenes(
            split="test",
            ROOT=root,
            resolution=resolution,
            aug_crop=False,
            seed=args.seed,
        ),
    }

    for name in args.datasets:
        if name not in builders:
            print(f"WARNING: Unknown dataset '{name}', skipping")
            continue

        root = osp.join(args.data_root, DATASET_ROOTS[name])
        if not osp.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        print(f"  Loading {name} from {root}...")
        ds = builders[name](root)
        n = min(args.num_samples, len(ds))
        indices = rng.choice(len(ds), n, replace=False)
        indices.sort()  # deterministic order
        if getattr(args, "sampling_manifest_dir", None):
            manifest_path = osp.join(args.sampling_manifest_dir, f"{name}.json")
            with open(manifest_path) as handle:
                frozen = json.load(handle)
            if (
                frozen["dataset_length"] != len(ds)
                or frozen["seed"] != args.seed
                or frozen["canonical_order"] != args.datasets
                or frozen["sampled_indices"] != indices.tolist()
            ):
                raise ValueError(
                    f"Dataset metadata or sampling differs from {manifest_path}. "
                    "Restore the recorded dataset version before reproducing Table 1."
                )
        sampled_indices = indices.tolist()
        if getattr(args, "max_pairs", 0):
            sampled_indices = sampled_indices[: args.max_pairs]
        datasets[name] = Subset(ds, sampled_indices)
        same_frame = same_frame_positions(ds, sampled_indices, args.num_workers)
        datasets[name].same_frame_positions = same_frame
        if getattr(args, "sampling_manifest_dir", None) and same_frame != [
            p for p in frozen["same_frame_positions"] if p < len(sampled_indices)
        ]:
            raise ValueError(
                f"Same-frame pairs {same_frame} differ from {manifest_path}."
            )
        if getattr(args, "output_dir", None):
            os.makedirs(args.output_dir, exist_ok=True)
            with open(
                osp.join(args.output_dir, f"{name}_sampling_{resolution}.json"), "w"
            ) as handle:
                json.dump(
                    {
                        "dataset": name,
                        "dataset_length": len(ds),
                        "seed": args.seed,
                        "canonical_order": args.datasets,
                        "sampled_indices": indices.tolist(),
                        "evaluated_indices": sampled_indices,
                        "same_frame_positions": same_frame,
                        "resolution": resolution,
                    },
                    handle,
                    indent=2,
                )
        print(f"    {name}: {n} samples (from {len(ds)} total)")

    return datasets


class _PairIsSameFrame(Dataset):
    """Whether a pair's two views are the same frame, each pair loaded alone.

    CO3D's frame jitter can draw one frame twice. Its loader also replaces frames
    without valid depth and remembers them, so in a long pass a pair's frames can
    depend on earlier pairs; loading every pair from a fresh copy avoids that.
    """

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        first, second = copy.deepcopy(self.dataset)[self.indices[position]]
        return all(first[key] == second[key] for key in ("label", "instance"))


def same_frame_positions(dataset, indices, num_workers):
    """Positions in indices whose two views are the same frame.

    Such a pair has no translation direction, so summarize_pairs leaves it out of
    every aggregate.
    """
    loader = DataLoader(
        _PairIsSameFrame(dataset, indices),
        batch_size=16,
        num_workers=num_workers,
        collate_fn=list,
    )
    flags = [flag for batch in loader for flag in batch]
    return [position for position, flag in enumerate(flags) if flag]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch helpers
# ═══════════════════════════════════════════════════════════════════════════════


# Adapted from DUSt3R dust3r/inference.py loss_of_one_batch (https://github.com/naver/dust3r).
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified by the Click3R authors: nested sparse_prompts dicts are also moved.
def _move_to_device(batch, device):
    """Move the batch's tensors to device in place, including nested sparse_prompts."""
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            if isinstance(view[name], dict):
                for sub_key, sub_val in view[name].items():
                    if isinstance(sub_val, torch.Tensor):
                        view[name][sub_key] = sub_val.to(device, non_blocking=True)
            elif isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)


def deep_copy_batch(batch):
    """Deep copy a (view1, view2) batch so prompts can be attached independently."""
    v1, v2 = batch
    v1c, v2c = {}, {}
    for v, vc in [(v1, v1c), (v2, v2c)]:
        for k, val in v.items():
            if isinstance(val, torch.Tensor):
                vc[k] = val.clone()
            elif isinstance(val, dict):
                vc[k] = {
                    sk: sv.clone()
                    if isinstance(sv, torch.Tensor)
                    else copy.deepcopy(sv)
                    for sk, sv in val.items()
                }
            elif isinstance(val, np.ndarray):
                vc[k] = val.copy()
            else:
                vc[k] = val  # strings, ints, etc.
    return v1c, v2c


def stack_views(view_list):
    """Stack list of B=1 view dicts into a single batch."""
    out = {}
    for k in view_list[0].keys():
        vals = [v[k] for v in view_list]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.cat(vals, dim=0)
        elif isinstance(vals[0], dict):
            out[k] = {}
            for sk in vals[0].keys():
                svals = [v[k][sk] for v in view_list]
                if isinstance(svals[0], torch.Tensor):
                    out[k][sk] = torch.cat(svals, dim=0)
                else:
                    out[k][sk] = svals
        elif isinstance(vals[0], np.ndarray):
            out[k] = np.concatenate(vals, axis=0)
        else:
            out[k] = vals
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt attachment (GT correspondences)
# ═══════════════════════════════════════════════════════════════════════════════


def _to_hom(X):
    """(..., 3) -> (..., 4)"""
    return torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)


def _project_points(K, w2c, pts3d):
    """Project 3D world points to 2D pixel coords.
    K: (B,3,3), w2c: (B,4,4), pts3d: (B,M,3)
    Returns uv: (B,M,2), z: (B,M)
    """
    Xw = _to_hom(pts3d)
    Xc = (w2c @ Xw.transpose(1, 2)).transpose(1, 2)
    z = Xc[..., 2].clamp(min=1e-6)
    xn, yn = Xc[..., 0] / z, Xc[..., 1] / z
    uvh = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1)
    uv = (K @ uvh.transpose(1, 2)).transpose(1, 2)[..., :2]
    return uv, z


def _attach_prompts_single(v1, v2, b, num_prompts, seed, dev):
    """Compute GT correspondences for a single sample within a batch.

    Returns (c1_pad, c2_pad, vmask, shape1, shape2) — all single-sample tensors.
    """
    pts3d_v1 = v1.get("pts3d", None)
    pts3d_v2 = v2.get("pts3d", None)
    K2 = v2.get("camera_intrinsics", None)
    c2w2 = v2.get("camera_pose", None)

    empty = (
        torch.zeros(MAX_P, 2, device=dev),
        torch.zeros(MAX_P, 2, device=dev),
        torch.zeros(MAX_P, dtype=torch.bool, device=dev),
        torch.tensor([224, 224], dtype=torch.long, device=dev),
        torch.tensor([224, 224], dtype=torch.long, device=dev),
    )

    if any(x is None for x in [pts3d_v1, pts3d_v2, K2, c2w2]):
        return empty

    # Extract single-sample intrinsics and pose
    K2_b = K2[b] if K2.dim() == 3 else K2
    c2w2_b = c2w2[b] if c2w2.dim() == 3 else c2w2
    w2c2_b = torch.inverse(c2w2_b)

    # Get pointmaps for this sample
    if pts3d_v1.dim() == 4:
        p3d_v1 = pts3d_v1[b]  # (H, W, 3)
        p3d_v2 = pts3d_v2[b]
    else:
        ts1 = v1["true_shape"]
        H1 = int(ts1[b, 0]) if isinstance(ts1, torch.Tensor) else int(ts1[b][0])
        W1 = int(ts1[b, 1]) if isinstance(ts1, torch.Tensor) else int(ts1[b][1])
        p3d_v1 = pts3d_v1[b].reshape(H1, W1, 3)
        ts2 = v2["true_shape"]
        H2 = int(ts2[b, 0]) if isinstance(ts2, torch.Tensor) else int(ts2[b][0])
        W2 = int(ts2[b, 1]) if isinstance(ts2, torch.Tensor) else int(ts2[b][1])
        p3d_v2 = pts3d_v2[b].reshape(H2, W2, 3)

    H1, W1, _ = p3d_v1.shape
    H2, W2, _ = p3d_v2.shape

    # Valid mask for this sample
    valid1 = v1.get("valid_mask", None)
    if valid1 is None:
        valid1 = torch.ones(H1, W1, dtype=torch.bool, device=dev)
    else:
        valid1 = valid1[b] if valid1.dim() >= 3 else valid1
        if isinstance(valid1, np.ndarray):
            valid1 = torch.from_numpy(valid1).to(dev)
        while valid1.dim() > 2:
            valid1 = valid1[0]

    # Flatten and project
    p3d_flat = p3d_v1.reshape(-1, 3)
    valid_flat = valid1.reshape(-1)

    uv2, z2 = _project_points(
        K2_b.unsqueeze(0), w2c2_b.unsqueeze(0), p3d_flat.unsqueeze(0)
    )
    uv2, z2 = uv2[0], z2[0]

    in_bounds = (
        (z2 > 0)
        & (uv2[:, 0] >= 0)
        & (uv2[:, 0] < W2 - 1)
        & (uv2[:, 1] >= 0)
        & (uv2[:, 1] < H2 - 1)
    )
    combined = valid_flat & in_bounds
    valid_idx = torch.where(combined)[0]
    if len(valid_idx) == 0:
        return (
            torch.zeros(MAX_P, 2, device=dev),
            torch.zeros(MAX_P, 2, device=dev),
            torch.zeros(MAX_P, dtype=torch.bool, device=dev),
            torch.tensor([H1, W1], dtype=torch.long, device=dev),
            torch.tensor([H2, W2], dtype=torch.long, device=dev),
        )

    # 3D distance check
    y1 = valid_idx // W1
    x1 = valid_idx % W1
    x2 = torch.round(uv2[valid_idx, 0]).long()
    y2 = torch.round(uv2[valid_idx, 1]).long()

    pts_v1 = p3d_v1[y1, x1]
    pts_v2 = p3d_v2[y2, x2]
    dist_3d = torch.norm(pts_v1 - pts_v2, dim=1)
    good = dist_3d <= 0.05  # 5cm threshold

    x1, y1 = x1[good], y1[good]
    x2, y2 = x2[good], y2[good]
    num_matches = len(x1)
    if num_matches == 0:
        return (
            torch.zeros(MAX_P, 2, device=dev),
            torch.zeros(MAX_P, 2, device=dev),
            torch.zeros(MAX_P, dtype=torch.bool, device=dev),
            torch.tensor([H1, W1], dtype=torch.long, device=dev),
            torch.tensor([H2, W2], dtype=torch.long, device=dev),
        )

    # Sample with deterministic RNG
    rng = np.random.RandomState(seed)
    actual = min(num_prompts, num_matches)
    sel = rng.choice(num_matches, actual, replace=False)
    sel = torch.from_numpy(sel).to(dev)

    c1 = torch.stack([x1[sel].float(), y1[sel].float()], dim=1)
    c2 = torch.stack([x2[sel].float(), y2[sel].float()], dim=1)

    c1_pad = torch.zeros(MAX_P, 2, device=dev)
    c2_pad = torch.zeros(MAX_P, 2, device=dev)
    vmask = torch.zeros(MAX_P, dtype=torch.bool, device=dev)

    c1_pad[:actual] = c1
    c2_pad[:actual] = c2
    vmask[:actual] = True

    if actual < MAX_P and actual > 0:
        rep_idx = torch.from_numpy(rng.randint(0, actual, MAX_P - actual)).to(dev)
        c1_pad[actual:] = c1[rep_idx]
        c2_pad[actual:] = c2[rep_idx]

    return (
        c1_pad,
        c2_pad,
        vmask,
        torch.tensor([H1, W1], dtype=torch.long, device=dev),
        torch.tensor([H2, W2], dtype=torch.long, device=dev),
    )


def attach_random_prompts(batch, num_prompts, seed):
    """
    Attach num_prompts GT correspondences sampled uniformly at random.
    Supports B>=1 by looping over batch items.

    Uses overlap-aware matching: project view1 pts3d to view2, check 3D distance.
    Deterministic via explicit seed (per-sample: seed + batch_index).
    """
    v1, v2 = batch
    dev = v1["img"].device
    B = v1["img"].shape[0]

    # Pre-allocate batch output tensors
    all_c1 = torch.zeros(B, MAX_P, 2, device=dev)
    all_c2 = torch.zeros(B, MAX_P, 2, device=dev)
    all_vmask = torch.zeros(B, MAX_P, dtype=torch.bool, device=dev)
    shape1_list, shape2_list = [], []

    for b in range(B):
        c1, c2, vmask, s1, s2 = _attach_prompts_single(
            v1, v2, b, num_prompts, seed + b, dev
        )
        all_c1[b] = c1
        all_c2[b] = c2
        all_vmask[b] = vmask
        shape1_list.append(s1)
        shape2_list.append(s2)

    pair_ids = (
        torch.arange(MAX_P, device=dev, dtype=torch.long).unsqueeze(0).expand(B, -1)
    )

    v1["sparse_prompts"] = {
        "coords": all_c1,
        "image_ids": torch.zeros(B, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": torch.stack(shape1_list),
        "valid_mask": all_vmask,
    }
    v2["sparse_prompts"] = {
        "coords": all_c2,
        "image_ids": torch.ones(B, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": torch.stack(shape2_list),
        "valid_mask": all_vmask,
    }


def _set_empty_prompts(v1, v2, dev, B=1):
    """Set empty (all-zero, all-invalid) prompts on views."""
    for v in (v1, v2):
        v["sparse_prompts"] = {
            "coords": torch.zeros(B, MAX_P, 2, device=dev),
            "image_ids": torch.zeros(B, MAX_P, device=dev, dtype=torch.long),
            "pair_ids": torch.arange(MAX_P, device=dev, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1),
            "shape": torch.tensor([[224, 224]], device=dev).expand(B, -1),
            "valid_mask": torch.zeros(B, MAX_P, dtype=torch.bool, device=dev),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Grid-oracle helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_valid_correspondences(v1, v2, b, dev):
    """Get ALL valid GT correspondences for sample b.

    Returns:
        c1: (N, 2) pixel coords in view1 (x, y)
        c2: (N, 2) pixel coords in view2 (x, y)
        shapes: (H1, W1, H2, W2)
        or (None, None, None) if no valid correspondences
    """
    pts3d_v1 = v1.get("pts3d", None)
    pts3d_v2 = v2.get("pts3d", None)
    K2 = v2.get("camera_intrinsics", None)
    c2w2 = v2.get("camera_pose", None)

    if any(x is None for x in [pts3d_v1, pts3d_v2, K2, c2w2]):
        return None, None, None

    K2_b = K2[b] if K2.dim() == 3 else K2
    c2w2_b = c2w2[b] if c2w2.dim() == 3 else c2w2
    w2c2_b = torch.inverse(c2w2_b)

    if pts3d_v1.dim() == 4:
        p3d_v1 = pts3d_v1[b]
        p3d_v2 = pts3d_v2[b]
    else:
        ts1 = v1["true_shape"]
        H1 = int(ts1[b, 0]) if isinstance(ts1, torch.Tensor) else int(ts1[b][0])
        W1 = int(ts1[b, 1]) if isinstance(ts1, torch.Tensor) else int(ts1[b][1])
        p3d_v1 = pts3d_v1[b].reshape(H1, W1, 3)
        ts2 = v2["true_shape"]
        H2 = int(ts2[b, 0]) if isinstance(ts2, torch.Tensor) else int(ts2[b][0])
        W2 = int(ts2[b, 1]) if isinstance(ts2, torch.Tensor) else int(ts2[b][1])
        p3d_v2 = pts3d_v2[b].reshape(H2, W2, 3)

    H1, W1, _ = p3d_v1.shape
    H2, W2, _ = p3d_v2.shape

    valid1 = v1.get("valid_mask", None)
    if valid1 is None:
        valid1 = torch.ones(H1, W1, dtype=torch.bool, device=dev)
    else:
        valid1 = valid1[b] if valid1.dim() >= 3 else valid1
        if isinstance(valid1, np.ndarray):
            valid1 = torch.from_numpy(valid1).to(dev)
        while valid1.dim() > 2:
            valid1 = valid1[0]

    p3d_flat = p3d_v1.reshape(-1, 3)
    valid_flat = valid1.reshape(-1)

    uv2, z2 = _project_points(
        K2_b.unsqueeze(0), w2c2_b.unsqueeze(0), p3d_flat.unsqueeze(0)
    )
    uv2, z2 = uv2[0], z2[0]

    in_bounds = (
        (z2 > 0)
        & (uv2[:, 0] >= 0)
        & (uv2[:, 0] < W2 - 1)
        & (uv2[:, 1] >= 0)
        & (uv2[:, 1] < H2 - 1)
    )
    combined = valid_flat & in_bounds
    valid_idx = torch.where(combined)[0]

    if len(valid_idx) == 0:
        return None, None, None

    y1 = valid_idx // W1
    x1 = valid_idx % W1
    x2 = torch.round(uv2[valid_idx, 0]).long()
    y2 = torch.round(uv2[valid_idx, 1]).long()

    pts_v1 = p3d_v1[y1, x1]
    pts_v2 = p3d_v2[y2, x2]
    dist_3d = torch.norm(pts_v1 - pts_v2, dim=1)
    good = dist_3d <= 0.05

    x1, y1 = x1[good], y1[good]
    x2, y2 = x2[good], y2[good]

    if len(x1) == 0:
        return None, None, None

    c1 = torch.stack([x1.float(), y1.float()], dim=1)
    c2 = torch.stack([x2.float(), y2.float()], dim=1)
    return c1, c2, (H1, W1, H2, W2)


def assign_to_pixel_grid(c1, H, W, cell_size):
    """Assign view1 correspondences to square grid cells of cell_size pixels.

    Args:
        c1: (N, 2) pixel coords (x, y) in view1
        H, W: image height/width
        cell_size: pixel size of each cell

    Returns:
        cell_to_indices: dict mapping cell_id -> list of correspondence indices
        grid_rows, grid_cols: number of cells per dimension
    """
    grid_rows = H // cell_size
    grid_cols = W // cell_size

    rows = (c1[:, 1] / cell_size).long().clamp(0, grid_rows - 1)
    cols = (c1[:, 0] / cell_size).long().clamp(0, grid_cols - 1)
    cell_ids = rows * grid_cols + cols

    cell_to_indices = {}
    for i, cid in enumerate(cell_ids.tolist()):
        cell_to_indices.setdefault(cid, []).append(i)
    return cell_to_indices, grid_rows, grid_cols


def _format_prompts_from_corrs(c1_sel, c2_sel, shapes, dev, seed):
    """Format selected correspondences into sparse_prompts dicts (B=1).

    Args:
        c1_sel: (K, 2) selected view1 coords
        c2_sel: (K, 2) selected view2 coords
        shapes: (H1, W1, H2, W2)
        dev: torch device
        seed: RNG seed for padding replication
    """
    H1, W1, H2, W2 = shapes
    actual = len(c1_sel)

    c1_pad = torch.zeros(MAX_P, 2, device=dev)
    c2_pad = torch.zeros(MAX_P, 2, device=dev)
    vmask = torch.zeros(MAX_P, dtype=torch.bool, device=dev)

    c1_pad[:actual] = c1_sel
    c2_pad[:actual] = c2_sel
    vmask[:actual] = True

    if actual < MAX_P and actual > 0:
        rng = np.random.RandomState(seed)
        rep_idx = torch.from_numpy(rng.randint(0, actual, MAX_P - actual)).to(dev)
        c1_pad[actual:] = c1_sel[rep_idx]
        c2_pad[actual:] = c2_sel[rep_idx]

    pair_ids = torch.arange(MAX_P, device=dev, dtype=torch.long).unsqueeze(0)

    prompts_v1 = {
        "coords": c1_pad.unsqueeze(0),
        "image_ids": torch.zeros(1, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": torch.tensor([[H1, W1]], device=dev, dtype=torch.long),
        "valid_mask": vmask.unsqueeze(0),
    }
    prompts_v2 = {
        "coords": c2_pad.unsqueeze(0),
        "image_ids": torch.ones(1, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": torch.tensor([[H2, W2]], device=dev, dtype=torch.long),
        "valid_mask": vmask.unsqueeze(0),
    }
    return prompts_v1, prompts_v2


# ═══════════════════════════════════════════════════════════════════════════════
# Loss computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_per_sample_loss(criterion, pred1, pred2, view1, view2, idx):
    """Extract single sample from symmetrized batch and compute loss."""

    def _slice(d, i):
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] > i:
                out[k] = v[i : i + 1]
            else:
                out[k] = v
        return out

    gt1 = _slice(view1, idx)
    gt2 = _slice(view2, idx)
    p1 = _slice(pred1, idx)
    p2 = _slice(pred2, idx)

    with torch.no_grad():
        loss_val, _ = criterion(gt1, gt2, p1, p2)
    return loss_val.item()


# ═══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════════


def compute_pose_metrics_summary(r_errors, t_errors, thresholds=(5, 15, 30)):
    """Compute summary metrics from error lists."""
    r = np.array(r_errors)
    t = np.array(t_errors)
    metrics = {}

    for th in thresholds:
        metrics[f"RRA_{th}"] = float(np.mean(r < th) * 100)
        metrics[f"RTA_{th}"] = float(np.mean(t < th) * 100)
        auc_val = calculate_auc_np(r, t, max_threshold=th)
        if isinstance(auc_val, tuple):
            auc_val = auc_val[0]
        metrics[f"mAA_{th}"] = float(auc_val * 100)

    metrics["R_mean"] = float(np.mean(r))
    metrics["R_median"] = float(np.median(r))
    metrics["T_mean"] = float(np.mean(t))
    metrics["T_median"] = float(np.median(t))
    return metrics


def summarize_pairs(dataset, r_errors, t_errors, losses=(), finite_losses=False):
    """Aggregate per-pair results without the dataset's same-frame pairs.

    Per-pair lists keep every evaluated pair. finite_losses drops infinite losses
    (failed oracle attempts) from the loss aggregates.
    """
    excluded = getattr(dataset, "same_frame_positions", [])
    if excluded:
        assert len(r_errors) == len(dataset), "a pair has no result"
        assert not losses or len(losses) == len(dataset), "a pair has no loss"
    skip = set(excluded)
    keep = [i for i in range(len(r_errors)) if i not in skip]
    metrics = compute_pose_metrics_summary(
        [r_errors[i] for i in keep], [t_errors[i] for i in keep]
    )
    if losses:
        kept = [losses[i] for i in keep] if excluded else list(losses)
        if finite_losses:
            kept = [loss for loss in kept if loss != float("inf")]
            metrics["mean_loss"] = float(np.nanmean(kept))
            metrics["median_loss"] = float(np.nanmedian(kept))
        else:
            metrics["mean_loss"] = float(np.mean(kept))
            metrics["median_loss"] = float(np.median(kept))
    metrics["num_samples"] = len(r_errors)
    metrics["excluded_positions"] = list(excluded)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Core evaluation functions
# ═══════════════════════════════════════════════════════════════════════════════


def _make_eval_loader(dataset, batch_size=1, num_workers=0):
    """Create an evaluation DataLoader with default collation, as DUSt3R's
    get_data_loader (dust3r/datasets/__init__.py) does: numpy arrays become
    tensors with a batch dimension.

    Args:
        dataset: Dataset or Subset
        batch_size: Samples per batch (default: 1)
        num_workers: Data loading workers (default: 0)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def evaluate_no_prompts(
    model, criterion, dataset, device, dataset_name, batch_size=16, num_workers=8
):
    """Evaluate a model without prompts (baseline). Returns loss + pose metrics.

    Supports batched processing (B>1) for throughput. Per-sample loss and pose
    are extracted from the symmetrized batch output.
    """
    all_losses, r_errors, t_errors = [], [], []
    loader = _make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers)

    _first_err = True
    for batch in tqdm(loader, desc=f"  [{dataset_name}] no-prompts"):
        try:
            for view in batch:
                view.pop("sparse_prompts", None)

            _move_to_device(batch, device)
            v1_sym, v2_sym = make_batch_symmetric(batch)

            with torch.no_grad():
                pred1, pred2 = model(v1_sym, v2_sym)

            B_orig = v1_sym["img"].shape[0] // 2

            for i in range(B_orig):
                sym_idx = 2 * i
                try:
                    sample_loss = compute_per_sample_loss(
                        criterion, pred1, pred2, v1_sym, v2_sym, idx=sym_idx
                    )
                    all_losses.append(sample_loss)
                except Exception:
                    if _first_err:
                        import traceback

                        traceback.print_exc()
                        _first_err = False
                try:
                    r, t = compute_pose_metrics_from_dust3r(
                        v1_sym, v2_sym, pred1, pred2, pair_idx=sym_idx
                    )
                except Exception:
                    if _first_err:
                        import traceback

                        traceback.print_exc()
                        _first_err = False
                    # A failed PnP counts as a 180° error so every row keeps all pairs.
                    r, t = 180.0, 180.0
                r_errors.append(r)
                t_errors.append(t)
        except Exception:
            if _first_err:
                import traceback

                traceback.print_exc()
                _first_err = False
            continue

    metrics = summarize_pairs(dataset, r_errors, t_errors, all_losses)
    return metrics, all_losses, r_errors, t_errors


def _extract_single_from_batch(view, b):
    """Extract sample b from a batched view dict as a B=1 dict (with clones)."""
    out = {}
    for k, v in view.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] > b:
            out[k] = v[b : b + 1].clone()
        elif isinstance(v, dict):
            out[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor) and sv.dim() >= 1 and sv.shape[0] > b:
                    out[k][sk] = sv[b : b + 1].clone()
                else:
                    out[k][sk] = copy.deepcopy(sv)
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > b:
            out[k] = v[b : b + 1].copy()
        elif isinstance(v, (list, tuple)) and len(v) > b:
            out[k] = [v[b]]
        else:
            out[k] = v
    return out


def evaluate_oracle(
    model,
    criterion,
    dataset,
    device,
    dataset_name,
    num_prompts,
    budget,
    seed=777,
    batch_size=16,
    num_workers=8,
):
    """
    Oracle evaluation: for each pair, try 'budget' random prompt sets and keep the
    one with the lowest loss.
    """
    all_losses, r_errors, t_errors = [], [], []
    # Size the DataLoader so model batch ≈ batch_size after symmetrization
    effective_bs = max(1, batch_size // (2 * budget))
    loader = _make_eval_loader(
        dataset, batch_size=effective_bs, num_workers=num_workers
    )

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"  [{dataset_name}] {num_prompts}p×{budget}")
    ):
        try:
            B = batch[0]["img"].shape[0]
            _move_to_device(batch, device)

            if budget == 1:
                # Batched single-attempt: attach prompts per-sample, forward all
                pair_seed = seed + batch_idx * effective_bs
                attach_random_prompts(batch, num_prompts, pair_seed)

                v1_sym, v2_sym = make_batch_symmetric(batch)
                with torch.no_grad():
                    pred1, pred2 = model(v1_sym, v2_sym)

                for i in range(B):
                    sym_idx = 2 * i
                    try:
                        sample_loss = compute_per_sample_loss(
                            criterion, pred1, pred2, v1_sym, v2_sym, idx=sym_idx
                        )
                        all_losses.append(sample_loss)
                    except Exception:
                        all_losses.append(float("inf"))
                    try:
                        r, t = compute_pose_metrics_from_dust3r(
                            v1_sym, v2_sym, pred1, pred2, pair_idx=sym_idx
                        )
                    except Exception:
                        # A failed PnP counts as a 180° error, as in best-of-N.
                        r, t = 180.0, 180.0
                    r_errors.append(r)
                    t_errors.append(t)

            else:
                # Oracle: batch B pairs × budget attempts together
                # Create B*budget copies with independent prompts
                attempt_v1s, attempt_v2s = [], []
                for b in range(B):
                    v1_b = _extract_single_from_batch(batch[0], b)
                    v2_b = _extract_single_from_batch(batch[1], b)
                    pair_seed = seed + batch_idx * effective_bs + b
                    for a in range(budget):
                        v1c, v2c = deep_copy_batch((v1_b, v2_b))
                        attach_random_prompts(
                            (v1c, v2c), num_prompts, pair_seed * 1000 + a
                        )
                        attempt_v1s.append(v1c)
                        attempt_v2s.append(v2c)

                stacked_v1 = stack_views(attempt_v1s)
                stacked_v2 = stack_views(attempt_v2s)

                v1_sym, v2_sym = make_batch_symmetric((stacked_v1, stacked_v2))

                with torch.no_grad():
                    pred1, pred2 = model(v1_sym, v2_sym)

                # Extract best-of-budget per pair
                # Layout: [pair0_att0, pair0_att1, ..., pair1_att0, pair1_att1, ...]
                for b in range(B):
                    attempt_losses = []
                    attempt_poses = []
                    for a in range(budget):
                        idx = b * budget + a
                        sym_idx = 2 * idx
                        try:
                            sample_loss = compute_per_sample_loss(
                                criterion, pred1, pred2, v1_sym, v2_sym, idx=sym_idx
                            )
                        except Exception:
                            sample_loss = float("inf")
                        attempt_losses.append(sample_loss)
                        try:
                            r, t = compute_pose_metrics_from_dust3r(
                                v1_sym, v2_sym, pred1, pred2, pair_idx=sym_idx
                            )
                            attempt_poses.append((r, t))
                        except Exception:
                            attempt_poses.append((180.0, 180.0))

                    best = int(np.argmin(attempt_losses))
                    all_losses.append(attempt_losses[best])
                    r_errors.append(attempt_poses[best][0])
                    t_errors.append(attempt_poses[best][1])

        except Exception:
            logging.getLogger(__name__).exception(
                "<ast.FunctionDef object at 0x14a57ea31b10>: failed batch for %s",
                dataset_name,
            )
            continue

    metrics = summarize_pairs(
        dataset, r_errors, t_errors, all_losses, finite_losses=True
    )
    metrics["num_prompts"] = num_prompts
    metrics["budget"] = budget
    return metrics, all_losses, r_errors, t_errors


def evaluate_grid_oracle(
    model,
    criterion,
    dataset,
    device,
    dataset_name,
    cell_size=14,
    batch_size=512,
    seed=777,
    num_workers=4,
):
    """Grid-oracle evaluation: divide view1 into cells, test 1 prompt per populated
    cell, rank by pose improvement, final eval with best cell's prompt.

    B=1 outer loop (each sample needs unique GT correspondences for grid search).
    Batched inner loop for the grid search across cells.

    Returns (metrics, losses, r_errors, t_errors).
    """
    from torch.utils.data import DataLoader

    all_losses, r_errors, t_errors = [], [], []
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    n_skipped = 0
    n_fallback = 0  # samples with no GT correspondences → use no-prompt result
    n_cells_tested = []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"  [{dataset_name}] grid_oracle_1p_pose")
    ):
        try:
            _move_to_device(batch, device)
            v1_orig, v2_orig = batch

            # ── Step 1: No-prompt baseline → get baseline pose error ──
            v1_np = {
                k: (
                    v.clone()
                    if isinstance(v, torch.Tensor)
                    else (
                        {
                            sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()
                        }
                        if isinstance(v, dict)
                        else (v.copy() if isinstance(v, np.ndarray) else v)
                    )
                )
                for k, v in v1_orig.items()
            }
            v2_np = {
                k: (
                    v.clone()
                    if isinstance(v, torch.Tensor)
                    else (
                        {
                            sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()
                        }
                        if isinstance(v, dict)
                        else (v.copy() if isinstance(v, np.ndarray) else v)
                    )
                )
                for k, v in v2_orig.items()
            }
            v1_np.pop("sparse_prompts", None)
            v2_np.pop("sparse_prompts", None)
            v1_sym, v2_sym = make_batch_symmetric((v1_np, v2_np))

            with torch.no_grad():
                pred1, pred2 = model(v1_sym, v2_sym)

            try:
                r_np, t_np = compute_pose_metrics_from_dust3r(
                    v1_sym, v2_sym, pred1, pred2, pair_idx=0
                )
            except Exception:
                # Keep the pair; a failed baseline ranks cells like a 360° error.
                r_np, t_np = 360.0, 360.0

            baseline_pose_err = max(r_np, t_np)

            # ── Step 2: Get all GT correspondences ──
            c1_all, c2_all, shapes = get_all_valid_correspondences(
                v1_orig, v2_orig, 0, device
            )

            if c1_all is None or len(c1_all) == 0:
                # No GT correspondences → fall back to no-prompt result
                n_fallback += 1
                try:
                    sample_loss = compute_per_sample_loss(
                        criterion, pred1, pred2, v1_sym, v2_sym, idx=0
                    )
                    all_losses.append(sample_loss)
                except Exception:
                    pass
                r_errors.append(r_np)
                t_errors.append(t_np)
                continue

            # ── Step 3: Assign to grid, pick 1 corr per cell ──
            H1, W1 = shapes[0], shapes[1]
            cell_to_indices, grid_rows, grid_cols = assign_to_pixel_grid(
                c1_all, H1, W1, cell_size
            )

            cell_list = sorted(cell_to_indices.keys())
            if len(cell_list) == 0:
                # No populated cells → fall back to no-prompt result
                n_fallback += 1
                try:
                    sample_loss = compute_per_sample_loss(
                        criterion, pred1, pred2, v1_sym, v2_sym, idx=0
                    )
                    all_losses.append(sample_loss)
                except Exception:
                    pass
                r_errors.append(r_np)
                t_errors.append(t_np)
                continue

            rng = np.random.RandomState(seed + batch_idx)
            cell_chosen = []  # (cell_idx, corr_idx)
            for cell_idx in cell_list:
                corr_indices = cell_to_indices[cell_idx]
                chosen = rng.choice(corr_indices)
                cell_chosen.append((cell_idx, chosen))

            n_cells_tested.append(len(cell_chosen))

            # ── Step 4: Batched grid search ──
            effective_bs = max(1, batch_size // 2)
            cell_results = []  # (cell_idx, pose_err, improvement, corr_idx)

            for chunk_start in range(0, len(cell_chosen), effective_bs):
                chunk = cell_chosen[chunk_start : chunk_start + effective_bs]

                v1_list, v2_list = [], []
                for cell_idx, corr_idx in chunk:
                    v1_c = {
                        k: (
                            v.clone()
                            if isinstance(v, torch.Tensor)
                            else (
                                {
                                    sk: sv.clone()
                                    if isinstance(sv, torch.Tensor)
                                    else sv
                                    for sk, sv in v.items()
                                }
                                if isinstance(v, dict)
                                else (v.copy() if isinstance(v, np.ndarray) else v)
                            )
                        )
                        for k, v in v1_orig.items()
                    }
                    v2_c = {
                        k: (
                            v.clone()
                            if isinstance(v, torch.Tensor)
                            else (
                                {
                                    sk: sv.clone()
                                    if isinstance(sv, torch.Tensor)
                                    else sv
                                    for sk, sv in v.items()
                                }
                                if isinstance(v, dict)
                                else (v.copy() if isinstance(v, np.ndarray) else v)
                            )
                        )
                        for k, v in v2_orig.items()
                    }
                    pv1, pv2 = _format_prompts_from_corrs(
                        c1_all[corr_idx : corr_idx + 1],
                        c2_all[corr_idx : corr_idx + 1],
                        shapes,
                        device,
                        seed + cell_idx,
                    )
                    v1_c["sparse_prompts"] = pv1
                    v2_c["sparse_prompts"] = pv2
                    v1_list.append(v1_c)
                    v2_list.append(v2_c)

                stacked_v1 = stack_views(v1_list)
                stacked_v2 = stack_views(v2_list)
                v1_s, v2_s = make_batch_symmetric((stacked_v1, stacked_v2))

                with torch.no_grad():
                    p1, p2 = model(v1_s, v2_s)

                for i, (cell_idx, corr_idx) in enumerate(chunk):
                    sym_idx = 2 * i
                    try:
                        r, t = compute_pose_metrics_from_dust3r(
                            v1_s, v2_s, p1, p2, pair_idx=sym_idx
                        )
                        pose_err = max(r, t)
                    except Exception:
                        pose_err = 360.0
                    improvement = baseline_pose_err - pose_err
                    cell_results.append((cell_idx, pose_err, improvement, corr_idx))

            if len(cell_results) == 0:
                n_skipped += 1
                continue

            # ── Step 5: Best cell → final eval ──
            cell_results.sort(key=lambda x: x[2], reverse=True)
            best_corr_idx = cell_results[0][3]

            # Final forward with best cell's prompt
            v1_f = {
                k: (
                    v.clone()
                    if isinstance(v, torch.Tensor)
                    else (
                        {
                            sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()
                        }
                        if isinstance(v, dict)
                        else (v.copy() if isinstance(v, np.ndarray) else v)
                    )
                )
                for k, v in v1_orig.items()
            }
            v2_f = {
                k: (
                    v.clone()
                    if isinstance(v, torch.Tensor)
                    else (
                        {
                            sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                            for sk, sv in v.items()
                        }
                        if isinstance(v, dict)
                        else (v.copy() if isinstance(v, np.ndarray) else v)
                    )
                )
                for k, v in v2_orig.items()
            }
            pv1, pv2 = _format_prompts_from_corrs(
                c1_all[best_corr_idx : best_corr_idx + 1],
                c2_all[best_corr_idx : best_corr_idx + 1],
                shapes,
                device,
                seed + batch_idx + 99999,
            )
            v1_f["sparse_prompts"] = pv1
            v2_f["sparse_prompts"] = pv2
            v1_s, v2_s = make_batch_symmetric((v1_f, v2_f))

            with torch.no_grad():
                p1, p2 = model(v1_s, v2_s)

            try:
                sample_loss = compute_per_sample_loss(
                    criterion, p1, p2, v1_s, v2_s, idx=0
                )
                all_losses.append(sample_loss)
            except Exception:
                pass

            try:
                r, t = compute_pose_metrics_from_dust3r(v1_s, v2_s, p1, p2, pair_idx=0)
            except Exception:
                # A failed PnP counts as a 180° error so every row keeps all pairs.
                r, t = 180.0, 180.0
            r_errors.append(r)
            t_errors.append(t)

        except Exception:
            logging.getLogger(__name__).exception(
                "<ast.FunctionDef object at 0x14a57ea57490>: failed batch for %s",
                dataset_name,
            )
            n_skipped += 1
            continue

    print(
        f"    Grid stats: {len(n_cells_tested)} grid-searched, "
        f"{n_fallback} no-corr fallback (used no-prompt result), "
        f"{n_skipped} skipped (pose fail)"
    )
    if n_cells_tested:
        print(
            f"    Cells/sample: mean={np.mean(n_cells_tested):.1f}, "
            f"min={np.min(n_cells_tested)}, max={np.max(n_cells_tested)}"
        )

    metrics = summarize_pairs(dataset, r_errors, t_errors, all_losses)
    metrics["cell_size"] = cell_size
    return metrics, all_losses, r_errors, t_errors


# ═══════════════════════════════════════════════════════════════════════════════
# MASt3R baseline evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_mast3r_no_prompts(
    mast3r_model, criterion, dataset, device, dataset_name, batch_size=16, num_workers=8
):
    """
    Evaluate MASt3R as a baseline (no prompts).

    MASt3R extends DUSt3R architecture and uses the same pipeline.
    Evaluated at 224px for fair comparison with DUSt3R and Click3R.
    Supports batched processing (B>1) for throughput.
    """
    all_losses, r_errors, t_errors = [], [], []
    loader = _make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers)

    for batch in tqdm(loader, desc=f"  [{dataset_name}] MASt3R"):
        try:
            _move_to_device(batch, device)
            v1_sym, v2_sym = make_batch_symmetric(batch)

            with torch.no_grad():
                pred1, pred2 = mast3r_model(v1_sym, v2_sym)

            B_orig = v1_sym["img"].shape[0] // 2

            for i in range(B_orig):
                sym_idx = 2 * i
                try:
                    sample_loss = compute_per_sample_loss(
                        criterion, pred1, pred2, v1_sym, v2_sym, idx=sym_idx
                    )
                    all_losses.append(sample_loss)
                except Exception:
                    pass
                try:
                    r, t = compute_pose_metrics_from_dust3r(
                        v1_sym, v2_sym, pred1, pred2, pair_idx=sym_idx
                    )
                except Exception:
                    # A failed PnP counts as a 180° error so every row keeps all pairs.
                    r, t = 180.0, 180.0
                r_errors.append(r)
                t_errors.append(t)
        except Exception:
            logging.getLogger(__name__).exception(
                "<ast.FunctionDef object at 0x14a57e8c37f0>: failed batch for %s",
                dataset_name,
            )
            continue

    metrics = summarize_pairs(dataset, r_errors, t_errors, all_losses)
    return metrics, all_losses, r_errors, t_errors


# ═══════════════════════════════════════════════════════════════════════════════
# Native aspect-ratio evaluation (loads from img_path on disk)
# ═══════════════════════════════════════════════════════════════════════════════


def resize_longside_14aligned(pil_img, target_longside):
    """Resize so longside ~ target, both dims 14-aligned for DINOv2. Returns [0,1] tensor."""
    W, H = pil_img.size
    if H >= W:  # portrait
        new_H = round(target_longside / 14) * 14
        new_W = round(W * (new_H / H) / 14) * 14
    else:  # landscape
        new_W = round(target_longside / 14) * 14
        new_H = round(H * (new_W / W) / 14) * 14
    new_W = max(new_W, 14)
    new_H = max(new_H, 14)
    resized = pil_img.resize((new_W, new_H), Image.Resampling.LANCZOS)
    return tvf.ToTensor()(resized)  # [0,1], shape (3, new_H, new_W)


def evaluate_native_resolution(
    model, dataset, device, dataset_name, target_longside=224
):
    """Eval with aspect-ratio-preserving resize. Loads from img_path on disk.

    B=1 per pair since different pairs may have different aspect ratios.
    Only computes pose metrics (no loss -- GT depthmap doesn't match native resolution).
    """
    r_errors, t_errors = [], []
    _first_err = True

    for i in tqdm(
        range(len(dataset)), desc=f"  [{dataset_name}] native-{target_longside}px"
    ):
        try:
            item = dataset[i]
            v1_dict, v2_dict = item

            # Check img_path availability
            img_path1 = v1_dict.get("img_path", None)
            img_path2 = v2_dict.get("img_path", None)
            if img_path1 is None or img_path2 is None:
                continue
            if isinstance(img_path1, (list, tuple)):
                img_path1 = img_path1[0]
            if isinstance(img_path2, (list, tuple)):
                img_path2 = img_path2[0]

            # Load raw preprocessed images from disk (original aspect ratio)
            img1_pil = Image.open(img_path1).convert("RGB")
            img2_pil = Image.open(img_path2).convert("RGB")

            # Resize longside to target, 14-aligned
            img1_t = resize_longside_14aligned(img1_pil, target_longside)
            img2_t = resize_longside_14aligned(img2_pil, target_longside)

            # Normalize to [-1,1] (DUSt3R format -- wrappers denormalize internally)
            img1_t = img1_t * 2 - 1
            img2_t = img2_t * 2 - 1

            # Build view dicts with GT camera params (convert numpy → tensor, add batch dim)
            def _ensure_batched_tensor(x):
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                return x

            cam_pose1 = _ensure_batched_tensor(v1_dict["camera_pose"])
            cam_pose2 = _ensure_batched_tensor(v2_dict["camera_pose"])
            cam_intr1 = _ensure_batched_tensor(v1_dict["camera_intrinsics"])
            cam_intr2 = _ensure_batched_tensor(v2_dict["camera_intrinsics"])

            view1 = {
                "img": img1_t.unsqueeze(0).to(device),
                "true_shape": torch.tensor([[img1_t.shape[1], img1_t.shape[2]]]),
                "camera_pose": cam_pose1,
                "camera_intrinsics": cam_intr1,
            }
            view2 = {
                "img": img2_t.unsqueeze(0).to(device),
                "true_shape": torch.tensor([[img2_t.shape[1], img2_t.shape[2]]]),
                "camera_pose": cam_pose2,
                "camera_intrinsics": cam_intr2,
            }

            with torch.no_grad():
                pred1, pred2 = model(view1, view2)

            r, t = compute_pose_metrics_from_dust3r(
                view1, view2, pred1, pred2, pair_idx=0
            )
            r_errors.append(r)
            t_errors.append(t)
        except Exception:
            if _first_err:
                import traceback

                traceback.print_exc()
                _first_err = False

    metrics = summarize_pairs(dataset, r_errors, t_errors)
    return metrics, [], r_errors, t_errors


# ═══════════════════════════════════════════════════════════════════════════════
# Results saving
# ═══════════════════════════════════════════════════════════════════════════════


def load_resumed_results(resume_dir, output_dir, expected_samples=None):
    """Load existing results from a previous run and copy them to the new output dir.

    A result whose num_samples differs from expected_samples[dataset] is incomplete;
    it is neither reused nor copied, so that configuration runs again.

    Returns:
        set of (dataset, model, config) tuples that are already completed,
        dict of (dataset, model, config) -> metrics for the summary table.
    """
    completed = set()
    all_results = {}
    if not osp.isdir(resume_dir):
        print(f"  WARNING: resume_from dir does not exist: {resume_dir}")
        return completed, all_results

    import shutil

    for fname in os.listdir(resume_dir):
        if not fname.endswith("_results.json"):
            continue
        # dataset, model and config come from the JSON: the file name
        # {dataset}_{model}_{config}_results.json is ambiguous because model and
        # config names contain underscores.
        src = osp.join(resume_dir, fname)
        try:
            with open(src) as f:
                data = json.load(f)
            ds = data["dataset"]
            model = data["model"]
            config = data["config"]
            expected = (expected_samples or {}).get(ds)
            if expected is not None and data.get("num_samples") != expected:
                print(
                    f"  Rerunning incomplete result {fname}: "
                    f"{data.get('num_samples')} of {expected} samples"
                )
                continue
            # Extract metrics (everything except per_pair data and metadata)
            metrics = {
                k: v
                for k, v in data.items()
                if k
                not in (
                    "dataset",
                    "model",
                    "config",
                    "per_pair_losses",
                    "per_pair_r_errors",
                    "per_pair_t_errors",
                )
            }
            completed.add((ds, model, config))
            all_results[(ds, model, config)] = metrics

            # Copy result file to new output dir
            dst = osp.join(output_dir, fname)
            if osp.abspath(src) != osp.abspath(dst):
                shutil.copy2(src, dst)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Could not parse {fname}: {e}")

    print(f"  Resumed {len(completed)} completed results from {resume_dir}")
    return completed, all_results


def result_exists(completed_set, dataset_name, model_name, config_name):
    """Check if a (dataset, model, config) result already exists from a resumed run."""
    return (dataset_name, model_name, config_name) in completed_set


def save_result(
    result,
    output_dir,
    dataset_name,
    model_name,
    config_name,
    losses=None,
    r_errors=None,
    t_errors=None,
):
    """Save a single evaluation result to JSON."""
    out = {
        "dataset": dataset_name,
        "model": model_name,
        "config": config_name,
        **result,
    }
    if losses:
        out["per_pair_losses"] = losses
    if r_errors:
        out["per_pair_r_errors"] = r_errors
    if t_errors:
        out["per_pair_t_errors"] = t_errors

    fname = f"{dataset_name}_{model_name}_{config_name}_results.json"
    path = osp.join(output_dir, fname)
    with open(path, "w") as f:
        json.dump(
            out,
            f,
            indent=2,
            default=lambda x: float(x)
            if isinstance(x, (np.floating, np.integer))
            else x,
        )
    return path


def print_summary_table(all_results, output_dir):
    """Print and save summary comparison table."""
    lines = []
    lines.append("=" * 120)
    lines.append("COMPREHENSIVE EVALUATION RESULTS")
    lines.append("=" * 120)

    # Group by dataset
    by_dataset = {}
    for key, metrics in all_results.items():
        ds = key[0]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append((key, metrics))

    for ds, entries in by_dataset.items():
        lines.append(f"\n{'─' * 100}")
        lines.append(f"Dataset: {ds}")
        lines.append(f"{'─' * 100}")
        header = f"{'Model':<30} {'Config':<20} {'Loss':>8} {'RRA@15':>8} {'RTA@15':>8} {'mAA@30':>8} {'N':>6}"
        lines.append(header)
        lines.append("-" * 100)

        # Sort by mAA@30 descending
        entries.sort(key=lambda x: x[1].get("mAA_30", 0), reverse=True)

        for (_, model, config), m in entries:
            loss_str = (
                f"{m.get('mean_loss', float('nan')):.3f}" if "mean_loss" in m else "N/A"
            )
            lines.append(
                f"{model:<30} {config:<20} {loss_str:>8} "
                f"{m.get('RRA_15', 0):>7.2f}% "
                f"{m.get('RTA_15', 0):>7.2f}% "
                f"{m.get('mAA_30', 0):>7.2f}% "
                f"{m.get('num_samples', 0):>6}"
            )

    summary_text = "\n".join(lines)
    print(summary_text)

    # Save to file
    with open(osp.join(output_dir, "comprehensive_results.txt"), "w") as f:
        f.write(summary_text)

    return summary_text


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    print("=" * 80)
    print("Click3R standard benchmark evaluation")
    print("=" * 80)
    print(f"  Datasets: {args.datasets}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Prompt counts: {args.num_prompts}")
    print(f"  Oracle budgets: {args.sampling_budgets}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  GPUs available: {num_gpus}")
    print(f"  Output: {args.output_dir}")
    print(f"  MASt3R: {'skip' if args.skip_mast3r else 'enabled'}")
    print(
        f"  DUSt3R pretrained: {'skip' if args.skip_dust3r_pretrained else 'enabled'}"
    )
    print(f"  DUSt3R trained: {'skip' if args.skip_dust3r_trained else 'enabled'}")
    print(f"  Click3R: {'skip' if args.skip_click3r else 'enabled'}")
    print(f"  VGGT: {'skip' if args.skip_vggt else 'enabled'}")
    print(f"  Pi3X: {'skip' if args.skip_pi3x else 'enabled'}")
    print(
        f"  VGGT@{args.highres_resolution}px: {'skip' if args.skip_vggt_highres else 'enabled'}"
    )
    print(
        f"  Pi3X@{args.highres_resolution}px: {'skip' if args.skip_pi3x_highres else 'enabled'}"
    )
    print(
        f"  VGGT native-{args.native_longside}px: {'skip' if args.skip_vggt_native else 'enabled'}"
    )
    print(
        f"  Pi3X native-{args.native_longside}px: {'skip' if args.skip_pi3x_native else 'enabled'}"
    )
    print(
        f"  Grid oracle: {'skip' if args.skip_grid_oracle else f'enabled (cell_size={args.grid_cell_size}px)'}"
    )
    if args.resume_from:
        print(f"  Resume from: {args.resume_from}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Save config
    with open(osp.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Build datasets ──
    print("\n" + "─" * 80)
    print("Building datasets...")
    print("─" * 80)
    datasets = build_datasets(args)

    if not datasets:
        print("ERROR: No datasets loaded!")
        return

    # ── Loss criterion (matches DUSt3R official eval: no conf weighting, GT scale) ──
    criterion = Regr3D_ScaleShiftInv(L21, gt_scale=True)

    # ── Resume from previous run if specified ──
    completed_set = set()
    all_results = {}  # key: (dataset, model, config) → metrics
    if args.resume_from:
        print(f"\n  Resuming from: {args.resume_from}")
        completed_set, all_results = load_resumed_results(
            args.resume_from,
            args.output_dir,
            {name: len(dataset) for name, dataset in datasets.items()},
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: DUSt3R pretrained baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_dust3r_pretrained:
        print("\n" + "=" * 80)
        print("Phase 1: DUSt3R pretrained (no prompts)")
        print("=" * 80)

        model_instance = eval(args.model)
        model = load_dust3r_model(args.dust3r_pretrained, model_instance, device)
        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "dust3r_pretrained", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "dust3r_pretrained", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "dust3r_pretrained",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(model)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: DUSt3R trained baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_dust3r_trained:
        print("\n" + "=" * 80)
        print("Phase 2: DUSt3R trained (no prompts)")
        print("=" * 80)

        model_instance = eval(args.model)
        model = load_dust3r_model(args.dust3r_checkpoint, model_instance, device)
        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "dust3r_trained", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "dust3r_trained", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "dust3r_trained",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(model)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3: Click3R baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_click3r:
        print("\n" + "=" * 80)
        print("Phase 3: Click3R (no prompts)")
        print("=" * 80)

        model_instance = eval(args.model)
        click3r_model = load_click3r_model(
            args.click3r_checkpoint, model_instance, device
        )
        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "click3r", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                click3r_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "click3r", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "click3r",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        # ══════════════════════════════════════════════════════════════════════
        # Phase 4: Click3R with GT prompts (oracle)
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 80)
        print("Phase 4: Click3R with GT prompts (oracle evaluation)")
        print("=" * 80)

        for num_p in args.num_prompts:
            for budget in args.sampling_budgets:
                config = f"{num_p}p_best{budget}"
                print(f"\n  Config: {num_p} prompts × budget {budget}")

                for ds_name, ds in datasets.items():
                    if result_exists(completed_set, ds_name, "click3r", config):
                        print(f"    {ds_name}: SKIPPED (resumed)")
                        continue
                    metrics, losses, r_errs, t_errs = evaluate_oracle(
                        click3r_model,
                        criterion,
                        ds,
                        device,
                        ds_name,
                        num_prompts=num_p,
                        budget=budget,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    all_results[(ds_name, "click3r", config)] = metrics
                    save_result(
                        metrics,
                        args.output_dir,
                        ds_name,
                        "click3r",
                        config,
                        losses,
                        r_errs,
                        t_errs,
                    )
                    _loss = metrics.get("mean_loss", None)
                    print(
                        f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                        f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
                    )

        # ══════════════════════════════════════════════════════════════════════
        # Phase 4b: Click3R with Grid-Oracle prompt selection
        # ══════════════════════════════════════════════════════════════════════
        if not args.skip_grid_oracle:
            print("\n" + "=" * 80)
            print(
                f"Phase 4b: Click3R with Grid-Oracle prompts "
                f"(cell_size={args.grid_cell_size}px)"
            )
            print("=" * 80)

            config = "grid_oracle_1p_pose"
            for ds_name, ds in datasets.items():
                if result_exists(completed_set, ds_name, "click3r", config):
                    print(f"    {ds_name}: SKIPPED (resumed)")
                    continue
                metrics, losses, r_errs, t_errs = evaluate_grid_oracle(
                    click3r_model,
                    criterion,
                    ds,
                    device,
                    ds_name,
                    cell_size=args.grid_cell_size,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    num_workers=args.num_workers,
                )
                all_results[(ds_name, "click3r", config)] = metrics
                save_result(
                    metrics,
                    args.output_dir,
                    ds_name,
                    "click3r",
                    config,
                    losses,
                    r_errs,
                    t_errs,
                )
                _loss = metrics.get("mean_loss", None)
                print(
                    f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                    f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
                )

        cleanup_model(click3r_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5: MASt3R baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    mast3r_model = None
    if not args.skip_mast3r:
        print("\n" + "=" * 80)
        print("Phase 5: MASt3R baseline (no prompts)")
        print("=" * 80)

        mast3r_model = load_mast3r_model(args.mast3r_model, device)

        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "mast3r", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_mast3r_no_prompts(
                mast3r_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "mast3r", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "mast3r",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5b: VGGT baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_vggt:
        print("\n" + "=" * 80)
        print("Phase 5b: VGGT baseline (no prompts)")
        print("=" * 80)

        from evaluation.baselines.vggt import load_vggt_model

        vggt_model = load_vggt_model(device=device)

        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "vggt", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                vggt_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.wrapper_batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "vggt", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "vggt",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(vggt_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5c: Pi3X baseline (no prompts)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_pi3x:
        print("\n" + "=" * 80)
        print("Phase 5c: Pi3X baseline (no prompts)")
        print("=" * 80)

        from evaluation.baselines.pi3x import load_pi3x_model

        pi3_model = load_pi3x_model(device=device)

        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, "pi3x", "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                pi3_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.wrapper_batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, "pi3x", "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                "pi3x",
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(pi3_model)
        restore_curope_class_patch()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5d/5e: VGGT & Pi3X at --highres_resolution (high-resolution baselines)
    # ══════════════════════════════════════════════════════════════════════════
    need_highres = (not args.skip_vggt_highres) or (not args.skip_pi3x_highres)
    highres_datasets = None
    if need_highres:
        print("\n" + "─" * 80)
        print(
            f"Building high-resolution datasets (resolution={args.highres_resolution}px)..."
        )
        print("─" * 80)
        highres_datasets = build_datasets(args, resolution=args.highres_resolution)

    hr = args.highres_resolution
    vggt_hr_tag = f"vggt_{hr}"
    pi3x_hr_tag = f"pi3x_{hr}"

    # Phase 5d: VGGT @ high-res
    if not args.skip_vggt_highres and highres_datasets:
        print("\n" + "=" * 80)
        print(f"Phase 5d: VGGT baseline ({hr}px)")
        print("=" * 80)

        from evaluation.baselines.vggt import load_vggt_model

        vggt_model = load_vggt_model(device=device)

        for ds_name, ds in highres_datasets.items():
            if result_exists(completed_set, ds_name, vggt_hr_tag, "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                vggt_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.highres_batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, vggt_hr_tag, "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                vggt_hr_tag,
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(vggt_model)

    # Phase 5e: Pi3X @ high-res
    if not args.skip_pi3x_highres and highres_datasets:
        print("\n" + "=" * 80)
        print(f"Phase 5e: Pi3X baseline ({hr}px)")
        print("=" * 80)

        from evaluation.baselines.pi3x import load_pi3x_model

        pi3_model = load_pi3x_model(device=device)

        for ds_name, ds in highres_datasets.items():
            if result_exists(completed_set, ds_name, pi3x_hr_tag, "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, losses, r_errs, t_errs = evaluate_no_prompts(
                pi3_model,
                criterion,
                ds,
                device,
                ds_name,
                batch_size=args.highres_batch_size,
                num_workers=args.num_workers,
            )
            all_results[(ds_name, pi3x_hr_tag, "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                pi3x_hr_tag,
                "no_prompts",
                losses,
                r_errs,
                t_errs,
            )
            _loss = metrics.get("mean_loss", None)
            print(
                f"    {ds_name}: loss={'N/A' if _loss is None else f'{_loss:.3f}'} "
                f"mAA@30={metrics.get('mAA_30', 0):.2f}%"
            )

        cleanup_model(pi3_model)
        restore_curope_class_patch()

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5f: VGGT @ native aspect ratio (long side --native_longside, 14-aligned)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_vggt_native:
        print("\n" + "=" * 80)
        print(f"Phase 5f: VGGT native aspect-ratio ({args.native_longside}px longside)")
        print("=" * 80)

        from evaluation.baselines.vggt import load_vggt_model

        vggt_model = load_vggt_model(device=device)
        tag = f"vggt_native{args.native_longside}"

        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, tag, "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, _, r_errs, t_errs = evaluate_native_resolution(
                vggt_model, ds, device, ds_name, target_longside=args.native_longside
            )
            all_results[(ds_name, tag, "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                tag,
                "no_prompts",
                r_errors=r_errs,
                t_errors=t_errs,
            )
            print(
                f"    {ds_name}: mAA@30={metrics.get('mAA_30', 0):.2f}% "
                f"({metrics.get('num_samples', 0)} samples)"
            )

        cleanup_model(vggt_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5g: Pi3X @ native aspect ratio (long side --native_longside, 14-aligned)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_pi3x_native:
        print("\n" + "=" * 80)
        print(f"Phase 5g: Pi3X native aspect-ratio ({args.native_longside}px longside)")
        print("=" * 80)

        from evaluation.baselines.pi3x import load_pi3x_model

        pi3_model = load_pi3x_model(device=device)
        tag = f"pi3x_native{args.native_longside}"

        for ds_name, ds in datasets.items():
            if result_exists(completed_set, ds_name, tag, "no_prompts"):
                print(f"    {ds_name}: SKIPPED (resumed)")
                continue
            metrics, _, r_errs, t_errs = evaluate_native_resolution(
                pi3_model, ds, device, ds_name, target_longside=args.native_longside
            )
            all_results[(ds_name, tag, "no_prompts")] = metrics
            save_result(
                metrics,
                args.output_dir,
                ds_name,
                tag,
                "no_prompts",
                r_errors=r_errs,
                t_errors=t_errs,
            )
            print(
                f"    {ds_name}: mAA@30={metrics.get('mAA_30', 0):.2f}% "
                f"({metrics.get('num_samples', 0)} samples)"
            )

        cleanup_model(pi3_model)
        restore_curope_class_patch()

    if mast3r_model is not None:
        cleanup_model(mast3r_model)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print_summary_table(all_results, args.output_dir)

    # Save all results as single JSON (with serializable keys)
    all_results_serializable = {}
    for (ds, model, config), metrics in all_results.items():
        key = f"{ds}/{model}/{config}"
        all_results_serializable[key] = metrics

    with open(osp.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(
            all_results_serializable,
            f,
            indent=2,
            default=lambda x: float(x)
            if isinstance(x, (np.floating, np.integer))
            else x,
        )

    print(f"\nAll results saved to: {args.output_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
