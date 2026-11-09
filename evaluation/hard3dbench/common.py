"""Shared Hard3DBench protocol (Table 2): models, inputs, prompts, metrics, results.

The Hard3DBench evaluators use these functions so that every model sees the same
224 x 224 inputs, the same ground-truth correspondence prompts and the same pose
and pointmap metrics as the paper.
"""

import contextlib
import hashlib
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from click3r.inference import make_batch_symmetric
from click3r.model import Click3R
from click3r.metrics import (
    calculate_auc_np,
    estimate_pose_from_pts3d,
    rotation_angle_single,
    translation_angle_single,
)
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import ImgNorm
from evaluation.baselines.loading import load_click3r_model, load_dust3r_model

REPO_ROOT = Path(__file__).resolve().parents[2]
LISTS_DIR = REPO_ROOT / "click3r/configs/hard3dbench"
IMAGE_SIZE = 224
SEED = 42
SPLIT_NAMES = {"indist": "In-Distribution", "ood": "Out-of-Distribution"}
SUBSET_NAMES = {"two_view": "2-view", "five_view": "5-view"}
DUST3R_MODELS = ("click3r", "dust3r_pretrained", "dust3r_retrained")


# ─── models ──────────────────────────────────────────────────────────────────


def architecture():
    """ViT-L/224 DUSt3R with correspondence adapters, as all three checkpoints.

    Uses the checkpoints' PatchEmbedDust3R (as click3r.model.load_model does), not
    the ManyAR_PatchEmbed of evaluation.data.create_model: both agree in float32,
    but ManyAR keeps the encoder stream in float32 under bf16 autocast and so
    changes the bf16 numbers of Table 2.
    """
    return Click3R(
        pos_embed="RoPE100",
        patch_embed_cls="PatchEmbedDust3R",
        img_size=(IMAGE_SIZE, IMAGE_SIZE),
        head_type="linear",
        output_mode="pts3d",
        depth_mode=("exp", -math.inf, math.inf),
        conf_mode=("exp", 1, math.inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        landscape_only=False,
    )


def load_model(name, checkpoint, device):
    """Load Click3R or a DUSt3R baseline (adapters bypassed) for evaluation."""
    if name != "click3r":
        return load_dust3r_model(checkpoint, architecture(), device)
    model = load_click3r_model(checkpoint, architecture(), device)
    if model.prompts_enabled is not True:
        raise RuntimeError("Click3R loaded without correspondence prompts")
    return model


def precision_for(device):
    """bfloat16 on GPUs with compute capability >= 8, float16 on older GPUs."""
    if device.type != "cuda":
        return "float32"
    major, _ = torch.cuda.get_device_capability(device)
    return "bfloat16" if major >= 8 else "float16"


def autocast(device, precision):
    if precision == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=getattr(torch, precision))


@torch.no_grad()
def predict(model, img1, img2, precision, prompts=None):
    """Run one symmetrized pair and return the (view 1, view 2) direction.

    Returns view-1 and view-2 pointmaps in the first camera's frame (H, W, 3) and
    their sigmoid confidences (H, W), all float32 numpy arrays.
    """
    device = next(model.parameters()).device
    shape = torch.tensor([img1.shape[-2:]], dtype=torch.int32)
    view1 = {"img": img1.to(device), "true_shape": shape, "idx": [0], "instance": ["0"]}
    view2 = {"img": img2.to(device), "true_shape": shape, "idx": [1], "instance": ["1"]}
    if prompts is not None:
        for view, sparse in zip((view1, view2), prompts, strict=True):
            view["sparse_prompts"] = {k: v.to(device) for k, v in sparse.items()}
    view1, view2 = make_batch_symmetric((view1, view2))
    with autocast(device, precision):
        pred1, pred2 = model(view1, view2)
    confs = [pred["conf"][0].float() for pred in (pred1, pred2)]
    return (
        pred1["pts3d"][0].float().cpu().numpy(),
        pred2["pts3d_in_other_view"][0].float().cpu().numpy(),
        torch.sigmoid(confs[0]).cpu().numpy(),
        torch.sigmoid(confs[1]).cpu().numpy(),
    )


# ─── inputs ──────────────────────────────────────────────────────────────────

# Out-of-Distribution crop: the size-224 branch of load_images in DUSt3R
# dust3r/utils/image.py (https://github.com/naver/dust3r), always with LANCZOS
# and without EXIF rotation, plus the matching depth and intrinsics transform.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified by the Click3R authors.


def crop_params(width, height, size=IMAGE_SIZE):
    """Resize the long edge to round(size * aspect), then center-crop a square."""
    long_edge = round(size * max(width / height, height / width))
    resize_w = int(round(width * long_edge / max(width, height)))
    resize_h = int(round(height * long_edge / max(width, height)))
    half = min(resize_w // 2, resize_h // 2)
    return {
        "orig_W": width,
        "orig_H": height,
        "resize_W": resize_w,
        "resize_H": resize_h,
        "crop_x0": resize_w // 2 - half,
        "crop_y0": resize_h // 2 - half,
        "final_W": 2 * half,
        "final_H": 2 * half,
    }


def load_image(path, size=IMAGE_SIZE):
    """Return the normalized (1, 3, size, size) image and its crop parameters."""
    image = Image.open(path).convert("RGB")
    crop = crop_params(*image.size, size)
    image = image.resize((crop["resize_W"], crop["resize_H"]), Image.LANCZOS)
    x0, y0 = crop["crop_x0"], crop["crop_y0"]
    image = image.crop((x0, y0, x0 + crop["final_W"], y0 + crop["final_H"]))
    return ImgNorm(image)[None], crop


def crop_depth(depth, crop):
    """Nearest-neighbour resize and crop of ground-truth depth onto the image."""
    size = (crop["resize_W"], crop["resize_H"])
    depth = np.array(Image.fromarray(depth).resize(size, Image.NEAREST))
    y0, x0 = crop["crop_y0"], crop["crop_x0"]
    depth = depth.astype(np.float32)
    return depth[y0 : y0 + crop["final_H"], x0 : x0 + crop["final_W"]]


def crop_intrinsics(K, crop):
    K = np.asarray(K).copy().astype(np.float64)
    K[0, :] *= crop["resize_W"] / crop["orig_W"]
    K[1, :] *= crop["resize_H"] / crop["orig_H"]
    K[0, 2] -= crop["crop_x0"]
    K[1, 2] -= crop["crop_y0"]
    return K


# In-Distribution views use the bundled DUSt3R dataset crop (centered on the
# principal point, LANCZOS rescale, final crop), exactly as the DUSt3R loaders.
_DATASET_CROP = BaseStereoViewDataset(resolution=IMAGE_SIZE)


def crop_like_dust3r_dataset(image, depth, K, size=IMAGE_SIZE, info=None):
    """Return (normalized image, depth, K) of a size x size DUSt3R dataset view."""
    image, depth, K = _DATASET_CROP._crop_resize_if_necessary(
        image, depth, K, (size, size), info=info
    )
    return ImgNorm(image)[None], depth, K


# ─── ground-truth prompts ────────────────────────────────────────────────────


def sample_prompts(depth1, depth2, c2w1, c2w2, K1, K2, num_prompts, seed=SEED):
    """Sample up to K ground-truth correspondences (pixel x, y) for one pair.

    Candidates are min(n, 10 K) random valid view-1 pixels, as in the paper's
    2-view runs (a larger pool, such as 80, would give K = 1 and K = 4 prompts to
    pairs that have none in the paper). Each is projected into view 2 with
    ground-truth depth and poses and kept when it lands inside the image in front
    of the camera and agrees with view-2 depth within 15%. View-2 coordinates stay
    sub-pixel.
    Returns None without a valid prompt.
    """
    height, width = depth1.shape
    pixels = np.argwhere(depth1 > 0.01)
    if len(pixels) < 10:
        return None
    rng = np.random.RandomState(seed)
    count = min(len(pixels), num_prompts * 10)
    candidates = pixels[rng.choice(len(pixels), count, replace=False)]
    rows, cols = candidates[:, 0], candidates[:, 1]
    z1 = depth1[rows, cols].astype(np.float64)
    x1 = (cols - K1[0, 2]) * z1 / K1[0, 0]
    y1 = (rows - K1[1, 2]) * z1 / K1[1, 1]
    points = np.stack([x1, y1, z1, np.ones_like(z1)], axis=-1)
    world = (c2w1.astype(np.float64) @ points.T).T[:, :3]
    world = np.hstack([world, np.ones((len(world), 1))])
    cam2 = (np.linalg.inv(c2w2.astype(np.float64)) @ world.T).T
    z2 = cam2[:, 2]
    u2 = K2[0, 0] * cam2[:, 0] / (z2 + 1e-8) + K2[0, 2]
    v2 = K2[1, 1] * cam2[:, 1] / (z2 + 1e-8) + K2[1, 2]
    keep = np.where((u2 >= 0) & (u2 < width) & (v2 >= 0) & (v2 < height) & (z2 > 0))[0]
    if len(keep) < 1:
        return None
    ui = np.clip(u2[keep].astype(int), 0, width - 1)
    vi = np.clip(v2[keep].astype(int), 0, height - 1)
    depth2_at = depth2[vi, ui].astype(np.float64)
    agree = (depth2_at > 0.01) & (
        np.abs(z2[keep] - depth2_at) / (depth2_at + 1e-6) < 0.15
    )
    keep = keep[agree]
    if len(keep) < 1:
        return None
    if len(keep) > num_prompts:
        keep = keep[rng.choice(len(keep), num_prompts, replace=False)]
    coords1 = np.stack([cols[keep], rows[keep]], axis=-1).astype(np.float32)
    coords2 = np.stack([u2[keep], v2[keep]], axis=-1).astype(np.float32)
    return coords1, coords2


def prompt_inputs(coords1, coords2, num_prompts, height, width):
    """Pad K correspondences to the model's per-view sparse prompt dictionaries."""
    used = len(coords1)
    padded = np.zeros((2, num_prompts, 2), dtype=np.float32)
    padded[0, :used], padded[1, :used] = coords1, coords2
    valid = torch.zeros(1, num_prompts, dtype=torch.bool)
    valid[0, :used] = True
    pair_ids = torch.arange(num_prompts, dtype=torch.long)[None]
    shape = torch.tensor([[height, width]], dtype=torch.long)
    return tuple(
        {
            "coords": torch.from_numpy(padded[i])[None],
            "image_ids": torch.full((1, num_prompts), i, dtype=torch.long),
            "pair_ids": pair_ids.clone(),
            "shape": shape.clone(),
            "valid_mask": valid.clone(),
        }
        for i in (0, 1)
    )


# ─── metrics ─────────────────────────────────────────────────────────────────


def relative_pose_error(pts1, pts2, conf1, conf2, c2w1, c2w2, K1, K2):
    """Relative rotation and translation-direction errors (degrees) from PnP.

    Each camera is recovered with PnP-RANSAC on its predicted pointmap, using the
    pixels whose sigmoid confidence exceeds the mean and ground-truth intrinsics.
    A failed PnP returns (nan, nan); Table 2 excludes such pairs from mAA.
    """
    cam1 = estimate_pose_from_pts3d(pts1, conf=conf1, K=K1)
    cam2 = estimate_pose_from_pts3d(pts2, conf=conf2, K=K2)
    if cam1 is None or cam2 is None:
        return math.nan, math.nan
    gt = torch.from_numpy(np.linalg.inv(c2w1) @ c2w2).float()
    pred = torch.inverse(cam1) @ cam2
    rre = rotation_angle_single(gt[:3, :3], pred[:3, :3])
    rte = translation_angle_single(gt[:3, 3], pred[:3, 3], handle_ambiguity=True)
    return float(rre.item()), float(rte.item())


def camera_points(depth, K):
    """Ground-truth (H, W, 3) points of a depth map in its camera's frame (float64)."""
    height, width = depth.shape
    rows, cols = np.mgrid[:height, :width].astype(np.float64)
    z = depth.astype(np.float64)
    x = (cols - K[0, 2]) * z / K[0, 0]
    y = (rows - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def pointmap_error(pts1, pts2, depth1, depth2, c2w1, c2w2, K1, K2):
    """2-view pointmap error (pointmap_error_views) over pixels with depth > 0.01."""
    valid1, valid2 = depth1 > 0.01, depth2 > 0.01
    if not (valid1.any() and valid2.any()):
        return None
    gt1 = camera_points(depth1, K1)
    gt2 = camera_points(depth2, K2)
    rel = np.linalg.inv(np.asarray(c2w1, np.float64)) @ np.asarray(c2w2, np.float64)
    flat = gt2.reshape(-1, 3)
    gt2 = ((rel[:3, :3] @ flat.T).T + rel[:3, 3]).reshape(gt2.shape)
    return pointmap_error_views([pts1, pts2], [gt1, gt2], [valid1, valid2])


def pointmap_error_views(pred, gt, valid):
    """Scale- and shift-invariant L2.1 pointmap error over all views of a sample.

    pred, gt: per-view (H, W, 3) pointmaps in the first camera's frame; valid:
    per-view pixel masks. Normalizes as DUSt3R's Regr3D_ScaleShiftInv(L21,
    gt_scale=True): the prediction is divided by its mean point distance, both
    sets are shifted by their median depth, and the prediction is rescaled to the
    ground-truth median distance to the median point. The per-pixel L2 error is
    then averaged over the valid pixels of all views together; DUSt3R's loss
    instead sums the per-view means. The unit is the ground-truth unit.
    """
    if not any(mask.any() for mask in valid):
        return None
    pred = [points.astype(np.float64) for points in pred]
    gt = [points.astype(np.float64) for points in gt]
    pooled = np.concatenate([p[m] for p, m in zip(pred, valid, strict=True)])
    scale = max(float(np.linalg.norm(pooled, axis=-1).mean()), 1e-6)
    pred = [points / scale for points in pred]

    pred_z = np.median(
        np.concatenate([p[..., 2][m] for p, m in zip(pred, valid, strict=True)])
    )
    gt_z = np.median(
        np.concatenate([g[..., 2][m] for g, m in zip(gt, valid, strict=True)])
    )
    for p, g in zip(pred, gt, strict=True):
        p[..., 2] -= pred_z
        g[..., 2] -= gt_z

    pooled = np.concatenate([p[m] for p, m in zip(pred, valid, strict=True)])
    gt_pooled = np.concatenate([g[m] for g, m in zip(gt, valid, strict=True)])
    pred_scale = np.median(np.linalg.norm(pooled - np.median(pooled, axis=0), axis=-1))
    pred_scale = float(np.clip(pred_scale, 1e-3, 1e3))
    gt_center = np.median(gt_pooled, axis=0)
    gt_scale = float(np.median(np.linalg.norm(gt_pooled - gt_center, axis=-1)))
    factor = max(gt_scale, 1e-6) / pred_scale

    errors = [
        np.linalg.norm((p * factor)[m] - g[m], axis=-1)
        for p, g, m in zip(pred, gt, valid, strict=True)
    ]
    return finite_or_none(float(np.concatenate(errors).mean()))


def finite_or_none(value):
    return value if value is not None and math.isfinite(value) else None


# ─── lists and results ───────────────────────────────────────────────────────


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load_list(subset, split):
    """Read a Hard3DBench sample list; returns (samples, file name, sha256)."""
    path = LISTS_DIR / f"{subset}_{split}.json"
    return json.loads(path.read_text())["samples"], path.name, sha256(path)


def result_path(results_dir, subset, split, model, num_prompts=None):
    name = model if num_prompts is None else f"{model}_k{num_prompts}"
    return Path(results_dir) / f"{subset}_{split}" / f"{name}.json"


def run_metadata(subset, split, model, num_prompts, list_name, list_sha256, **extra):
    """Metadata shared by every result file; extra keys are added verbatim."""
    meta = {
        "benchmark": "Hard3DBench",
        "subset": SUBSET_NAMES[subset],
        "split": SPLIT_NAMES[split],
        "model": model,
        "num_prompts": num_prompts,
        "manifest": list_name,
        "manifest_sha256": list_sha256,
    }
    meta.update(extra)
    meta.setdefault("seed", SEED)
    return meta


# Settings that must match, together with the device type (`device_type`), for an
# existing result file to count as complete.
RESUME_KEYS = (
    "benchmark",
    "subset",
    "split",
    "model",
    "num_prompts",
    "manifest_sha256",
    "checkpoint_sha256",
    "precision",
    "seed",
)


def device_type(metadata):
    """cpu or cuda: CPU and GPU runs give different values, also in float32."""
    return str(metadata.get("device")).split(":")[0]


def is_complete(path, metadata, sample_ids):
    """True if `path` holds these samples under the same settings.

    An existing file for other samples or settings raises instead of being
    overwritten; delete it or use another results directory.
    """
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    changed = [
        key for key in RESUME_KEYS if saved["metadata"].get(key) != metadata.get(key)
    ]
    if device_type(saved["metadata"]) != device_type(metadata):
        changed.append("device")
    saved_ids = [sample["sample_id"] for sample in saved["samples"]]
    if changed or saved_ids != list(sample_ids):
        reason = f"settings {changed}" if changed else "a different sample selection"
        raise FileExistsError(f"{path} exists with {reason}")
    return True


def write_results(path, metadata, samples):
    """Write {metadata, samples} atomically; floats keep full precision."""
    metadata = dict(metadata, num_samples=len(samples))
    lines = ",\n".join(json.dumps(sample, allow_nan=False) for sample in samples)
    text = (
        '{\n"metadata": '
        + json.dumps(metadata, indent=2, allow_nan=False)
        + ',\n"samples": [\n'
        + lines
        + "\n]\n}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


# ─── Table 2 cells ───────────────────────────────────────────────────────────

ROW_NAMES = {
    "dust3r_pretrained": "DUSt3R",
    "dust3r_retrained": "DUSt3R (retrained)",
    "mast3r_512": "MASt3R (512)",
    "vggt_224": "VGGT (224)",
    "vggt_518": "VGGT (518)",
    "pi3_224": "Pi3 (224)",
    "pi3_518": "Pi3 (518)",
}


def row_name(model, num_prompts):
    """Table 2 row of a model and K."""
    if model != "click3r":
        return ROW_NAMES[model]
    if num_prompts == 0:
        return "Click3R w/o prompts"
    return f"Click3R ({num_prompts} prompt{'' if num_prompts == 1 else 's'})"


def cell_values(subset, rows):
    """Finite-only mAA@30 (%) and the median pointmap error of one result file.

    5-view mAA@30 pools the ten pairwise poses of every tuple. A value is None when
    none is finite.
    """
    if subset == "two_view":
        rre = [row["rre"] for row in rows]
        rte = [row["rte"] for row in rows]
    else:
        rre = [value for row in rows for value in row["pair_rre"]]
        rte = [value for row in rows for value in row["pair_rte"]]
    rre, rte, errors = (
        np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        for values in (rre, rte, [row["pointmap_error"] for row in rows])
    )
    finite = np.isfinite(rre) & np.isfinite(rte)
    errors = errors[np.isfinite(errors)]
    return (
        float(calculate_auc_np(rre[finite], rte[finite], 30) * 100)
        if finite.any()
        else None,
        float(np.median(errors)) if len(errors) else None,
    )


def _half_up(value, digits):
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP)


def table2_line(path, subset, split, model, num_prompts):
    """The Table 2 cells of a result file in one line, rounded as in the paper.

    Half up: mAA@30 to one decimal (5-view from its two-decimal value, as printed)
    and the pointmap median to three decimals.
    """
    maa, pointmap = cell_values(subset, json.loads(Path(path).read_text())["samples"])
    if maa is not None and subset == "five_view":
        maa = _half_up(maa, 2)
    maa = "n/a" if maa is None else _half_up(maa, 1)
    pointmap = "n/a" if pointmap is None else _half_up(pointmap, 3)
    column = f"{SUBSET_NAMES[subset]} {SPLIT_NAMES[split]}"
    return (
        f"{row_name(model, num_prompts)}  {column}  mAA@30 {maa}  pointmap {pointmap}"
    )
