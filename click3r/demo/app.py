#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Portions adapted from DUSt3R (dust3r/model.py, dust3r/demo.py,
# dust3r/utils/image.py) and CroCo (models/blocks.py, Copyright (C) 2022-present
# Naver Corporation, CC BY-NC-SA 4.0).
# Modified by the Click3R authors: interactive correspondence-click demo.
#
# --------------------------------------------------------
# Interactive Demo script for DUSt3R with SAM-style correspondence prompts
# --------------------------------------------------------

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

# Resolve bundled dependencies when launching directly from the source tree.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
DUST3R_PATH = os.path.join(REPO_ROOT, "third_party", "dust3r")
SAM_PATH = os.path.join(REPO_ROOT, "third_party", "segment-anything")
VGGT_PATH = os.path.join(REPO_ROOT, "third_party", "vggt")
PI3_PATH = os.path.join(REPO_ROOT, "third_party", "Pi3")

for path in [REPO_ROOT, DUST3R_PATH, SAM_PATH, VGGT_PATH, PI3_PATH]:
    if path not in sys.path:
        sys.path.insert(0, path)


import gradio as gr
import matplotlib
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageOps

# iPhone photos are HEIC; PIL reads HEIC/HEIF only after pillow-heif registers its opener.
# Without pillow-heif the demo still runs but cannot read HEIC.
if importlib.util.find_spec("pillow_heif") is not None:
    from pillow_heif import register_heif_opener

    register_heif_opener()

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.demo import get_3D_model_from_scene
from dust3r.utils.device import collate_with_cat, to_numpy
from dust3r.utils.geometry import normalize_pointcloud

# Build the model from saved constructor metadata.
from click3r.model import Click3R, model_from_spec  # noqa: F401

from click3r.demo.session import (
    DemoSession as DemoSession,
    current_session as _session,
    new_session as new_session,
    session_edit as session_edit,
    session_scope as session_scope,
)
from click3r.demo.display import ReconstructionViewer, point_size_control
from click3r.demo.views import prepare_model_view
from click3r.demo.results import ReconstructionResult

DEBUG = False
PAPER_TITLE = (
    "Click3R: Interactive Stereo 3D Reconstruction with Sparse Correspondence Clicks"
)
DISPLAY_LONG_SIDE = 640  # Pixels; small inputs would otherwise be hard to click.
CLICK_MARKER_SCALE = 0.03  # 3D click marker radius per unit of framed scene size.
VIEWER_FOV = 0.8  # Radians; vertical field of view of Gradio's 3D viewer.
EXAMPLE_IMAGES = {}  # Decoded example inputs by resolved path, filled by load_examples.
# Optional comparison models are imported lazily by their load functions.
VGGT_MODEL_NAME = "facebook/VGGT-1B"
PI3_MODEL_NAME = "yyfz233/Pi3X"


def debug_log(message):
    """Print a message when DEBUG is set."""
    if DEBUG:
        print(f"[DEBUG] {message}")


def compute_percentile_threshold(scene, percentile):
    """
    Compute confidence threshold from scene based on percentile.

    Args:
        scene: Scene object from global_aligner
        percentile: Percentile value (0-100)

    Returns:
        float: Confidence threshold at the given percentile
    """
    from dust3r.utils.device import to_numpy

    # Get all confidence values from the scene
    conf_values = []
    for i in range(len(scene.im_conf)):
        conf = to_numpy(scene.im_conf[i])
        conf_values.append(conf.flatten())

    all_conf = np.concatenate(conf_values)

    threshold = np.percentile(all_conf, percentile)

    return threshold


def fix_dust3r_glb_orientation(glb_path):
    """
    Post-process a dust3r GLB file to undo the 180-degree Y rotation
    that dust3r's _convert_scene_output_to_glb applies internally.
    This makes the initial viewing angle correct in Gradio's 3D viewer.

    Args:
        glb_path: Path to the GLB file (modified in-place)
    Returns:
        The same glb_path
    """
    if glb_path is None or not os.path.exists(glb_path):
        return glb_path
    try:
        from scipy.spatial.transform import Rotation as R_scipy

        glb_scene = trimesh.load(glb_path)
        rot_undo = np.eye(4)
        rot_undo[:3, :3] = R_scipy.from_euler("y", np.deg2rad(180)).as_matrix()
        glb_scene.apply_transform(rot_undo)
        glb_scene.export(file_obj=glb_path)
        debug_log(f"Fixed GLB orientation: {glb_path}")
    except Exception as e:
        debug_log(f"Warning: Could not fix GLB orientation: {e}")
    return glb_path


def create_3d_model_from_world_points(
    output_dir,
    pred1,
    pred2,
    view1,
    view2,
    min_conf_percentile=10,
    as_pointcloud=True,
    cam_size=0.05,
    model_name="model",
):
    """
    Create 3D model directly from world-frame points (for VGGT and Pi3).

    Args:
        output_dir: Directory to save the model
        pred1, pred2: Predictions with 'pts3d_world' and 'conf' keys
        view1, view2: View dictionaries with 'img' keys
        min_conf_percentile: Percentile threshold (0-100) - keep points above this percentile
        as_pointcloud: Whether to export as pointcloud
        cam_size: Camera size for visualization
        model_name: Name prefix for the output file

    Returns:
        tuple: (path to the saved GLB file, 4x4 transform applied to the scene or None)
    """
    import trimesh
    from dust3r.utils.device import to_numpy

    # Extract world points and confidence
    pts3d_world_1 = to_numpy(pred1["pts3d_world"])  # (B, H, W, 3)
    pts3d_world_2 = to_numpy(pred2["pts3d_world"])  # (B, H, W, 3)
    conf1 = to_numpy(pred1["conf"])  # (B, H, W)
    conf2 = to_numpy(pred2["conf"])  # (B, H, W)

    # Extract images and denormalize from [-1, 1] to [0, 1]
    img1 = to_numpy(view1["img"])  # (B, 3, H, W)
    img2 = to_numpy(view2["img"])  # (B, 3, H, W)
    img1 = (img1 * 0.5 + 0.5).transpose(0, 2, 3, 1)  # (B, H, W, 3)
    img2 = (img2 * 0.5 + 0.5).transpose(0, 2, 3, 1)  # (B, H, W, 3)

    # One percentile threshold over the confidences of both views.
    conf_combined = np.concatenate([conf1[0].flatten(), conf2[0].flatten()])
    conf_threshold = np.percentile(conf_combined, min_conf_percentile)

    debug_log(
        f"{model_name}: Confidence range [{conf_combined.min():.3f}, {conf_combined.max():.3f}], "
        f"{min_conf_percentile}th percentile threshold = {conf_threshold:.3f}"
    )

    # Create confidence masks using percentile threshold
    mask1 = conf1[0] > conf_threshold
    mask2 = conf2[0] > conf_threshold

    # Combine points and colors from both views
    pts_combined = np.concatenate(
        [pts3d_world_1[0][mask1], pts3d_world_2[0][mask2]], axis=0
    )

    col_combined = np.concatenate([img1[0][mask1], img2[0][mask2]], axis=0)

    debug_log(
        f"{model_name}: Kept {len(pts_combined)} points (view1: {mask1.sum()}, view2: {mask2.sum()})"
    )

    scene = trimesh.Scene()

    if as_pointcloud:
        pct = trimesh.PointCloud(
            pts_combined.reshape(-1, 3), colors=col_combined.reshape(-1, 3)
        )
        scene.add_geometry(pct)
    else:
        # Create meshes from individual views
        from dust3r.viz import pts3d_to_trimesh, cat_meshes

        meshes = []
        meshes.append(pts3d_to_trimesh(img1[0], pts3d_world_1[0], mask1))
        meshes.append(pts3d_to_trimesh(img2[0], pts3d_world_2[0], mask2))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # Add camera frustums if camera poses are available
    if "camera_pose_predicted" in pred1 and "camera_pose_predicted" in pred2:
        from dust3r.viz import add_scene_cam, CAM_COLORS

        cam_pose_1 = to_numpy(pred1["camera_pose_predicted"])[0]  # (4, 4)
        cam_pose_2 = to_numpy(pred2["camera_pose_predicted"])[0]  # (4, 4)

        # Approximate focal length for a 90-degree horizontal field of view.
        H, W = img1[0].shape[:2]
        focal_length = W / 2.0  # W / (2 * tan(45 deg))

        # Add camera visualizations with images displayed on frustums
        add_scene_cam(
            scene,
            cam_pose_1,
            CAM_COLORS[0],
            img1[0],
            focal_length,
            imsize=(W, H),
            screen_width=cam_size,
        )
        add_scene_cam(
            scene,
            cam_pose_2,
            CAM_COLORS[1],
            img2[0],
            focal_length,
            imsize=(W, H),
            screen_width=cam_size,
        )

        debug_log(f"Added camera frustums with images to {model_name}")

    # View the scene from camera 0. DUSt3R's GLB export uses
    # inv(cam0 @ OPENGL @ rot_180y); the 180-degree Y rotation makes the scene
    # face backwards in Gradio's 3D viewer, so it is omitted here.
    glb_transform = None
    if "camera_pose_predicted" in pred1:
        cam0_pose = to_numpy(pred1["camera_pose_predicted"])[0]  # (4, 4) - first camera
        OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        glb_transform = np.linalg.inv(cam0_pose @ OPENGL)
        scene.apply_transform(glb_transform)
        debug_log(
            f"{model_name}: Applied GLB view normalization (centered at camera 0)"
        )

    outfile = os.path.join(output_dir, f"{model_name}.glb")
    debug_log(f"Saving world-frame 3D model to {outfile}")
    scene.export(file_obj=outfile)

    return outfile, glb_transform


def load_click3r(checkpoint_path, device="cuda"):
    """Load the trained DUSt3R model with SAM adapters from checkpoint."""
    debug_log(f"Loading model from {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )

    # Get model args from checkpoint
    if "args" in ckpt and hasattr(ckpt["args"], "model"):
        model_str = ckpt["args"].model
        debug_log(f"Model config from checkpoint: {model_str}")
        # Evaluate the model string to create the model
        model = model_from_spec(model_str)
    elif ckpt.get("metadata", {}).get("model"):
        model = model_from_spec(ckpt["metadata"]["model"])
    else:
        # Fallback: the default 224x224 Click3R configuration.
        debug_log("No model args in checkpoint, using default 224 config")
        model = Click3R(
            pos_embed="RoPE100",
            img_size=(224, 224),
            head_type="linear",
            output_mode="pts3d",
            depth_mode=("exp", -float("inf"), float("inf")),
            conf_mode=("exp", 1, float("inf")),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
        )

    if "model" in ckpt:
        state_dict = ckpt["model"]
        # Remove '_orig_mod.' prefix if it exists (from torch.compile)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            debug_log("Detected torch.compile prefix '_orig_mod.', removing it...")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # Remove 'module.' prefix if it exists (from DDP training)
        if any(k.startswith("module.") for k in state_dict.keys()):
            debug_log("Detected DDP prefix 'module.', removing it...")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        debug_log(
            f"Loaded checkpoint - missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

    model = model.to(device)
    model.eval()

    img_size = model.img_size if hasattr(model, "img_size") else (224, 224)
    debug_log(f"Model loaded and ready - expected img_size: {img_size}")

    return model, img_size


def load_original_dust3r(checkpoint_path, device="cuda"):
    """Load the original DUSt3R model from checkpoint."""
    debug_log(f"Loading original DUSt3R model from {checkpoint_path}")

    try:
        from dust3r.model import (
            AsymmetricCroCo3DStereo as OriginalAsymmetricCroCo3DStereo,
        )

        # Load using from_pretrained for original checkpoints
        model = OriginalAsymmetricCroCo3DStereo.from_pretrained(checkpoint_path).to(
            device
        )
        model.eval()

        img_size = model.img_size if hasattr(model, "img_size") else (224, 224)
        debug_log(f"Original DUSt3R model loaded - img_size: {img_size}")

        return model, img_size
    except Exception as e:
        debug_log(f"Error loading original DUSt3R: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return None, None


def load_vggt_model(model_name=VGGT_MODEL_NAME, device="cuda"):
    """Load VGGT model."""
    debug_log(f"Loading VGGT model: {model_name}")

    try:
        from evaluation.baselines.vggt import VGGTWrapper

        vggt_model = VGGTWrapper(model_name=model_name, device=device)
        debug_log("VGGT model loaded successfully")
        return vggt_model
    except Exception as e:
        debug_log(f"Error loading VGGT: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return None


def load_pi3x_model(model_name=PI3_MODEL_NAME, device="cuda"):
    """Load the Pi3X comparison adapter."""
    try:
        from evaluation.baselines.pi3x import Pi3XWrapper

        return Pi3XWrapper(model_name=model_name, device=device)
    except Exception as exc:
        debug_log(f"Error loading Pi3X: {exc}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return None


def transform_coords_to_dust3r_space(
    coords, orig_shape, size=224, verbose=False, use_simple_scaling=False
):
    """
    Transform coordinates from original image space to DUSt3R's resized+cropped space.

    Args:
        coords: (B, P, 2) pixel coordinates (x, y) in original image space
        orig_shape: (B, 2) original image shape (H, W)
        size: target size (default 224)
        verbose: if True, print debug info
        use_simple_scaling: if True, use simple scaling (x * size/W, y * size/H) instead of resize+crop

    Returns:
        coords_transformed: (B, P, 2) coordinates in DUSt3R space
    """
    B, P, _ = coords.shape
    coords_transformed = torch.zeros_like(coords)

    for b in range(B):
        H_orig, W_orig = orig_shape[b]
        H_orig, W_orig = int(H_orig.item()), int(W_orig.item())

        if use_simple_scaling:
            # Simple scaling: just scale to target size
            scale_x = size / W_orig
            scale_y = size / H_orig

            if verbose:
                debug_log(
                    f"    Simple scaling: {W_orig}x{H_orig} -> {size}x{size} (scale {scale_x:.3f}, {scale_y:.3f})"
                )

            coords_transformed[b, :, 0] = coords[b, :, 0] * scale_x
            coords_transformed[b, :, 1] = coords[b, :, 1] * scale_y
        else:
            # Resize+crop method (matching load_images logic)
            # Step 1: Resize (exactly matching _resize_pil_image logic)
            S = max(W_orig, H_orig)
            long_edge_size = round(size * max(W_orig / H_orig, H_orig / W_orig))
            W_resized = round(W_orig * long_edge_size / S)
            H_resized = round(H_orig * long_edge_size / S)

            scale_x = W_resized / W_orig
            scale_y = H_resized / H_orig

            # Step 2: Center crop to size x size
            cx, cy = W_resized // 2, H_resized // 2
            half = min(cx, cy)
            crop_x_start = cx - half
            crop_y_start = cy - half

            if verbose:
                debug_log(
                    f"    Resize+crop: {W_orig}x{H_orig} -> resize to {W_resized}x{H_resized} -> crop from ({crop_x_start}, {crop_y_start}) to ({crop_x_start + size}, {crop_y_start + size})"
                )

            # Apply transformation: scale then translate by crop
            coords_transformed[b, :, 0] = coords[b, :, 0] * scale_x - crop_x_start
            coords_transformed[b, :, 1] = coords[b, :, 1] * scale_y - crop_y_start

    return coords_transformed


def sample_3d_at_prompts(pts3d, coords, shape):
    """
    Sample 3D points at prompted pixel locations using bilinear interpolation.

    Args:
        pts3d: (B, H, W, 3) 3D points
        coords: (B, P, 2) pixel coordinates (x, y) in pts3d space
        shape: (B, 2) image shape (H, W) of pts3d

    Returns:
        sampled_pts: (B, P, 3) 3D points at prompted locations
    """
    B, H, W, _ = pts3d.shape
    device = pts3d.device

    # Ensure coords and shape are on the correct device
    coords = coords.to(device)
    shape = shape.to(device)

    # Normalize coordinates to [-1, 1] for grid_sample
    W_img = shape[:, 1:2].float().unsqueeze(1)  # (B, 1, 1)
    H_img = shape[:, 0:1].float().unsqueeze(1)  # (B, 1, 1)

    x_norm = 2.0 * coords[:, :, 0:1] / (W_img - 1) - 1.0  # (B, P, 1)
    y_norm = 2.0 * coords[:, :, 1:2] / (H_img - 1) - 1.0  # (B, P, 1)
    coords_norm = torch.cat([x_norm, y_norm], dim=-1)  # (B, P, 2)

    # Permute pts3d for grid_sample: (B, H, W, 3) -> (B, 3, H, W)
    pts3d_permuted = pts3d.permute(0, 3, 1, 2)

    # Create grid: (B, P, 1, 2)
    grid = coords_norm.unsqueeze(2)

    # Sample: output (B, 3, P, 1)
    sampled = torch.nn.functional.grid_sample(
        pts3d_permuted, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    # Reshape to (B, P, 3)
    sampled_pts = sampled.squeeze(-1).permute(0, 2, 1)

    return sampled_pts


def compute_correspondence_metrics(
    pred1, pred2, prompts1, prompts2, norm_mode="avg_dis"
):
    """
    Compute metrics for correspondence prompts.

    Args:
        pred1, pred2: Prediction dicts with 'pts3d', 'pts3d_in_other_view', 'conf'
        prompts1, prompts2: Prompt dicts with 'coords', 'shape', 'valid_mask'
        norm_mode: Normalization mode for 3D points

    Returns:
        dict: Metrics including correspondence_distance, mean_conf_at_prompts
    """
    if prompts1 is None or prompts2 is None:
        return {
            "correspondence_distance": 0.0,
            "mean_conf_at_prompts": 0.0,
            "num_valid_prompts": 0,
        }

    # Extract prompt info
    coords1 = prompts1["coords"]  # (B, P, 2)
    coords2 = prompts2["coords"]  # (B, P, 2)
    valid_mask = prompts1["valid_mask"]  # (B, P)
    shape1 = prompts1["shape"]  # (B, 2)
    shape2 = prompts2["shape"]

    # Get predicted 3D points
    pts3d_1 = pred1["pts3d"]  # (B, H, W, 3)
    pts3d_2 = pred2["pts3d_in_other_view"]  # (B, H, W, 3)

    # Apply normalization (same as training)
    if norm_mode:
        B, H, W, _ = pts3d_1.shape
        valid1 = torch.ones((B, H, W), dtype=torch.bool, device=pts3d_1.device)
        valid2 = torch.ones((B, H, W), dtype=torch.bool, device=pts3d_2.device)
        pts3d_1, pts3d_2 = normalize_pointcloud(
            pts3d_1, pts3d_2, norm_mode=norm_mode, valid1=valid1, valid2=valid2
        )

    # Sample 3D points at prompted locations
    sampled_pts1 = sample_3d_at_prompts(pts3d_1, coords1, shape1)  # (B, P, 3)
    sampled_pts2 = sample_3d_at_prompts(pts3d_2, coords2, shape2)  # (B, P, 3)

    # Measure mean 3D distance between clicked points
    if valid_mask.any():
        pts1_valid = sampled_pts1[valid_mask]  # (N, 3)
        pts2_valid = sampled_pts2[valid_mask]  # (N, 3)

        # L2 distance
        corr_distance = torch.norm(pts1_valid - pts2_valid, dim=-1).mean().item()

        # Get confidence at prompted locations
        conf1 = pred1["conf"]  # (B, H, W)
        conf2 = pred2["conf"]  # (B, H, W)

        # Sample confidence at prompts (add channel dim for grid_sample)
        conf1_at_prompts = sample_3d_at_prompts(
            conf1.unsqueeze(-1), coords1, shape1
        ).squeeze(-1)  # (B, P)
        conf2_at_prompts = sample_3d_at_prompts(
            conf2.unsqueeze(-1), coords2, shape2
        ).squeeze(-1)  # (B, P)

        # Mean confidence at valid prompts
        mean_conf = (
            (conf1_at_prompts[valid_mask].mean() + conf2_at_prompts[valid_mask].mean())
            / 2.0
        ).item()

        return {
            "correspondence_distance": corr_distance,
            "mean_conf_at_prompts": mean_conf,
            "num_valid_prompts": int(valid_mask.sum()),
        }
    else:
        return {
            "correspondence_distance": 0.0,
            "mean_conf_at_prompts": 0.0,
            "num_valid_prompts": 0,
        }


def coords_to_token_indices(coords_xy, shape_hw, image_embedding_size=(14, 14)):
    """Return the flat index and (x, y) grid position of the patch token containing each pixel."""
    B, P, _ = coords_xy.shape
    Ht, Wt = image_embedding_size

    H_img = shape_hw[:, 0].float().clamp(min=1).view(B, 1)
    W_img = shape_hw[:, 1].float().clamp(min=1).view(B, 1)
    patch_h = (H_img / float(Ht)).view(B, 1)
    patch_w = (W_img / float(Wt)).view(B, 1)

    x_pix = coords_xy[..., 0].float()
    y_pix = coords_xy[..., 1].float()
    x_tok = (x_pix / patch_w).long()
    y_tok = (y_pix / patch_h).long()

    x_tok = x_tok.clamp(0, Wt - 1)
    y_tok = y_tok.clamp(0, Ht - 1)
    token_indices = y_tok * Wt + x_tok

    return token_indices, x_tok, y_tok


def compute_cross_attention_with_weights_rope(
    cross_attn_module, query, key, value, qpos, kpos
):
    """Compute cross-attention with RoPE and return weights."""
    B, Nq, C = query.shape
    Nk = key.shape[1]
    Nv = value.shape[1]

    q = (
        cross_attn_module.projq(query)
        .reshape(B, Nq, cross_attn_module.num_heads, C // cross_attn_module.num_heads)
        .permute(0, 2, 1, 3)
    )
    k = (
        cross_attn_module.projk(key)
        .reshape(B, Nk, cross_attn_module.num_heads, C // cross_attn_module.num_heads)
        .permute(0, 2, 1, 3)
    )
    v = (
        cross_attn_module.projv(value)
        .reshape(B, Nv, cross_attn_module.num_heads, C // cross_attn_module.num_heads)
        .permute(0, 2, 1, 3)
    )

    if cross_attn_module.rope is not None:
        q = cross_attn_module.rope(q, qpos)
        k = cross_attn_module.rope(k, kpos)

    attn = (q @ k.transpose(-2, -1)) * cross_attn_module.scale
    attn = attn.softmax(dim=-1)
    attn_weights = attn
    attn = cross_attn_module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
    x = cross_attn_module.proj(x)
    x = cross_attn_module.proj_drop(x)

    weights_averaged = attn_weights.mean(dim=1)
    return x, weights_averaged


def forward_block_with_attention(block, feat_q, feat_kv, pos_q, pos_kv):
    """Forward through decoder block while capturing attention."""
    x = feat_q + block.drop_path(block.attn(block.norm1(feat_q), pos_q))

    cross_attn_out, attn_weights = compute_cross_attention_with_weights_rope(
        block.cross_attn,
        block.norm2(x),
        block.norm_y(feat_kv),
        block.norm_y(feat_kv),
        pos_q,
        pos_kv,
    )
    x = x + block.drop_path(cross_attn_out)
    output = x + block.drop_path(block.mlp(block.norm3(x)))

    return output, attn_weights


def run_decoder_with_attention(model, f1, f2, pos1, pos2):
    """Run decoder while capturing attention weights.

    Returns same format as model._decoder():
    - dec1, dec2: Lists of decoder outputs (one per layer)
    - attention_maps: List of attention dicts (one per layer)
    """
    attention_maps = []
    final_output = [(f1, f2)]

    for blk1, blk2 in zip(model.dec_blocks, model.dec_blocks2, strict=False):
        cur_f1, cur_f2 = final_output[-1]

        f1_out, weights_1to2 = forward_block_with_attention(
            blk1, cur_f1, cur_f2, pos1, pos2
        )
        f2_out, weights_2to1 = forward_block_with_attention(
            blk2, cur_f2, cur_f1, pos2, pos1
        )

        attention_maps.append(
            {
                "attn_1to2": weights_1to2,
                "attn_2to1": weights_2to1,
            }
        )

        final_output.append((f1_out, f2_out))

    # Normalize last output (same as model._decoder does)
    del final_output[1]  # duplicate with final_output[0]
    final_output[-1] = tuple(map(model.dec_norm, final_output[-1]))

    # Return same format as model._decoder: zip(*final_output) gives (dec1, dec2)
    dec1, dec2 = zip(*final_output, strict=False)
    return dec1, dec2, attention_maps


def generate_attention_heatmaps(
    images,
    attention_map,
    prompt_coords,
    prompt_token_coords,
    valid_mask,
    layer_idx,
    image_embedding_size=(14, 14),
):
    """
    Generate per-prompt attention heatmaps for a single layer.
    Returns list of PIL images (one per prompt).
    """
    Ht, Wt = image_embedding_size

    # Convert images to numpy
    img1 = images[0].permute(1, 2, 0).cpu().numpy()
    img2 = images[1].permute(1, 2, 0).cpu().numpy()
    img1 = np.clip(img1 * 0.5 + 0.5, 0, 1)
    img2 = np.clip(img2 * 0.5 + 0.5, 0, 1)

    # Get valid prompts (move to CPU and convert to numpy for matplotlib)
    valid1 = valid_mask[0].cpu().numpy()
    valid2 = valid_mask[1].cpu().numpy()
    coords1 = prompt_coords[0][valid1].cpu().numpy()
    coords2 = prompt_coords[1][valid2].cpu().numpy()
    tok_coords1 = prompt_token_coords[0][valid1].cpu().numpy()
    tok_coords2 = prompt_token_coords[1][valid2].cpu().numpy()

    # coords are already in the resized image's pixel space (scaled by the caller).

    n_valid = min(len(coords1), len(coords2))

    if n_valid == 0:
        return []

    # Get attention for this layer (sample_idx=0 since batch size is 1)
    attn_1to2 = attention_map["attn_1to2"][0]  # (N1, N2)

    # Token indices
    tok_idx1 = tok_coords1[:, 1] * Wt + tok_coords1[:, 0]

    # Color map for correspondence pairs
    colors = plt.cm.rainbow(np.linspace(0, 1, n_valid))

    heatmap_images = []

    for i in range(n_valid):
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: View 1 with prompt marker
        ax1.imshow(img1)
        ax1.set_title(f"Prompt {i + 1} in View 1", fontsize=12, fontweight="bold")
        circle = plt.Circle(
            (coords1[i, 0], coords1[i, 1]), 5, color=colors[i], fill=False, linewidth=3
        )
        ax1.add_patch(circle)
        ax1.text(
            coords1[i, 0] + 7,
            coords1[i, 1],
            str(i + 1),
            color=colors[i],
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        ax1.axis("off")

        # Right: Attention heatmap on View 2
        prompt_tok_idx = tok_idx1[i]
        attn_from_prompt = attn_1to2[prompt_tok_idx].reshape(Ht, Wt).cpu().numpy()

        # Overlay attention on grid
        im = ax2.imshow(
            attn_from_prompt, cmap="hot", interpolation="bilinear", alpha=0.9
        )

        # Mark all prompts in view2
        for j in range(n_valid):
            x_t, y_t = tok_coords2[j]
            if i == j:
                # Corresponding prompt - highlight with thick border
                rect = patches.Rectangle(
                    (x_t - 0.45, y_t - 0.45),
                    0.9,
                    0.9,
                    linewidth=4,
                    edgecolor="cyan",
                    facecolor="none",
                )
                ax2.add_patch(rect)
                ax2.text(
                    x_t,
                    y_t - 0.8,
                    "★",
                    color="cyan",
                    fontsize=20,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
            else:
                # Other prompts - thin border
                rect = patches.Rectangle(
                    (x_t - 0.4, y_t - 0.4),
                    0.8,
                    0.8,
                    linewidth=2,
                    edgecolor=colors[j],
                    facecolor="none",
                )
                ax2.add_patch(rect)

        corr_attn_value = attn_from_prompt[tok_coords2[i, 1], tok_coords2[i, 0]]
        ax2.set_title(
            f"Layer {layer_idx}: Attention to View 2\nCorrespondence Attn: {corr_attn_value:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xticks(range(Wt))
        ax2.set_yticks(range(Ht))
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        plt.colorbar(im, ax=ax2, fraction=0.046)

        plt.tight_layout()

        # Render the figure to an RGB PIL image.
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_array = buf[:, :, :3]  # Drop alpha channel, keep RGB
        heatmap_img = Image.fromarray(img_array)
        heatmap_images.append(heatmap_img)

        plt.close(fig)

    return heatmap_images


def create_3d_correspondence_visualization(
    sampled_pts1, sampled_pts2, pair_colors, output_path
):
    """
    Create a 3D visualization of correspondence points.

    Args:
        sampled_pts1: (P, 3) numpy array of 3D points from view 1
        sampled_pts2: (P, 3) numpy array of 3D points from view 2
        pair_colors: List of RGB colors for each pair (length P)
        output_path: Path to save the .glb file

    Returns:
        Path to the saved .glb file
    """
    debug_log(
        f"Creating 3D correspondence visualization with {len(sampled_pts1)} pairs"
    )

    # Create spheres for each correspondence point
    spheres = []
    radius = 0.02  # Scene units

    for i in range(len(sampled_pts1)):
        color = pair_colors[i]

        # Point from view 1
        sphere1 = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere1.apply_translation(sampled_pts1[i])
        sphere1.visual.vertex_colors = color
        spheres.append(sphere1)

        # Point from view 2
        sphere2 = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere2.apply_translation(sampled_pts2[i])
        sphere2.visual.vertex_colors = color
        spheres.append(sphere2)

        # Connect with a line
        line_points = np.array([sampled_pts1[i], sampled_pts2[i]])
        line = trimesh.load_path(line_points.reshape(-1, 3))
        line.colors = [color[:3]] * len(line.entities)  # Use RGB only
        spheres.append(line)

    # Combine all geometries
    if spheres:
        scene = trimesh.Scene(spheres)
        scene.export(output_path)
        debug_log(f"Saved 3D correspondence visualization to {output_path}")
        return output_path
    else:
        debug_log("No correspondence points to visualize")
        return None


def sample_world_points_at_prompts(
    pred1, pred2, prompts1, prompts2, use_simple_scaling=False
):
    """
    Sample world-frame 3D points at prompted locations (for VGGT and Pi3).

    Args:
        pred1, pred2: Predictions with 'pts3d_world' keys
        prompts1, prompts2: Prompt dicts with 'coords' and 'shape' keys
        use_simple_scaling: Whether to use simple scaling for coords

    Returns:
        Tuple of (sampled_pts1_np, sampled_pts2_np) - (P, 3) numpy arrays
    """
    if prompts1 is None or prompts2 is None:
        return None, None

    # Extract world points
    pts3d_world_1 = pred1["pts3d_world"]  # (B, H, W, 3)
    pts3d_world_2 = pred2["pts3d_world"]  # (B, H, W, 3)

    # Resolution of the predicted point maps.
    B1, H1, W1, _ = pts3d_world_1.shape
    B2, H2, W2, _ = pts3d_world_2.shape
    debug_log(
        f"World points shape: view1=[{B1}, {H1}, {W1}, 3], view2=[{B2}, {H2}, {W2}, 3]"
    )

    # Extract prompt info
    coords1 = prompts1["coords"]  # (B, P, 2)
    coords2 = prompts2["coords"]  # (B, P, 2)
    valid_mask = prompts1["valid_mask"]  # (B, P)
    shape1 = prompts1["shape"]  # (B, 2)
    shape2 = prompts2["shape"]

    debug_log(
        f"Prompt coords shape: {coords1.shape}, expected shape from prompts: {shape1}"
    )
    debug_log(
        f"Sample coord from view1: {coords1[0, 0].cpu().numpy() if coords1.shape[1] > 0 else 'none'}"
    )

    # VGGT and Pi3 may predict at a different resolution than the prompt shape;
    # if so, rescale the click coordinates to it.
    actual_shape1 = torch.tensor([[H1, W1]], dtype=torch.float32, device=coords1.device)
    actual_shape2 = torch.tensor([[H2, W2]], dtype=torch.float32, device=coords2.device)

    if not (
        torch.allclose(shape1, actual_shape1) and torch.allclose(shape2, actual_shape2)
    ):
        debug_log(
            f"WARNING: Shape mismatch! Prompts expect [{shape1[0, 0].item():.0f}, {shape1[0, 1].item():.0f}] "
            f"but pts3d_world is [{H1}, {W1}]. Rescaling coordinates..."
        )
        # Rescale coordinates from prompt space to actual pts3d space
        scale_h1 = H1 / shape1[0, 0].item()
        scale_w1 = W1 / shape1[0, 1].item()
        scale_h2 = H2 / shape2[0, 0].item()
        scale_w2 = W2 / shape2[0, 1].item()

        coords1_rescaled = coords1.clone()
        coords1_rescaled[:, :, 0] *= scale_w1  # x coords
        coords1_rescaled[:, :, 1] *= scale_h1  # y coords

        coords2_rescaled = coords2.clone()
        coords2_rescaled[:, :, 0] *= scale_w2
        coords2_rescaled[:, :, 1] *= scale_h2

        debug_log(
            f"Rescaled coords: {coords1_rescaled[0, 0].cpu().numpy() if coords1_rescaled.shape[1] > 0 else 'none'}"
        )

        # Sample using rescaled coords and actual shapes
        sampled_pts1 = sample_3d_at_prompts(
            pts3d_world_1, coords1_rescaled, actual_shape1
        )  # (B, P, 3)
        sampled_pts2 = sample_3d_at_prompts(
            pts3d_world_2, coords2_rescaled, actual_shape2
        )  # (B, P, 3)
    else:
        # Sample 3D points at prompted locations (shapes match, no rescaling needed)
        sampled_pts1 = sample_3d_at_prompts(pts3d_world_1, coords1, shape1)  # (B, P, 3)
        sampled_pts2 = sample_3d_at_prompts(pts3d_world_2, coords2, shape2)  # (B, P, 3)

    # Filter by valid mask and convert to numpy
    if valid_mask.any():
        sampled_pts1_np = to_numpy(sampled_pts1[valid_mask])  # (N, 3)
        sampled_pts2_np = to_numpy(sampled_pts2[valid_mask])  # (N, 3)
        debug_log(f"Sampled {len(sampled_pts1_np)} world-frame correspondence points")
        debug_log(
            f"Sample 3D point from view1: {sampled_pts1_np[0] if len(sampled_pts1_np) > 0 else 'none'}"
        )
        return sampled_pts1_np, sampled_pts2_np
    else:
        return None, None


def scene_clouds(scene):
    """World-frame vertices of the reconstructed points or mesh, by geometry name."""
    clouds = {}
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geometry = scene.geometry[name]
        if isinstance(geometry, trimesh.PointCloud) or len(geometry.vertices) > 5000:
            clouds[name] = trimesh.transform_points(
                np.asarray(geometry.vertices), transform
            )
    return clouds


def inlier_bounds(points):
    """Median point and the distance beyond which display points count as outliers."""
    center = np.median(points, axis=0)
    return center, 2 * np.percentile(np.linalg.norm(points - center, axis=1), 90)


def click_marker_radius(scene):
    """Marker radius as a fixed share of the scene size the viewer frames.

    Sizing by local point density would enlarge markers wherever a model
    reconstructs the clicked surface sparsely, so views and examples would differ.
    """
    points = np.concatenate(list(scene_clouds(scene).values()))
    center, limit = inlier_bounds(points)
    distance = np.linalg.norm(points - center, axis=1)
    return CLICK_MARKER_SCALE * np.percentile(distance[distance <= limit], 75)


def frame_model_for_viewer(model_path):
    """Drop far outliers and embed the initial camera orbit in the display GLB.

    The viewer aims at the center of the model's bounding box from 1.1 times its
    diagonal, so a few far points leave the scene small and off center. Removing
    them recenters the view, and the radius brings about 75% of the points into it.
    """
    if not model_path:
        return model_path
    scene = trimesh.load(model_path)
    clouds = scene_clouds(scene)
    center, limit = inlier_bounds(np.concatenate(list(clouds.values())))
    kept = []
    for name, vertices in clouds.items():
        keep = np.linalg.norm(vertices - center, axis=1) <= limit
        geometry = scene.geometry[name]
        if isinstance(geometry, trimesh.PointCloud):
            scene.geometry[name] = trimesh.PointCloud(
                np.asarray(geometry.vertices)[keep],
                colors=np.asarray(geometry.colors)[keep],
            )
        else:
            geometry.update_vertices(keep)
        kept.append(vertices[keep])
    target = scene.bounds.mean(axis=0)
    radius = np.percentile(
        np.linalg.norm(np.concatenate(kept) - target, axis=1), 75
    ) / np.tan(VIEWER_FOV / 2)
    # GLB scene extras survive Gradio's file-cache normalization. Custom fields
    # added to FileData itself are discarded by that normalization.
    scene.metadata["camera_position"] = [None, None, float(radius)]
    scene.export(model_path)
    return model_path


def overlay_correspondences_on_model(
    model_path, sampled_pts1, sampled_pts2, pair_colors, output_path
):
    """
    Overlay correspondence points on an existing 3D model.

    Args:
        model_path: Path to the base .glb model
        sampled_pts1: (P, 3) numpy array of 3D points from view 1
        sampled_pts2: (P, 3) numpy array of 3D points from view 2
        pair_colors: List of RGBA colors for each pair (length P)
        output_path: Path to save the overlaid .glb file

    Returns:
        Path to the saved .glb file with overlaid correspondences
    """
    if model_path is None or not os.path.exists(model_path):
        debug_log(f"Base model not found at {model_path}, skipping overlay")
        return None

    debug_log(f"Overlaying correspondences on model at {model_path}")

    try:
        base_scene = trimesh.load(model_path)

        radius = click_marker_radius(
            base_scene
            if isinstance(base_scene, trimesh.Scene)
            else trimesh.Scene(base_scene)
        )

        debug_log(f"Sphere radius: {radius:.4f}")

        # Create correspondence geometries
        corr_geometries = []
        for i in range(len(sampled_pts1)):
            color = pair_colors[i]

            # Point from view 1
            sphere1 = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere1.apply_translation(sampled_pts1[i])
            sphere1.visual.vertex_colors = color
            corr_geometries.append(sphere1)

            # Point from view 2
            sphere2 = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            sphere2.apply_translation(sampled_pts2[i])
            sphere2.visual.vertex_colors = color
            corr_geometries.append(sphere2)

            # Connect the pair with a cylinder, thicker than a line.
            direction = sampled_pts2[i] - sampled_pts1[i]
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                cylinder = trimesh.creation.cylinder(radius=radius * 0.3, height=length)
                # Rotate cylinder to align with direction
                z_axis = np.array([0, 0, 1])
                rotation_axis = np.cross(z_axis, direction)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        angle, rotation_axis
                    )
                    cylinder.apply_transform(rotation_matrix)
                # Translate to midpoint
                midpoint = (sampled_pts1[i] + sampled_pts2[i]) / 2
                cylinder.apply_translation(midpoint)
                cylinder.visual.vertex_colors = color
                corr_geometries.append(cylinder)

        # Combine base model with correspondences
        if isinstance(base_scene, trimesh.Scene):
            # Add correspondence geometries to existing scene
            for idx, geom in enumerate(corr_geometries):
                base_scene.add_geometry(geom, node_name=f"correspondence_{idx}")
            combined_scene = base_scene
        else:
            # Create new scene with base model and correspondences
            geometries = [base_scene] + corr_geometries
            combined_scene = trimesh.Scene(geometries)

        combined_scene.export(output_path)
        debug_log(f"Saved overlaid model to {output_path}")
        return output_path

    except Exception as e:
        debug_log(f"Error overlaying correspondences: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return None


def create_correspondence_prompts_from_clicks(
    correspondence_pairs, img1_shape, img2_shape, device="cuda"
):
    """
    Create sparse correspondence prompts from user clicks.

    Click coordinates are mapped from the input image to DUSt3R's 224×224 input
    with the resize and center crop of load_images, as in training.

    Args:
        correspondence_pairs: List of ((x1, y1), (x2, y2)) in normalized coords [0, 1]
        img1_shape: (H, W) of input image 1
        img2_shape: (H, W) of input image 2
        device: torch device

    Returns:
        Tuple of (prompts_view1, prompts_view2) - separate prompts for each view with coords in 224×224 space
    """
    if len(correspondence_pairs) == 0:
        return None, None

    H1, W1 = img1_shape
    H2, W2 = img2_shape
    P = len(correspondence_pairs)

    # Convert normalized coords to pixel coords in ORIGINAL image space
    coords1_orig = []
    coords2_orig = []
    for (x1_norm, y1_norm), (x2_norm, y2_norm) in correspondence_pairs:
        coords1_orig.append([x1_norm * W1, y1_norm * H1])
        coords2_orig.append([x2_norm * W2, y2_norm * H2])

    coords1_orig = torch.tensor(
        coords1_orig, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, P, 2)
    coords2_orig = torch.tensor(
        coords2_orig, dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, P, 2)

    # Original image shapes
    orig_shape1 = torch.tensor([[H1, W1]], dtype=torch.float32, device=device)  # (1, 2)
    orig_shape2 = torch.tensor([[H2, W2]], dtype=torch.float32, device=device)  # (1, 2)

    # Map clicks to DUSt3R's 224×224 input (resize + center crop), the prompt space used in training.
    coords1_224 = transform_coords_to_dust3r_space(
        coords1_orig, orig_shape1, size=224, verbose=False, use_simple_scaling=False
    )
    coords2_224 = transform_coords_to_dust3r_space(
        coords2_orig, orig_shape2, size=224, verbose=False, use_simple_scaling=False
    )

    debug_log("Coordinate transformation for prompts:")
    debug_log(
        f"  View1: {img1_shape} -> 224×224, sample orig: {coords1_orig[0, 0].cpu().numpy()}, transformed: {coords1_224[0, 0].cpu().numpy()}"
    )
    debug_log(
        f"  View2: {img2_shape} -> 224×224, sample orig: {coords2_orig[0, 0].cpu().numpy()}, transformed: {coords2_224[0, 0].cpu().numpy()}"
    )

    # Create pair IDs (same for both views to link correspondences)
    pair_ids = torch.arange(P, dtype=torch.long, device=device).unsqueeze(0)  # (1, P)

    # Create valid mask - mark points outside the 224×224 crop as invalid
    valid_mask1 = (
        (coords1_224[..., 0] >= 0)
        & (coords1_224[..., 0] < 224)
        & (coords1_224[..., 1] >= 0)
        & (coords1_224[..., 1] < 224)
    )
    valid_mask2 = (
        (coords2_224[..., 0] >= 0)
        & (coords2_224[..., 0] < 224)
        & (coords2_224[..., 1] >= 0)
        & (coords2_224[..., 1] < 224)
    )
    # Both points in a pair must be valid
    valid_mask = valid_mask1 & valid_mask2

    num_invalid = (~valid_mask).sum().item()
    if num_invalid > 0:
        debug_log(
            f"  WARNING: {num_invalid}/{P} correspondence pairs fall outside valid crop region"
        )

    # Separate prompts for each view
    # The prompt shape is the 224×224 model input, as in training.
    prompts_view1 = {
        "coords": coords1_224,  # (1, P, 2) - transformed to 224×224 space
        "image_ids": torch.zeros(
            1, P, dtype=torch.long, device=device
        ),  # all 0s for view1
        "pair_ids": pair_ids,
        "shape": torch.tensor(
            [[224, 224]], dtype=torch.float32, device=device
        ),  # (1, 2), always 224×224
        "valid_mask": valid_mask,
    }

    prompts_view2 = {
        "coords": coords2_224,  # (1, P, 2) - transformed to 224×224 space
        "image_ids": torch.ones(
            1, P, dtype=torch.long, device=device
        ),  # all 1s for view2
        "pair_ids": pair_ids,
        "shape": torch.tensor(
            [[224, 224]], dtype=torch.float32, device=device
        ),  # (1, 2), always 224×224
        "valid_mask": valid_mask,
    }

    debug_log(f"Created prompts for {P} pairs ({valid_mask.sum().item()} valid)")
    debug_log(f"  View1: coords {coords1_224.shape}, shape {prompts_view1['shape']}")
    debug_log(f"  View2: coords {coords2_224.shape}, shape {prompts_view2['shape']}")

    return prompts_view1, prompts_view2


# Define colors for correspondence pairs (same color for both points in a pair)
PAIR_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "magenta",
    "navy",
]


def prompts_for_view_role(prompts, role):
    """Image IDs describe the two model inputs, including reversed image pairs."""
    if prompts is None:
        return None
    result = dict(prompts)
    result["image_ids"] = torch.full_like(prompts["image_ids"], role)
    return result


def get_pair_color(pair_index):
    """Get color for a correspondence pair."""
    return PAIR_COLORS[pair_index % len(PAIR_COLORS)]


def get_pair_color_rgba(pair_index):
    """Get RGBA color for a correspondence pair (for 3D visualization)."""
    color_name = get_pair_color(pair_index)
    # Convert matplotlib color names to RGBA
    from matplotlib.colors import to_rgba

    rgba = to_rgba(color_name)
    return [int(c * 255) for c in rgba]


def display_size(image):
    """Size an image is shown at; clicks are stored as normalized coordinates."""
    scale = DISPLAY_LONG_SIDE / max(image.size)
    return round(image.width * scale), round(image.height * scale)


def prepare_display_image(image):
    """Resize an input for clicking without adding a crop border or shading."""
    if image is None:
        return None
    return image.resize(display_size(image), Image.LANCZOS)


def refresh_display_images():
    """Resize the current pair for display once, when it changes; clicks reuse the result."""
    _session()["display_images"] = [
        prepare_display_image(image) for image in _session()["current_image_pair"]
    ]
    return _session()["display_images"]


def display_images():
    """The current pair at display size, cached because resizing two photos takes ~130 ms per click."""
    cached = _session().get("display_images") or [None, None]
    if any(
        shown is None and image is not None
        for shown, image in zip(cached, _session()["current_image_pair"])
    ):
        return refresh_display_images()
    return cached


def add_click_marker(image, x_norm, y_norm, color="red", radius=8, pair_id=None):
    """Add a visible click marker to an image."""
    if image is None:
        debug_log("add_click_marker: No image provided")
        return None

    try:
        # Create a copy and draw the click point
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Convert normalized coordinates to pixel coordinates
        width, height = img_copy.size
        pixel_x = int(x_norm * width)
        pixel_y = int(y_norm * height)

        debug_log(
            f"Adding {color} marker (pair {pair_id}) at ({x_norm:.3f}, {y_norm:.3f}) -> ({pixel_x}, {pixel_y})"
        )

        # Draw a circle at click location
        draw.ellipse(
            [pixel_x - radius, pixel_y - radius, pixel_x + radius, pixel_y + radius],
            outline=color,
            fill=None,
            width=4,
        )

        # Draw crosshairs
        draw.line(
            [pixel_x - radius * 2, pixel_y, pixel_x + radius * 2, pixel_y],
            fill=color,
            width=3,
        )
        draw.line(
            [pixel_x, pixel_y - radius * 2, pixel_x, pixel_y + radius * 2],
            fill=color,
            width=3,
        )

        # Draw a small filled center dot
        draw.ellipse([pixel_x - 2, pixel_y - 2, pixel_x + 2, pixel_y + 2], fill=color)

        # Add pair number if provided
        if pair_id is not None:
            draw.text(
                (pixel_x + radius + 2, pixel_y - radius), str(pair_id), fill=color
            )

        return img_copy
    except Exception as e:
        debug_log(f"Error in add_click_marker: {e}")
        return image


def read_input_image(path):
    """The central square of an input, the only part the 224 models see.

    DUSt3R's load_images applies the EXIF orientation of phone photos and center
    crops to a square; cropping here shows users exactly what they can click on.
    """
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def open_input_image(path):
    """An input image, from memory for the bundled examples."""
    if path in EXAMPLE_IMAGES:
        return EXAMPLE_IMAGES[path].copy()
    return read_input_image(path)


def update_images_display(files):
    """Update the displayed images when files are uploaded."""

    debug_log(f"update_images_display called with {len(files) if files else 0} files")

    _session()["uploaded_images"] = []
    _session()["current_image_pair"] = [None, None]
    _session()["display_images"] = [None, None]
    _session()["correspondence_points"] = []
    _session()["click_count"] = 0
    _session()["image_augmented"] = False

    if not files or len(files) != 2:
        return None, None, "Please upload exactly two images."

    try:
        _session()["uploaded_images"] = [
            open_input_image(getattr(f, "name", f)) for f in files[:2]
        ]
        _session()["current_image_pair"] = [
            _session()["uploaded_images"][0],
            _session()["uploaded_images"][1],
        ]
        _session()["correspondence_points"] = []
        _session()["image_augmented"] = False

        debug_log(
            f"Loaded {len(_session().uploaded_images)} images, sizes: {[img.size for img in _session().uploaded_images]}"
        )

        img1_display, img2_display = refresh_display_images()

        return (
            img1_display,
            img2_display,
            "Click matching points in both images.",
        )
    except Exception as e:
        debug_log(f"Error loading images: {e}")
        return None, None, f"Error loading images: {e}"


def augment_second_image():
    """Toggle a 180-degree rotation of the second image."""

    if _session()["current_image_pair"][1] is None:
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            "No images loaded.",
        )

    try:
        if not _session()["image_augmented"]:
            # Rotate 180 degrees
            _session()["current_image_pair"][1] = _session()["uploaded_images"][
                1
            ].rotate(180)
            _session()["image_augmented"] = True
            status = "✓ Second image flipped 180°"
        else:
            # Restore original
            _session()["current_image_pair"][1] = _session()["uploaded_images"][1]
            _session()["image_augmented"] = False
            status = "✓ Second image restored to original"

        # Clear correspondences when augmenting
        _session()["correspondence_points"] = []

        debug_log(f"Augmentation toggled: {_session().image_augmented}")

        img1_display, img2_display = refresh_display_images()

        return (
            img1_display,
            img2_display,
            f"{status}\nCorrespondences cleared. Click new points.",
        )
    except Exception as e:
        debug_log(f"Error in augment_second_image: {e}")
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            f"Error: {e}",
        )


def add_correspondence_point(x_norm, y_norm, image_idx, changed_only=False):
    """Add a correspondence point and update displays.

    changed_only: return gr.skip() for an image whose markers did not change, so a click sends one image back
    unless it completes a pair (which colours the matching marker in the other image).
    """

    if changed_only:
        # A double click lands twice on the same spot; keep only the first.
        last = (_session()["correspondence_points"] or [None])[-1]
        now = time.monotonic()
        repeated = (
            last is not None
            and last["image_idx"] == image_idx
            and abs(last["x"] - x_norm) < 0.012
            and abs(last["y"] - y_norm) < 0.012
            and now - _session().get("last_click_time", 0.0) < 0.5
        )
        _session()["last_click_time"] = now
        if repeated:
            return gr.skip(), gr.skip(), gr.skip()

    _session()["click_count"] += 1
    debug_log(
        f"add_correspondence_point: click #{_session().click_count} at ({x_norm:.3f}, {y_norm:.3f}) on image {image_idx}"
    )

    if (
        _session()["current_image_pair"][0] is None
        or _session()["current_image_pair"][1] is None
    ):
        debug_log("No images loaded")
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            "No images loaded.",
            None,
        )

    try:
        # Add point to correspondence list
        point_info = {
            "x": x_norm,
            "y": y_norm,
            "image_idx": image_idx,
            "click_id": _session()["click_count"],
        }
        _session()["correspondence_points"].append(point_info)
        debug_log(f"Added point: {point_info}")

        # Separate points by image
        img1_points = [
            p for p in _session()["correspondence_points"] if p["image_idx"] == 0
        ]
        img2_points = [
            p for p in _session()["correspondence_points"] if p["image_idx"] == 1
        ]

        debug_log(
            f"Image 0 points: {len(img1_points)}, Image 1 points: {len(img2_points)}"
        )

        # Create correspondence pairs
        pairs = []
        for i in range(min(len(img1_points), len(img2_points))):
            p1 = img1_points[i]
            p2 = img2_points[i]
            pairs.append(((p1["x"], p1["y"]), (p2["x"], p2["y"])))

        # Create marked versions of images
        img1_marked, img2_marked = display_images()
        completes_pair = len(pairs) > min(
            len(img1_points) - (image_idx == 0), len(img2_points) - (image_idx == 1)
        )

        # Add markers to image 1 - each pair gets same color
        for i, point in enumerate(img1_points):
            pair_id = i + 1 if i < len(pairs) else None
            color = get_pair_color(i) if i < len(pairs) else "lightgray"
            img1_marked = add_click_marker(
                img1_marked,
                point["x"],
                point["y"],
                color=color,
                radius=8,
                pair_id=pair_id,
            )

        # Add markers to image 2 - matching colors from same pairs
        for i, point in enumerate(img2_points):
            pair_id = i + 1 if i < len(pairs) else None
            color = get_pair_color(i) if i < len(pairs) else "lightgray"
            img2_marked = add_click_marker(
                img2_marked,
                point["x"],
                point["y"],
                color=color,
                radius=8,
                pair_id=pair_id,
            )

        # Generate status text
        aug_status = " [FLIPPED 180°]" if _session()["image_augmented"] else ""
        status_text = "Matching colors mark correspondence pairs\n"
        status_text += f"Image 2 status: {aug_status}\n"
        status_text += f"Total clicks: {len(_session().correspondence_points)}\n"
        status_text += f"Image 1 points: {len(img1_points)}\n"
        status_text += f"Image 2 points: {len(img2_points)}\n"
        status_text += f"Correspondence pairs: {len(pairs)}\n\n"

        for i, ((x1, y1), (x2, y2)) in enumerate(pairs):
            color = get_pair_color(i)
            status_text += (
                f"Pair {i + 1} ({color}): ({x1:.3f},{y1:.3f}) ↔ ({x2:.3f},{y2:.3f})\n"
            )

        # Show unpaired points
        if len(img1_points) > len(pairs):
            status_text += f"\nUnpaired Image 1 points: {len(img1_points) - len(pairs)}"
        if len(img2_points) > len(pairs):
            status_text += f"\nUnpaired Image 2 points: {len(img2_points) - len(pairs)}"

        if len(_session()["correspondence_points"]) > 0:
            status_text += (
                f"\nLast click: #{_session().click_count} on image {image_idx + 1}"
            )

        debug_log(
            f"Created {len(pairs)} pairs with colors: {[get_pair_color(i) for i in range(len(pairs))]}"
        )

        if changed_only and not completes_pair:
            if image_idx == 0:
                img2_marked = gr.skip()
            else:
                img1_marked = gr.skip()
        return img1_marked, img2_marked, status_text

    except Exception as e:
        debug_log(f"Error in add_correspondence_point: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            f"Error adding point: {e}",
        )


def handle_img1_click(evt: gr.SelectData):
    """Handle clicks on image 1."""
    debug_log(f"handle_img1_click called with evt: {evt}")
    if evt is None:
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            "No click data received",
        )

    try:
        # Extract click coordinates from event
        if hasattr(evt, "index") and evt.index is not None:
            if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
                x_pixel, y_pixel = evt.index[0], evt.index[1]
            else:
                debug_log(f"Invalid index format: {evt.index}")
                return (
                    _session()["current_image_pair"][0],
                    _session()["current_image_pair"][1],
                    "Invalid click format",
                )
        else:
            debug_log(f"No index in event: {vars(evt)}")
            return (
                _session()["current_image_pair"][0],
                _session()["current_image_pair"][1],
                "No click coordinates found",
            )

        # Convert to normalized coordinates
        if _session()["current_image_pair"][0]:
            width, height = display_size(_session()["current_image_pair"][0])
            x_norm = x_pixel / width
            y_norm = y_pixel / height
            debug_log(
                f"Image 1 click: ({x_pixel}, {y_pixel}) -> ({x_norm:.3f}, {y_norm:.3f})"
            )
            return add_correspondence_point(x_norm, y_norm, 0, changed_only=True)
        else:
            return (
                _session()["current_image_pair"][0],
                _session()["current_image_pair"][1],
                "No image loaded",
            )

    except Exception as e:
        debug_log(f"Error in handle_img1_click: {e}")
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            f"Click error: {e}",
        )


def handle_img2_click(evt: gr.SelectData):
    """Handle clicks on image 2."""
    debug_log(f"handle_img2_click called with evt: {evt}")
    if evt is None:
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            "No click data received",
        )

    try:
        # Extract click coordinates from event
        if hasattr(evt, "index") and evt.index is not None:
            if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 2:
                x_pixel, y_pixel = evt.index[0], evt.index[1]
            else:
                debug_log(f"Invalid index format: {evt.index}")
                return (
                    _session()["current_image_pair"][0],
                    _session()["current_image_pair"][1],
                    "Invalid click format",
                )
        else:
            debug_log(f"No index in event: {vars(evt)}")
            return (
                _session()["current_image_pair"][0],
                _session()["current_image_pair"][1],
                "No click coordinates found",
            )

        # Convert to normalized coordinates
        if _session()["current_image_pair"][1]:
            width, height = display_size(_session()["current_image_pair"][1])
            x_norm = x_pixel / width
            y_norm = y_pixel / height
            debug_log(
                f"Image 2 click: ({x_pixel}, {y_pixel}) -> ({x_norm:.3f}, {y_norm:.3f})"
            )
            return add_correspondence_point(x_norm, y_norm, 1, changed_only=True)
        else:
            return (
                _session()["current_image_pair"][0],
                _session()["current_image_pair"][1],
                "No image loaded",
            )

    except Exception as e:
        debug_log(f"Error in handle_img2_click: {e}")
        return (
            _session()["current_image_pair"][0],
            _session()["current_image_pair"][1],
            f"Click error: {e}",
        )


def clear_correspondences():
    """Clear all correspondence points."""

    debug_log("clear_correspondences called")
    _session()["correspondence_points"] = []
    _session()["click_count"] = 0

    if _session()["current_image_pair"][0] is not None:
        aug_status = " [FLIPPED 180°]" if _session()["image_augmented"] else ""
        debug_log("Cleared correspondences, returning unmarked images")

        img1_display, img2_display = display_images()

        return (
            img1_display,
            img2_display,
            f"Cleared all correspondence points.\nImage 2: {aug_status}",
        )
    else:
        return None, None, "No images loaded."


def run_reconstruction(
    min_conf_thr,
    cam_size,
    as_pointcloud,
    show_original,
    show_trained,
    show_overlap_gt,
    show_vggt,
    show_pi3,
    show_attention,
    attention_layer,
    show_corr_overlay,
    use_simple_scaling,
    model,
    device,
    output_dir,
    img_size,
    original_model=None,
    trained_vanilla_model=None,
    overlap_gt_model=None,
    vggt_model=None,
    pi3_model=None,
) -> ReconstructionResult:
    """Run the reconstruction with current correspondences, comparing multiple models."""

    debug_log(
        f"run_reconstruction called with {len(_session().correspondence_points)} points"
    )
    debug_log(f"Model expects img_size: {img_size}")
    debug_log(f"Show original DUSt3R: {show_original}")
    debug_log(f"Show trained vanilla DUSt3R: {show_trained}")
    debug_log(f"Show VGGT: {show_vggt}")
    debug_log(f"Show Pi3: {show_pi3}")
    debug_log(f"Show attention: {show_attention}, Layer: {attention_layer}")
    debug_log(f"Show correspondence overlay: {show_corr_overlay}")
    debug_log(f"Image 2 augmented (180° flip): {_session().image_augmented}")

    if not _session()["uploaded_images"] or len(_session()["uploaded_images"]) < 2:
        return ReconstructionResult(
            None, None, None, None, None, None, "Upload an image pair first."
        )

    # Group points by image
    img1_points = [
        p for p in _session()["correspondence_points"] if p["image_idx"] == 0
    ]
    img2_points = [
        p for p in _session()["correspondence_points"] if p["image_idx"] == 1
    ]

    if len(img1_points) != len(img2_points):
        return ReconstructionResult(
            None,
            None,
            None,
            None,
            None,
            None,
            "Complete each correspondence in both images before reconstructing.",
        )

    try:
        debug_log("Starting reconstruction process...")

        # Save images to temporary files with proper file objects
        # Use current_image_pair instead of uploaded_images to capture augmentation
        temp_files = []
        temp_paths = []

        for i, img in enumerate(
            _session()["current_image_pair"][:2]
        ):
            os.makedirs(output_dir, exist_ok=True)
            temp_path = os.path.join(output_dir, f"input_{i}_{uuid.uuid4().hex}.png")
            # Fastest zlib level: same pixels, ~3x faster for full-resolution photos (1.1 s -> 0.4 s each).
            img.save(temp_path, "PNG", compress_level=1)
            temp_paths.append(temp_path)

            # Object with a .name attribute, like an uploaded file.
            class TempFile:
                def __init__(self, path):
                    self.name = path

            temp_files.append(TempFile(temp_path))

        debug_log(f"Saved temp images: {[f.name for f in temp_files]}")

        # Load images using DUSt3R pipeline - use model's expected resolution
        # img_size is a tuple (H, W), but load_images expects a single int (will make square)
        resolution = img_size[0] if isinstance(img_size, (tuple, list)) else img_size
        debug_log(f"Loading images at resolution {resolution} to match model")

        image_paths = [f.name for f in temp_files]
        loaded_images = load_images(image_paths, size=resolution)
        image_pairs = make_pairs(loaded_images, scene_graph="complete", symmetrize=True)

        debug_log(f"Loaded {len(loaded_images)} images, {len(image_pairs)} pairs")
        debug_log(
            f"Pairs structure: {[(type(p), len(p) if hasattr(p, '__len__') else 'no len') for p in image_pairs]}"
        )

        # === Prompted reconstruction ===
        # Create correspondence pairs
        correspondence_pairs = []
        for i in range(min(len(img1_points), len(img2_points))):
            p1 = img1_points[i]
            p2 = img2_points[i]
            correspondence_pairs.append(((p1["x"], p1["y"]), (p2["x"], p2["y"])))

        debug_log(f"Created {len(correspondence_pairs)} correspondence pairs")
        for i, ((x1, y1), (x2, y2)) in enumerate(
            correspondence_pairs[:5]
        ):  # Show first 5
            debug_log(
                f"  Pair {i}: img1({x1:.4f}, {y1:.4f}) <-> img2({x2:.4f}, {y2:.4f})"
            )

        # Create correspondence prompts (separate for each view)
        # Shapes of the current pair, whose second image may be rotated.
        img1_shape = (
            _session()["current_image_pair"][0].size[1],
            _session()["current_image_pair"][0].size[0],
        )  # (H, W) from PIL
        img2_shape = (
            _session()["current_image_pair"][1].size[1],
            _session()["current_image_pair"][1].size[0],
        )  # (H, W) from PIL

        debug_log(f"Current image shapes for prompts: {img1_shape}, {img2_shape}")
        debug_log(f"Dust3r loaded_images[0] keys: {loaded_images[0].keys()}")
        debug_log(
            f"Dust3r loaded_images[0]['img'] shape: {loaded_images[0]['img'].shape}"
        )
        debug_log(
            f"Dust3r loaded_images[0]['true_shape']: {loaded_images[0]['true_shape']}"
        )
        debug_log(
            f"Dust3r loaded_images[1]['true_shape']: {loaded_images[1]['true_shape']}"
        )

        prompts_view1, prompts_view2 = create_correspondence_prompts_from_clicks(
            correspondence_pairs, img1_shape, img2_shape, device=device
        )

        debug_log("Created correspondence prompts for both views")
        if prompts_view1 is not None:
            debug_log(
                f"  Prompts view1 coords shape: {prompts_view1['coords'].shape}, sample: {prompts_view1['coords'][0, :3] if prompts_view1['coords'].shape[1] >= 3 else prompts_view1['coords'][0]}"
            )
        if prompts_view2 is not None:
            debug_log(
                f"  Prompts view2 coords shape: {prompts_view2['coords'].shape}, sample: {prompts_view2['coords'][0, :3] if prompts_view2['coords'].shape[1] >= 3 else prompts_view2['coords'][0]}"
            )

        # Run prompted reconstruction
        attention_maps_storage = None
        with torch.no_grad():
            results_with_prompts = {}
            for view1_dict, view2_dict in image_pairs:
                # pairs contains actual image dictionaries, not indices
                view1 = prepare_model_view(view1_dict, device)
                view2 = prepare_model_view(view2_dict, device)

                # Attach each image's prompts by its index: make_pairs with symmetrize=True
                # yields both (img1, img0) and (img0, img1).
                idx1 = view1_dict["idx"]
                idx2 = view2_dict["idx"]

                if idx1 == 0:
                    view1["sparse_prompts"] = prompts_view1
                    debug_log(f"  view1 (idx={idx1}) <- prompts_view1 (from img[0])")
                elif idx1 == 1:
                    view1["sparse_prompts"] = prompts_view2
                    debug_log(f"  view1 (idx={idx1}) <- prompts_view2 (from img[1])")

                if idx2 == 0:
                    view2["sparse_prompts"] = prompts_view1
                    debug_log(f"  view2 (idx={idx2}) <- prompts_view1 (from img[0])")
                elif idx2 == 1:
                    view2["sparse_prompts"] = prompts_view2
                    debug_log(f"  view2 (idx={idx2}) <- prompts_view2 (from img[1])")

                view1["sparse_prompts"] = prompts_for_view_role(
                    view1["sparse_prompts"], 0
                )
                view2["sparse_prompts"] = prompts_for_view_role(
                    view2["sparse_prompts"], 1
                )
                debug_log(
                    f"Calling model for prompted: view1 keys={list(view1.keys())}, view2 keys={list(view2.keys())}"
                )

                # If attention visualization is enabled, manually run through encoder and decoder
                if show_attention:
                    debug_log("Running with attention capture enabled")

                    # Run encoder - returns 3 tuples: (shapes), (features), (positions)
                    (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                        model._encode_symmetrized(view1, view2)
                    )

                    # Get prompts from views
                    prompts1 = view1.get("sparse_prompts", None)
                    prompts2 = view2.get("sparse_prompts", None)

                    # Project encoder features to decoder dimension
                    f1 = model.decoder_embed(feat1)
                    f2 = model.decoder_embed(feat2)

                    # Condition decoder inputs on the correspondence prompts.
                    if getattr(model, "prompts_enabled", False):
                        if (
                            hasattr(model, "correspondence_adapter1")
                            and prompts1 is not None
                        ):
                            f1 = model.correspondence_adapter1(
                                f1,
                                coords_xy=prompts1.get("coords"),
                                image_ids=prompts1.get("image_ids"),
                                pair_ids=prompts1.get("pair_ids"),
                                image_sizes=prompts1.get("shape"),
                                token_positions=pos1,
                                valid_mask=prompts1.get("valid_mask", None),
                            )
                        if (
                            hasattr(model, "correspondence_adapter2")
                            and prompts2 is not None
                        ):
                            f2 = model.correspondence_adapter2(
                                f2,
                                coords_xy=prompts2.get("coords"),
                                image_ids=prompts2.get("image_ids"),
                                pair_ids=prompts2.get("pair_ids"),
                                image_sizes=prompts2.get("shape"),
                                token_positions=pos2,
                                valid_mask=prompts2.get("valid_mask", None),
                            )

                    # Decode once, returning both features and attention maps.
                    dec1, dec2, attention_maps = run_decoder_with_attention(
                        model, f1, f2, pos1, pos2
                    )
                    attention_maps_storage = attention_maps
                    debug_log(
                        f"Captured attention from {len(attention_maps)} decoder layers"
                    )

                    # Run heads to get final predictions
                    # _downstream_head expects: (head_num, decout_list, img_shape)
                    with torch.cuda.amp.autocast(enabled=False):
                        pred1 = model._downstream_head(
                            1, [tok.float() for tok in dec1], shape1
                        )
                        pred2 = model._downstream_head(
                            2, [tok.float() for tok in dec2], shape2
                        )

                    # Rename pts3d key for view2 (same as model forward does)
                    pred2["pts3d_in_other_view"] = pred2.pop("pts3d")
                else:
                    # Standard forward pass
                    pred1, pred2 = model(view1, view2)

                # Use the 'idx' field from each image dict as the key
                idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                results_with_prompts[idx1, idx2] = {
                    "pred1": pred1,
                    "pred2": pred2,
                    "view1": view1,
                    "view2": view2,
                }

        debug_log("Prompted reconstruction completed")

        # === Optional overlap_gt_model: Click3R trained without occluded GT (if enabled) ===
        results_overlap_gt = {}
        scene_overlap_gt = None
        model_overlap_gt = None
        if show_overlap_gt and overlap_gt_model is not None:
            debug_log("Running Click3R (Trained Without Occluded GT) with prompts...")
            with torch.no_grad():
                for view1_dict, view2_dict in image_pairs:
                    view1 = prepare_model_view(view1_dict, device)
                    view2 = prepare_model_view(view2_dict, device)

                    # Attach prompts to correct view based on image idx
                    idx1 = view1_dict["idx"]
                    idx2 = view2_dict["idx"]

                    if idx1 == 0:
                        view1["sparse_prompts"] = prompts_view1
                    elif idx1 == 1:
                        view1["sparse_prompts"] = prompts_view2

                    if idx2 == 0:
                        view2["sparse_prompts"] = prompts_view1
                    elif idx2 == 1:
                        view2["sparse_prompts"] = prompts_view2

                    # Standard forward pass with prompts
                    view1["sparse_prompts"] = prompts_for_view_role(
                        view1["sparse_prompts"], 0
                    )
                    view2["sparse_prompts"] = prompts_for_view_role(
                        view2["sparse_prompts"], 1
                    )
                    pred1, pred2 = overlap_gt_model(view1, view2)
                    idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                    results_overlap_gt[idx1, idx2] = {
                        "pred1": pred1,
                        "pred2": pred2,
                        "view1": view1,
                        "view2": view2,
                    }
            debug_log("Click3R (Trained Without Occluded GT) reconstruction completed")

        # === VGGT reconstruction (if enabled) ===
        results_vggt = {}
        model_vggt = None
        if show_vggt and vggt_model is not None:
            debug_log("Running VGGT reconstruction...")
            with torch.no_grad():
                for view1_dict, view2_dict in image_pairs:
                    view1 = prepare_model_view(view1_dict, device)
                    view2 = prepare_model_view(view2_dict, device)

                    # Run VGGT (no prompts support)
                    pred1, pred2 = vggt_model(view1, view2)
                    idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                    results_vggt[idx1, idx2] = {
                        "pred1": pred1,
                        "pred2": pred2,
                        "view1": view1,
                        "view2": view2,
                    }
            debug_log("VGGT reconstruction completed")

        # === Pi3 reconstruction (if enabled) ===
        results_pi3 = {}
        model_pi3 = None
        if show_pi3 and pi3_model is not None:
            debug_log("Running Pi3 reconstruction...")
            with torch.no_grad():
                for view1_dict, view2_dict in image_pairs:
                    view1 = prepare_model_view(view1_dict, device)
                    view2 = prepare_model_view(view2_dict, device)

                    # Run Pi3 (no prompts support)
                    pred1, pred2 = pi3_model(view1, view2)
                    idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                    results_pi3[idx1, idx2] = {
                        "pred1": pred1,
                        "pred2": pred2,
                        "view1": view1,
                        "view2": view2,
                    }
            debug_log("Pi3 reconstruction completed")

        # === Compute metrics for prompted reconstruction ===
        comparison_pair_key = (0, 1) if (0, 1) in results_with_prompts else (1, 0)
        prompted_metrics = compute_correspondence_metrics(
            results_with_prompts[comparison_pair_key]["pred1"],
            results_with_prompts[comparison_pair_key]["pred2"],
            prompts_view1,
            prompts_view2,
            norm_mode="avg_dis",
        )
        debug_log(f"Prompted metrics: {prompted_metrics}")

        # === 3D points at the clicks: extracted after global alignment, in the aligned frame ===
        debug_log(
            "Deferring correspondence point extraction until after global alignment..."
        )

        # === Original DUSt3R reconstruction (if enabled) ===
        model_original = None
        scene_original = None
        if show_original and original_model is not None:
            debug_log("Running original DUSt3R reconstruction...")
            with torch.no_grad():
                results_original = {}
                for view1_dict, view2_dict in image_pairs:
                    view1 = prepare_model_view(
                        view1_dict, device, keep_numpy_shape_batch=True
                    )
                    view2 = prepare_model_view(
                        view2_dict, device, keep_numpy_shape_batch=True
                    )

                    # No prompts for original model
                    pred1, pred2 = original_model(view1, view2)
                    idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                    results_original[idx1, idx2] = {
                        "pred1": pred1,
                        "pred2": pred2,
                        "view1": view1,
                        "view2": view2,
                    }
            debug_log("Original DUSt3R reconstruction completed")

        # === Trained Vanilla DUSt3R reconstruction (if enabled) ===
        model_trained_vanilla = None
        scene_trained_vanilla = None
        if show_trained and trained_vanilla_model is not None:
            debug_log("Running trained vanilla DUSt3R reconstruction...")
            with torch.no_grad():
                results_trained_vanilla = {}
                for view1_dict, view2_dict in image_pairs:
                    view1 = prepare_model_view(view1_dict, device)
                    view2 = prepare_model_view(view2_dict, device)

                    # No prompts for trained vanilla model
                    pred1, pred2 = trained_vanilla_model(view1, view2)
                    idx1, idx2 = view1_dict["idx"], view2_dict["idx"]
                    results_trained_vanilla[idx1, idx2] = {
                        "pred1": pred1,
                        "pred2": pred2,
                        "view1": view1,
                        "view2": view2,
                    }
            debug_log("Trained vanilla DUSt3R reconstruction completed")

        # Global alignment
        debug_log("Running global alignment...")

        # Collate all pairs into single dict (like dust3r inference returns)
        from dust3r.utils.device import to_cpu

        # Prepare Click3R results for alignment
        prompted_collated = collate_with_cat(
            list(results_with_prompts.values()), lists=False
        )
        prompted_collated = to_cpu(prompted_collated)
        debug_log(
            "Moved prompted results to CPU (global_aligner will handle GPU transfer)"
        )

        scene_with_prompts = global_aligner(
            prompted_collated, device=device, mode=GlobalAlignerMode.PairViewer
        )

        # Run global alignment for original DUSt3R (if enabled)
        if show_original and original_model is not None:
            debug_log("Running global optimization for original DUSt3R...")
            original_collated = collate_with_cat(
                list(results_original.values()), lists=False
            )
            original_collated = to_cpu(original_collated)
            scene_original = global_aligner(
                original_collated, device=device, mode=GlobalAlignerMode.PairViewer
            )

        # Run global alignment for trained vanilla DUSt3R (if enabled)
        if show_trained and trained_vanilla_model is not None:
            debug_log("Running global optimization for trained vanilla DUSt3R...")
            trained_vanilla_collated = collate_with_cat(
                list(results_trained_vanilla.values()), lists=False
            )
            trained_vanilla_collated = to_cpu(trained_vanilla_collated)
            scene_trained_vanilla = global_aligner(
                trained_vanilla_collated,
                device=device,
                mode=GlobalAlignerMode.PairViewer,
            )

        # Run global alignment for Click3R trained without occluded GT (if enabled)
        if show_overlap_gt and overlap_gt_model is not None:
            debug_log(
                "Running global optimization for Click3R (Trained Without Occluded GT)..."
            )
            overlap_gt_collated = collate_with_cat(
                list(results_overlap_gt.values()), lists=False
            )
            overlap_gt_collated = to_cpu(overlap_gt_collated)
            scene_overlap_gt = global_aligner(
                overlap_gt_collated, device=device, mode=GlobalAlignerMode.PairViewer
            )

        # Skip global alignment for VGGT and Pi3 - they output world-frame points directly
        debug_log(
            "Skipping global alignment for VGGT and Pi3 (using world-frame points directly)"
        )

        debug_log("Global alignment completed")

        # === 3D correspondence points, extracted per model from its aligned scene ===

        def extract_correspondence_points_from_scene(
            scene, prompts_v1, prompts_v2, model_name="", use_simple_scaling=False
        ):
            """Helper to extract correspondence points from a scene."""
            debug_log(
                f"Extracting 3D correspondence points for {model_name}... (simple_scaling={use_simple_scaling})"
            )

            if prompts_v1 is None or prompts_v2 is None:
                return None, None
            try:
                from dust3r.utils.device import to_numpy

                pts3d_all = to_numpy(scene.get_pts3d())

                if isinstance(pts3d_all, list):
                    pts3d_world_list = pts3d_all
                else:
                    pts3d_world_list = [pts3d_all]

                if len(pts3d_world_list) < 2:
                    raise ValueError(
                        f"Expected at least 2 images, got {len(pts3d_world_list)}"
                    )

                # Convert to torch tensors
                H_model, W_model = pts3d_world_list[0].shape[:2]
                pts3d_world_0 = (
                    torch.from_numpy(pts3d_world_list[0])
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                pts3d_world_1 = (
                    torch.from_numpy(pts3d_world_list[1])
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
                # Prompts are already in the 224×224 input space, so this mapping is the identity.
                coords1_transformed = transform_coords_to_dust3r_space(
                    prompts_v1["coords"].cpu(),
                    prompts_v1["shape"].cpu(),
                    size=224,
                    verbose=False,
                    use_simple_scaling=use_simple_scaling,
                ).to(device)

                coords2_transformed = transform_coords_to_dust3r_space(
                    prompts_v2["coords"].cpu(),
                    prompts_v2["shape"].cpu(),
                    size=224,
                    verbose=False,
                    use_simple_scaling=use_simple_scaling,
                ).to(device)

                # Create shape tensors for pts3d space
                shape1_model = torch.tensor(
                    [[H_model, W_model]], dtype=torch.float32, device=device
                )
                shape2_model = torch.tensor(
                    [[H_model, W_model]], dtype=torch.float32, device=device
                )

                # Sample at transformed prompt locations
                sampled_pts1 = sample_3d_at_prompts(
                    pts3d_world_0, coords1_transformed, shape1_model
                )
                sampled_pts2 = sample_3d_at_prompts(
                    pts3d_world_1, coords2_transformed, shape2_model
                )

                sampled_pts1_np = sampled_pts1[0].cpu().numpy()
                sampled_pts2_np = sampled_pts2[0].cpu().numpy()

                debug_log(
                    f"  {model_name}: Sampled {len(sampled_pts1_np)} correspondence points"
                )

                # Match the GLB export, which maps world points to the frame of camera 0
                # (dust3r/demo.py line 102).

                # Get camera poses from scene
                cams2world = scene.get_im_poses().cpu().numpy()
                cam0_pose = cams2world[0]  # First camera pose (4x4 matrix)

                # OPENGL matrix (from dust3r/viz.py line 326)
                OPENGL = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                )

                # dust3r's GLB uses inv(cam0 @ OPENGL @ rot_180y), but we post-process
                # the GLB to undo the rot_180y via fix_dust3r_glb_orientation().
                # So overlay points must match: use inv(cam0 @ OPENGL) without rot.
                transform = np.linalg.inv(cam0_pose @ OPENGL)

                # Apply transformation to all correspondence points
                # Convert to homogeneous coordinates (add 1 as 4th component)
                pts1_homogeneous = np.concatenate(
                    [sampled_pts1_np, np.ones((len(sampled_pts1_np), 1))], axis=1
                )
                pts2_homogeneous = np.concatenate(
                    [sampled_pts2_np, np.ones((len(sampled_pts2_np), 1))], axis=1
                )

                pts1_transformed = (transform @ pts1_homogeneous.T).T[:, :3]
                pts2_transformed = (transform @ pts2_homogeneous.T).T[:, :3]

                return pts1_transformed, pts2_transformed

            except Exception as e:
                debug_log(f"  {model_name}: ERROR - {e}")
                debug_log(f"  {model_name}: {traceback.format_exc()}")
                return None, None

        # Extract for PROMPTED model
        sampled_prompted_pts1_np, sampled_prompted_pts2_np = (
            extract_correspondence_points_from_scene(
                scene_with_prompts,
                prompts_view1,
                prompts_view2,
                "PROMPTED",
                use_simple_scaling,
            )
        )

        # Get colors for each pair
        pair_colors = [get_pair_color_rgba(i) for i in range(len(correspondence_pairs))]

        import time

        timestamp = str(int(time.time() * 1000))

        # Generate 3D models - use separate output directories to avoid conflicts
        debug_log("Generating 3D models...")

        prompted_dir = os.path.join(output_dir, f"prompted_{timestamp}")
        os.makedirs(prompted_dir, exist_ok=True)

        # Compute percentile-based threshold for DUSt3R models
        prompted_conf_thr = compute_percentile_threshold(
            scene_with_prompts, min_conf_thr
        )
        debug_log(
            f"Prompted confidence threshold (percentile {min_conf_thr}): {prompted_conf_thr:.3f}"
        )

        debug_log(f"Prompted output: {prompted_dir}")
        model_with_prompts = get_3D_model_from_scene(
            prompted_dir,
            silent=False,
            scene=scene_with_prompts,
            min_conf_thr=prompted_conf_thr,
            as_pointcloud=as_pointcloud,
            cam_size=cam_size,
        )
        model_with_prompts = fix_dust3r_glb_orientation(model_with_prompts)

        # Log the model bounds (printed only with --debug).
        if model_with_prompts and os.path.exists(model_with_prompts):
            try:
                debug_scene = trimesh.load(model_with_prompts)
                if isinstance(debug_scene, trimesh.Scene):
                    all_bounds = []
                    for geom in debug_scene.geometry.values():
                        if hasattr(geom, "bounds"):
                            all_bounds.append(geom.bounds)
                    if all_bounds:
                        all_bounds = np.array(all_bounds)
                        min_bounds = all_bounds[:, 0, :].min(axis=0)
                        max_bounds = all_bounds[:, 1, :].max(axis=0)
                        debug_log(
                            f"  Scene bounds: X[{min_bounds[0]:.3f}, {max_bounds[0]:.3f}], "
                            f"Y[{min_bounds[1]:.3f}, {max_bounds[1]:.3f}], "
                            f"Z[{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]"
                        )
                else:
                    bounds = debug_scene.bounds
                    debug_log(
                        f"  Model bounds: X[{bounds[0, 0]:.3f}, {bounds[1, 0]:.3f}], "
                        f"Y[{bounds[0, 1]:.3f}, {bounds[1, 1]:.3f}], "
                        f"Z[{bounds[0, 2]:.3f}, {bounds[1, 2]:.3f}]"
                    )
            except Exception as e:
                debug_log(f"  Could not load model for bounds check: {e}")

        # Generate 3D model for original DUSt3R (if enabled)
        if show_original and scene_original is not None:
            original_dir = os.path.join(output_dir, f"original_{timestamp}")
            os.makedirs(original_dir, exist_ok=True)
            original_conf_thr = compute_percentile_threshold(
                scene_original, min_conf_thr
            )
            debug_log(
                f"Original DUSt3R confidence threshold (percentile {min_conf_thr}): {original_conf_thr:.3f}"
            )
            debug_log(f"Original output: {original_dir}")
            model_original = get_3D_model_from_scene(
                original_dir,
                silent=False,
                scene=scene_original,
                min_conf_thr=original_conf_thr,
                as_pointcloud=as_pointcloud,
                cam_size=cam_size,
            )
            model_original = fix_dust3r_glb_orientation(model_original)

        # Generate 3D model for trained vanilla DUSt3R (if enabled)
        if show_trained and scene_trained_vanilla is not None:
            trained_vanilla_dir = os.path.join(
                output_dir, f"trained_vanilla_{timestamp}"
            )
            os.makedirs(trained_vanilla_dir, exist_ok=True)
            trained_vanilla_conf_thr = compute_percentile_threshold(
                scene_trained_vanilla, min_conf_thr
            )
            debug_log(
                f"Trained vanilla DUSt3R confidence threshold (percentile {min_conf_thr}): {trained_vanilla_conf_thr:.3f}"
            )
            debug_log(f"Trained vanilla output: {trained_vanilla_dir}")
            model_trained_vanilla = get_3D_model_from_scene(
                trained_vanilla_dir,
                silent=False,
                scene=scene_trained_vanilla,
                min_conf_thr=trained_vanilla_conf_thr,
                as_pointcloud=as_pointcloud,
                cam_size=cam_size,
            )
            model_trained_vanilla = fix_dust3r_glb_orientation(model_trained_vanilla)

        # Generate 3D model for Click3R trained without occluded GT (if enabled)
        if show_overlap_gt and scene_overlap_gt is not None:
            overlap_gt_dir = os.path.join(output_dir, f"overlap_gt_{timestamp}")
            os.makedirs(overlap_gt_dir, exist_ok=True)
            overlap_gt_conf_thr = compute_percentile_threshold(
                scene_overlap_gt, min_conf_thr
            )
            debug_log(
                f"Overlap GT Click3R confidence threshold (percentile {min_conf_thr}): {overlap_gt_conf_thr:.3f}"
            )
            debug_log(f"Click3R (Trained Without Occluded GT) output: {overlap_gt_dir}")
            model_overlap_gt = get_3D_model_from_scene(
                overlap_gt_dir,
                silent=False,
                scene=scene_overlap_gt,
                min_conf_thr=overlap_gt_conf_thr,
                as_pointcloud=as_pointcloud,
                cam_size=cam_size,
            )
            model_overlap_gt = fix_dust3r_glb_orientation(model_overlap_gt)

        # Generate 3D model for VGGT (if enabled) - use world points directly
        vggt_glb_transform = None
        pi3_glb_transform = None
        if show_vggt and vggt_model is not None and len(results_vggt) > 0:
            vggt_dir = os.path.join(output_dir, f"vggt_{timestamp}")
            os.makedirs(vggt_dir, exist_ok=True)
            debug_log(f"VGGT output: {vggt_dir}")

            # Use the (0, 1) pair so pred1/pred2 are images 0/1, matching prompts_view1/2;
            # make_pairs lists (1, 0) first.
            if (0, 1) in results_vggt:
                pair_key = (0, 1)
            else:
                pair_key = list(results_vggt.keys())[0]
            vggt_pred1 = results_vggt[pair_key]["pred1"]
            vggt_pred2 = results_vggt[pair_key]["pred2"]
            vggt_view1 = results_vggt[pair_key]["view1"]
            vggt_view2 = results_vggt[pair_key]["view2"]

            model_vggt, vggt_glb_transform = create_3d_model_from_world_points(
                vggt_dir,
                vggt_pred1,
                vggt_pred2,
                vggt_view1,
                vggt_view2,
                min_conf_percentile=min_conf_thr,
                as_pointcloud=as_pointcloud,
                cam_size=cam_size,
                model_name="vggt_scene",
            )

        # Generate 3D model for Pi3 (if enabled) - use world points directly
        if show_pi3 and pi3_model is not None and len(results_pi3) > 0:
            pi3_dir = os.path.join(output_dir, f"pi3_{timestamp}")
            os.makedirs(pi3_dir, exist_ok=True)
            debug_log(f"Pi3 output: {pi3_dir}")

            # Use the (0, 1) pair, as for VGGT.
            if (0, 1) in results_pi3:
                pair_key = (0, 1)
            else:
                pair_key = list(results_pi3.keys())[0]
            pi3_pred1 = results_pi3[pair_key]["pred1"]
            pi3_pred2 = results_pi3[pair_key]["pred2"]
            pi3_view1 = results_pi3[pair_key]["view1"]
            pi3_view2 = results_pi3[pair_key]["view2"]

            model_pi3, pi3_glb_transform = create_3d_model_from_world_points(
                pi3_dir,
                pi3_pred1,
                pi3_pred2,
                pi3_view1,
                pi3_view2,
                min_conf_percentile=min_conf_thr,
                as_pointcloud=as_pointcloud,
                cam_size=cam_size,
                model_name="pi3_scene",
            )

        debug_log("3D models generated")

        # === Optionally overlay correspondences on reconstruction models ===
        if show_corr_overlay:
            debug_log("Creating overlay versions with correspondences...")

            # Overlay on prompted model (using prompted correspondences)
            if model_with_prompts and sampled_prompted_pts1_np is not None:
                prompted_overlay_path = os.path.join(
                    prompted_dir, f"scene_overlay_{timestamp}.glb"
                )
                model_with_prompts = overlay_correspondences_on_model(
                    model_with_prompts,
                    sampled_prompted_pts1_np,
                    sampled_prompted_pts2_np,
                    pair_colors,
                    prompted_overlay_path,
                )

            # Overlay on original model (if enabled - extract separately)
            if model_original and show_original and scene_original is not None:
                sampled_orig_pts1_np, sampled_orig_pts2_np = (
                    extract_correspondence_points_from_scene(
                        scene_original,
                        prompts_view1,
                        prompts_view2,
                        "ORIGINAL",
                        use_simple_scaling,
                    )
                )
                if sampled_orig_pts1_np is not None:
                    original_overlay_path = os.path.join(
                        original_dir, f"scene_overlay_{timestamp}.glb"
                    )
                    model_original = overlay_correspondences_on_model(
                        model_original,
                        sampled_orig_pts1_np,
                        sampled_orig_pts2_np,
                        pair_colors,
                        original_overlay_path,
                    )

            # Overlay on trained vanilla model (if enabled - extract separately)
            if (
                model_trained_vanilla
                and show_trained
                and scene_trained_vanilla is not None
            ):
                sampled_vanilla_pts1_np, sampled_vanilla_pts2_np = (
                    extract_correspondence_points_from_scene(
                        scene_trained_vanilla,
                        prompts_view1,
                        prompts_view2,
                        "TRAINED_VANILLA",
                        use_simple_scaling,
                    )
                )
                if sampled_vanilla_pts1_np is not None:
                    trained_vanilla_overlay_path = os.path.join(
                        trained_vanilla_dir, f"scene_overlay_{timestamp}.glb"
                    )
                    model_trained_vanilla = overlay_correspondences_on_model(
                        model_trained_vanilla,
                        sampled_vanilla_pts1_np,
                        sampled_vanilla_pts2_np,
                        pair_colors,
                        trained_vanilla_overlay_path,
                    )

            # Overlay on overlap GT model (if enabled - extract separately)
            if model_overlap_gt and show_overlap_gt and scene_overlap_gt is not None:
                sampled_overlap_gt_pts1_np, sampled_overlap_gt_pts2_np = (
                    extract_correspondence_points_from_scene(
                        scene_overlap_gt,
                        prompts_view1,
                        prompts_view2,
                        "OVERLAP_GT",
                        use_simple_scaling,
                    )
                )
                if sampled_overlap_gt_pts1_np is not None:
                    overlap_gt_overlay_path = os.path.join(
                        overlap_gt_dir, f"scene_overlay_{timestamp}.glb"
                    )
                    model_overlap_gt = overlay_correspondences_on_model(
                        model_overlap_gt,
                        sampled_overlap_gt_pts1_np,
                        sampled_overlap_gt_pts2_np,
                        pair_colors,
                        overlap_gt_overlay_path,
                    )

            # Overlay on VGGT model (if enabled - sample world points directly)
            if (
                model_vggt
                and show_vggt
                and vggt_model is not None
                and len(results_vggt) > 0
            ):
                if (0, 1) in results_vggt:
                    pair_key = (0, 1)
                else:
                    pair_key = list(results_vggt.keys())[0]
                vggt_pred1 = results_vggt[pair_key]["pred1"]
                vggt_pred2 = results_vggt[pair_key]["pred2"]

                sampled_vggt_pts1_np, sampled_vggt_pts2_np = (
                    sample_world_points_at_prompts(
                        vggt_pred1,
                        vggt_pred2,
                        prompts_view1,
                        prompts_view2,
                        use_simple_scaling,
                    )
                )
                if sampled_vggt_pts1_np is not None:
                    # Apply the same GLB transform to overlay points
                    if vggt_glb_transform is not None:
                        pts1_h = np.concatenate(
                            [
                                sampled_vggt_pts1_np,
                                np.ones((len(sampled_vggt_pts1_np), 1)),
                            ],
                            axis=1,
                        )
                        pts2_h = np.concatenate(
                            [
                                sampled_vggt_pts2_np,
                                np.ones((len(sampled_vggt_pts2_np), 1)),
                            ],
                            axis=1,
                        )
                        sampled_vggt_pts1_np = (vggt_glb_transform @ pts1_h.T).T[:, :3]
                        sampled_vggt_pts2_np = (vggt_glb_transform @ pts2_h.T).T[:, :3]
                    vggt_overlay_path = os.path.join(vggt_dir, "vggt_scene_overlay.glb")
                    model_vggt = overlay_correspondences_on_model(
                        model_vggt,
                        sampled_vggt_pts1_np,
                        sampled_vggt_pts2_np,
                        pair_colors,
                        vggt_overlay_path,
                    )

            # Overlay on Pi3 model (if enabled - sample world points directly)
            if (
                model_pi3
                and show_pi3
                and pi3_model is not None
                and len(results_pi3) > 0
            ):
                if (0, 1) in results_pi3:
                    pair_key = (0, 1)
                else:
                    pair_key = list(results_pi3.keys())[0]
                pi3_pred1 = results_pi3[pair_key]["pred1"]
                pi3_pred2 = results_pi3[pair_key]["pred2"]

                sampled_pi3_pts1_np, sampled_pi3_pts2_np = (
                    sample_world_points_at_prompts(
                        pi3_pred1,
                        pi3_pred2,
                        prompts_view1,
                        prompts_view2,
                        use_simple_scaling,
                    )
                )
                if sampled_pi3_pts1_np is not None:
                    # Apply the same GLB transform to overlay points
                    if pi3_glb_transform is not None:
                        pts1_h = np.concatenate(
                            [
                                sampled_pi3_pts1_np,
                                np.ones((len(sampled_pi3_pts1_np), 1)),
                            ],
                            axis=1,
                        )
                        pts2_h = np.concatenate(
                            [
                                sampled_pi3_pts2_np,
                                np.ones((len(sampled_pi3_pts2_np), 1)),
                            ],
                            axis=1,
                        )
                        sampled_pi3_pts1_np = (pi3_glb_transform @ pts1_h.T).T[:, :3]
                        sampled_pi3_pts2_np = (pi3_glb_transform @ pts2_h.T).T[:, :3]
                    pi3_overlay_path = os.path.join(pi3_dir, "pi3_scene_overlay.glb")
                    model_pi3 = overlay_correspondences_on_model(
                        model_pi3,
                        sampled_pi3_pts1_np,
                        sampled_pi3_pts2_np,
                        pair_colors,
                        pi3_overlay_path,
                    )

            debug_log("Overlay generation completed")

        # Generate comparison text with metrics
        comparison_text = f"Reconstruction complete. {len(correspondence_pairs)} matching point pairs.\n"
        comparison_text += f"Correspondence overlay on models: {'ON' if show_corr_overlay else 'OFF'}\n"
        if show_original and original_model is not None:
            comparison_text += "✓ Comparing with original pretrained DUSt3R model\n"
        if show_trained and trained_vanilla_model is not None:
            comparison_text += "✓ Comparing with trained vanilla DUSt3R model\n"
        if show_overlap_gt and overlap_gt_model is not None:
            comparison_text += (
                "✓ Comparing with Click3R (Trained Without Occluded GT)\n"
            )
        if show_vggt and vggt_model is not None:
            comparison_text += "✓ Comparing with VGGT\n"
        if show_pi3 and pi3_model is not None:
            comparison_text += "✓ Comparing with Pi3\n"
        comparison_text += "\n"

        # Show metrics
        if DEBUG:
            comparison_text += f"3D distance at clicks: {prompted_metrics['correspondence_distance']:.4f}\n"

        # Show pair details with colors
        comparison_text += "=== CORRESPONDENCE PAIRS ===\n"
        for i, ((x1, y1), (x2, y2)) in enumerate(correspondence_pairs):
            color = get_pair_color(i)
            comparison_text += (
                f"Pair {i + 1} ({color}): ({x1:.3f},{y1:.3f}) ↔ ({x2:.3f},{y2:.3f})\n"
            )

        # Generate attention visualizations if requested
        attention_heatmaps = []
        if show_attention and attention_maps_storage is not None:
            try:
                debug_log(f"Generating attention heatmaps for layer {attention_layer}")

                # Get image embedding size from model
                img_emb_size = (14, 14)  # Default for 224x224 images with patch_size=16
                if hasattr(model, "patch_embed"):
                    if hasattr(model.patch_embed, "grid_size"):
                        img_emb_size = model.patch_embed.grid_size

                # Get loaded images tensors (index [0] to remove batch dimension)
                images_tensor = torch.stack(
                    [loaded_images[0]["img"][0], loaded_images[1]["img"][0]]
                )  # (2, 3, H, W)

                # Prepare prompt coordinates and token coordinates
                prompt_coords_tensor = torch.stack(
                    [prompts_view1["coords"][0], prompts_view2["coords"][0]]
                )  # (2, P, 2)

                # Scale prompt coordinates from the prompt shape to the loaded image size.
                H1_orig, W1_orig = prompts_view1["shape"][0].cpu().numpy()
                H2_orig, W2_orig = prompts_view2["shape"][0].cpu().numpy()
                H_resized, W_resized = loaded_images[0]["img"].shape[2:4]

                scale_x1 = W_resized / W1_orig
                scale_y1 = H_resized / H1_orig
                scale_x2 = W_resized / W2_orig
                scale_y2 = H_resized / H2_orig

                prompt_coords_scaled = prompt_coords_tensor.clone()
                prompt_coords_scaled[0, :, 0] *= scale_x1
                prompt_coords_scaled[0, :, 1] *= scale_y1
                prompt_coords_scaled[1, :, 0] *= scale_x2
                prompt_coords_scaled[1, :, 1] *= scale_y2

                # Convert to token coordinates
                _, x_tok1, y_tok1 = coords_to_token_indices(
                    prompts_view1["coords"], prompts_view1["shape"], img_emb_size
                )
                _, x_tok2, y_tok2 = coords_to_token_indices(
                    prompts_view2["coords"], prompts_view2["shape"], img_emb_size
                )

                tok_coords1 = torch.stack([x_tok1[0], y_tok1[0]], dim=-1)
                tok_coords2 = torch.stack([x_tok2[0], y_tok2[0]], dim=-1)
                prompt_token_coords = torch.stack([tok_coords1, tok_coords2])

                valid_masks = torch.stack(
                    [prompts_view1["valid_mask"][0], prompts_view2["valid_mask"][0]]
                )

                attention_heatmaps = generate_attention_heatmaps(
                    images_tensor,
                    attention_maps_storage[attention_layer],
                    prompt_coords_scaled,
                    prompt_token_coords,
                    valid_masks,
                    attention_layer,
                    img_emb_size,
                )

                debug_log(f"Generated {len(attention_heatmaps)} attention heatmaps")
                comparison_text += "\n=== ATTENTION VISUALIZATION ===\n"
                comparison_text += f"Layer {attention_layer}: {len(attention_heatmaps)} prompt heatmaps generated\n"

            except Exception as e:
                debug_log(f"Error generating attention heatmaps: {e}")
                debug_log(f"Traceback: {traceback.format_exc()}")
                comparison_text += (
                    f"\nWarning: Could not generate attention visualizations ({e})\n"
                )

        # Cleanup temp files
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
                debug_log(f"Cleaned up {temp_path}")
            except OSError:
                pass

        debug_log("Reconstruction completed successfully")
        return ReconstructionResult(
            model_original,
            model_trained_vanilla,
            model_vggt,
            model_pi3,
            model_overlap_gt,
            model_with_prompts,
            comparison_text,
        )

    except Exception as e:
        debug_log(f"Error in reconstruction: {e}")
        debug_log(f"Traceback: {traceback.format_exc()}")
        message = f"Reconstruction failed: {e}"
        if DEBUG:
            message += f"\n{traceback.format_exc()}"
        return ReconstructionResult(None, None, None, None, None, None, message)


def load_examples(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return []
    records = json.loads(manifest_path.read_text())["examples"]
    for record in records:
        for key in ("image1", "image2"):
            record[key] = str((manifest_path.parent / record[key]).resolve())
            # Decode once at startup; reopening files per selection is slow on network disks.
            EXAMPLE_IMAGES[record[key]] = read_input_image(record[key])
    return records


def load_example(state, example, with_clicks=False):
    result = session_edit(
        state, update_images_display, [example["image1"], example["image2"]]
    )
    state = result[-1]
    state["example"] = example
    if with_clicks:
        with session_scope(state):
            for corr in example["correspondences"]:
                add_correspondence_point(corr["x1"], corr["y1"], 0)
                result = (*add_correspondence_point(corr["x2"], corr["y2"], 1), state)
    return result


def create_interface(
    model,
    device,
    output_dir,
    checkpoint_name,
    img_size=(224, 224),
    original_model=None,
    trained_vanilla_model=None,
    vggt_model=None,
    pi3_model=None,
    overlap_gt_model=None,
    examples_path=None,
    gpu_decorator=None,
):
    examples = load_examples(
        examples_path or Path(REPO_ROOT) / "assets/examples/manifest.json"
    )
    by_id = {item["id"]: item for item in examples}

    # Suggested clicks exist only for the examples: the button is enabled for an example and disabled for uploads.
    def select_example(example_id, state):
        if example_id not in by_id:
            return gr.skip(), gr.skip(), "Choose an example.", state, gr.skip()
        return (*load_example(state, by_id[example_id]), gr.update(interactive=True))

    def suggested_clicks(state):
        if not state or not state.get("example"):
            gr.Info("Suggested clicks are only available for the examples.")
            return gr.skip(), gr.skip(), gr.skip(), state
        return load_example(state, state["example"], with_clicks=True)

    def uploaded(files, state):
        if not files or len(files) != 2:
            gr.Warning("Please upload exactly two images of the same scene.")
            # keep the current pair; clear the upload box
            return gr.skip(), gr.skip(), gr.skip(), state, gr.skip(), gr.skip(), None
        result = session_edit(state, update_images_display, files)
        if result[0] is None or result[1] is None:
            # Unreadable file (e.g. HEIC): keep the current pair and say why.
            gr.Warning("Could not read one of the images. Please upload JPEG, PNG or HEIC images.")
            return gr.skip(), gr.skip(), gr.skip(), state, gr.skip(), gr.skip(), None
        result[-1]["example"] = None
        return (*result, None, gr.update(interactive=False), gr.skip())

    def click_left(state, evt: gr.SelectData):
        return session_edit(state, handle_img1_click, evt)

    def click_right(state, evt: gr.SelectData):
        return session_edit(state, handle_img2_click, evt)

    def reconstruct(state, confidence, camera_size, pointcloud, overlay):
        state = copy.deepcopy(state) if state else new_session()
        with session_scope(state):
            result = run_reconstruction(
                confidence,
                camera_size,
                pointcloud,
                original_model is not None,
                False,
                False,
                False,
                False,
                False,
                0,
                overlay,
                False,
                model,
                device,
                str(Path(output_dir) / state["id"] / uuid.uuid4().hex),
                img_size,
                original_model=original_model,
            )
        return (
            frame_model_for_viewer(result.original),
            frame_model_for_viewer(result.prompted),
            result.message,
        )

    if gpu_decorator is not None:
        reconstruct = gpu_decorator(reconstruct)

    with gr.Blocks(title=PAPER_TITLE) as interface:
        state = gr.State(value=None)
        gr.Markdown(
            f"# {PAPER_TITLE}\n"
            "Click3R reconstructs a scene in 3D from two images, guided by a few "
            "clicks that mark the same point in both views.\n\n"
            "1. **Choose an example** or **upload two images** of the same scene.\n"
            "2. **Add correspondence clicks.** For an example, press **Add suggested "
            "clicks**. For your own images, click a point in Image 1 and then the same "
            "point in Image 2; repeat for more pairs. Markers of the same color form "
            "one pair, and **Clear clicks** removes them.\n"
            "3. **Press Reconstruct** to compare the 3D results of DUSt3R and Click3R."
        )
        with gr.Row(equal_height=True):
            example_picker = gr.Dropdown(
                choices=[(e["title"], e["id"]) for e in examples],
                label="Example pair",
                value=examples[0]["id"] if examples else None,
            )
            files = gr.File(
                label="Upload two images",
                file_count="multiple",
                # explicit extensions: some systems give HEIC no image MIME type, so "image" alone greys them out
                file_types=["image", ".heic", ".heif"],
                type="filepath",
            )
        with gr.Row():
            suggested_button = gr.Button("Add suggested clicks")
        with gr.Row():
            # JPEG: Gradio encodes each returned image; JPEG takes ~5 ms per image, WebP ~85 ms.
            left = gr.Image(label="Image 1", type="pil", interactive=False, format="jpeg")
            right = gr.Image(label="Image 2", type="pil", interactive=False, format="jpeg")
        status = gr.Textbox(
            label="Correspondences", interactive=False, lines=4, visible=False
        )
        with gr.Row():
            clear = gr.Button("Clear clicks")
            run = gr.Button("Reconstruct", variant="primary")
        with gr.Accordion("Display settings", open=False):
            confidence = gr.Slider(0, 99, value=10, label="Confidence percentile")
            camera_size = gr.Slider(0.05, 0.5, value=0.2, label="Camera size")
            point_size_control()
            pointcloud = gr.Checkbox(value=True, label="Point cloud")
            overlay = gr.Checkbox(value=True, label="Show clicks in 3D")
            show_info = gr.Checkbox(value=False, label="Show information panels")
        with gr.Row():
            with gr.Column(min_width=160, visible=original_model is not None):
                original = ReconstructionViewer(label="DUSt3R")
            with gr.Column(min_width=160):
                prompted = ReconstructionViewer(label="Click3R")
        results = gr.Textbox(
            label="Reconstruction", interactive=False, lines=4, visible=False
        )
        show_info.change(
            lambda visible: (gr.update(visible=visible), gr.update(visible=visible)),
            [show_info],
            [status, results],
            queue=False,
        )
        outputs = [left, right, status, state]
        # Edits run one at a time and read the state when they run, so every queued click builds on the last one.
        edit_queue = dict(concurrency_limit=1, concurrency_id="session_edits")
        # Clicks: queue every click (the default drops clicks made while one is pending) and update the image
        # without a loading overlay.
        click_queue = dict(edit_queue, trigger_mode="multiple", show_progress="hidden")
        # Only user changes count: an upload clears the picker without reloading.
        example_picker.input(
            select_example,
            [example_picker, state],
            [*outputs, suggested_button],
            **edit_queue,
        )
        suggested_button.click(suggested_clicks, [state], outputs, **edit_queue)
        files.upload(
            uploaded,
            [files, state],
            [*outputs, example_picker, suggested_button, files],
            **edit_queue,
        )
        left.select(click_left, [state], outputs, **click_queue)
        right.select(click_right, [state], outputs, **click_queue)
        clear.click(
            lambda st: session_edit(st, clear_correspondences),
            [state],
            outputs,
            **edit_queue,
            show_progress="hidden",
        )
        run.click(
            reconstruct,
            [state, confidence, camera_size, pointcloud, overlay],
            [original, prompted, results],
            concurrency_limit=1,
            concurrency_id="reconstruction",
        )
        if examples:
            interface.load(
                select_example,
                [example_picker, state],
                [*outputs, suggested_button],
                **edit_queue,
            )
    return interface


def main():
    parser = argparse.ArgumentParser(description="Click3R interactive stereo demo")
    parser.add_argument("--model", default="checkpoints/click3r.pth")
    parser.add_argument(
        "--dust3r-model",
        default="checkpoints/dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="interactive_output")
    parser.add_argument(
        "--examples", default=str(Path(REPO_ROOT) / "assets/examples/manifest.json")
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument(
        "--server-port",
        type=int,
        help="Default: the first free port from 7860",
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    global DEBUG
    DEBUG = args.debug
    model, img_size = load_click3r(args.model, args.device)
    original = None
    if args.dust3r_model:
        if not Path(args.dust3r_model).is_file():
            raise FileNotFoundError(args.dust3r_model)
        original, _ = load_original_dust3r(args.dust3r_model, args.device)
        if original is None:
            raise RuntimeError("Could not load the DUSt3R comparison checkpoint")
    interface = create_interface(
        model,
        args.device,
        args.output,
        Path(args.model).name,
        img_size=img_size,
        original_model=original,
        examples_path=args.examples,
    )
    interface.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[str(Path(args.output).resolve())],
    )


if __name__ == "__main__":
    main()
