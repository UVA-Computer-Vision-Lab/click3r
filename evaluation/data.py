#!/usr/bin/env python3
"""Calibrated image-pair loading and model helpers for evaluation."""

import os.path as osp
import sys
from math import inf

import numpy as np
import torch
from PIL import Image

# ─── paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "dust3r"))
sys.path.insert(0, osp.join(REPO_ROOT, "third_party", "segment-anything"))

from click3r.model import Click3R

# ─── constants ────────────────────────────────────────────────────────────────
MAX_P = 12  # Must match training
IMG_SIZE = 224
DATA_ROOT_DEFAULT = "data/processed"
OOD_DATA_ROOT = "data/ood"

# Optional (old_prefix, new_prefix) pairs that remap_path applies to source paths.
PATH_REMAPS = []

# ScanNet special case: color_90 → color, frame_XXXX.jpg → XXXXX.jpg
SCANNET_COLOR_REMAP = ("color_90", "color")

INDIST_DATASET_ROOTS = {
    "Co3d_test": "co3d_processed_for_dust3r",
    "WildRGBD_test": "wildrgbd_processed_for_dust3r",
    "MegaDepth_val": "megadepth_processed_for_dust3r",
    "BlendedMVS_val": "blendedmvs_processed_for_dust3r",
    "Habitat_val": "habitat_processed_for_dust3r",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════


def create_model():
    """Create the Click3R model architecture."""
    return Click3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(IMG_SIZE, IMG_SIZE),
        head_type="linear",
        output_mode="pts3d",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )


def load_click3r(checkpoint_path, device):
    """Load Click3R with correspondence adapters enabled."""
    from evaluation.baselines.loading import load_click3r_model

    return load_click3r_model(checkpoint_path, create_model(), device)


def load_dust3r_base(checkpoint_path, device):
    """Load a DUSt3R baseline with correspondence adapters bypassed."""
    from evaluation.baselines.loading import load_dust3r_model

    return load_dust3r_model(checkpoint_path, create_model(), device)


def remap_path(path):
    """Apply user-supplied source path mappings."""
    for old_prefix, new_prefix in PATH_REMAPS:
        if path.startswith(old_prefix):
            path = new_prefix + path[len(old_prefix) :]
            break

    # ScanNet special: color_90 → color
    if SCANNET_COLOR_REMAP[0] in path:
        path = path.replace(SCANNET_COLOR_REMAP[0], SCANNET_COLOR_REMAP[1])
        # Also remap frame_XXXX.jpg → XXXXX.jpg for ScanNet
        base = osp.basename(path)
        if base.startswith("frame_"):
            num = base.replace("frame_", "").replace(".jpg", "")
            new_name = f"{int(num):05d}.jpg"
            path = osp.join(osp.dirname(path), new_name)

    return path


def load_and_preprocess_image(img_path, size=IMG_SIZE):
    """Load image, center-crop to square, resize to size×size, normalize to [-1,1].

    Returns:
        img_tensor: (1, 3, H, W) float32 tensor in [-1,1]
        true_shape: np.array([H, W])
        original_size: (orig_H, orig_W) before crop
    """
    img = Image.open(img_path).convert("RGB")
    orig_W, orig_H = img.size

    # Center crop to square (DUSt3R convention)
    short_side = min(orig_W, orig_H)
    left = (orig_W - short_side) // 2
    top = (orig_H - short_side) // 2
    img = img.crop((left, top, left + short_side, top + short_side))
    img = img.resize((size, size), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2 - 1  # [0,1] → [-1,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return tensor, np.array([size, size]), (orig_H, orig_W)


def make_ood_batch(entry, device):
    """Build a (v1, v2) batch from a self-contained OOD benchmark entry.

    Loads images, constructs camera matrices, and optionally loads depth/pts3d.
    """
    # Preprocessed entries (with image_space) store final image paths, so
    # remap_path, including the ScanNet color_90 remap, is skipped for them.
    resolve = (lambda path: path) if entry.get("image_space") else remap_path
    img1_path = resolve(entry["image1_path"])
    img2_path = resolve(entry["image2_path"])

    if not osp.isfile(img1_path) or not osp.isfile(img2_path):
        return None, None, f"Images not found: {img1_path}, {img2_path}"

    img1, _, orig1 = load_and_preprocess_image(img1_path)
    img2, _, orig2 = load_and_preprocess_image(img2_path)
    # true_shape must be a tensor for _interleave_imgs (stacks along batch dim)
    ts1 = torch.tensor([[IMG_SIZE, IMG_SIZE]], dtype=torch.long)
    ts2 = torch.tensor([[IMG_SIZE, IMG_SIZE]], dtype=torch.long)

    # Camera matrices from JSON
    pose1 = torch.tensor(entry["pose1_c2w"], dtype=torch.float32)
    pose2 = torch.tensor(entry["pose2_c2w"], dtype=torch.float32)
    K1 = torch.tensor(entry["K1"], dtype=torch.float32)
    K2 = torch.tensor(entry["K2"], dtype=torch.float32)

    # Map K from the original image to the 224 x 224 center crop. Preprocessed
    # entries store 224 x 224 images but K for the original resolution, so the
    # crop is computed from the recorded img_size1/img_size2.
    camera_orig1 = orig1
    camera_orig2 = orig2
    if entry.get("image_space") == "center_crop_224":
        if orig1 != (IMG_SIZE, IMG_SIZE) or orig2 != (IMG_SIZE, IMG_SIZE):
            raise ValueError("Preprocessed images must be 224 x 224")
        camera_orig1 = tuple(reversed(entry["img_size1"]))
        camera_orig2 = tuple(reversed(entry["img_size2"]))
    orig_H1, orig_W1 = camera_orig1
    short1 = min(orig_H1, orig_W1)
    scale1 = IMG_SIZE / short1
    offset_x1 = (orig_W1 - short1) / 2
    offset_y1 = (orig_H1 - short1) / 2

    K1_scaled = K1.clone()
    K1_scaled[0, 0] *= scale1  # fx
    K1_scaled[1, 1] *= scale1  # fy
    K1_scaled[0, 2] = (K1[0, 2] - offset_x1) * scale1  # cx
    K1_scaled[1, 2] = (K1[1, 2] - offset_y1) * scale1  # cy

    orig_H2, orig_W2 = camera_orig2
    short2 = min(orig_H2, orig_W2)
    scale2 = IMG_SIZE / short2
    offset_x2 = (orig_W2 - short2) / 2
    offset_y2 = (orig_H2 - short2) / 2

    K2_scaled = K2.clone()
    K2_scaled[0, 0] *= scale2
    K2_scaled[1, 1] *= scale2
    K2_scaled[0, 2] = (K2[0, 2] - offset_x2) * scale2
    K2_scaled[1, 2] = (K2[1, 2] - offset_y2) * scale2

    # instance/idx/label are required by DUSt3R's is_symmetrized() and
    # _interleave_imgs(). They must be list-like with len == B.
    v1 = {
        "img": img1.to(device),
        "true_shape": ts1,
        "camera_pose": pose1.unsqueeze(0).to(device),
        "camera_intrinsics": K1_scaled.unsqueeze(0).to(device),
        "instance": ["0"],
        "idx": [0],
        "label": [entry.get("pair_id", "ood")],
    }
    v2 = {
        "img": img2.to(device),
        "true_shape": ts2,
        "camera_pose": pose2.unsqueeze(0).to(device),
        "camera_intrinsics": K2_scaled.unsqueeze(0).to(device),
        "instance": ["0"],
        "idx": [0],
        "label": [entry.get("pair_id", "ood")],
    }

    # Try to load depth and compute pts3d for GT prompt generation
    depth1_path = resolve(entry.get("depth1_path", ""))
    depth2_path = resolve(entry.get("depth2_path", ""))
    _try_load_depth_pts3d(v1, depth1_path, pose1, K1_scaled, orig1, device)
    _try_load_depth_pts3d(v2, depth2_path, pose2, K2_scaled, orig2, device)

    return v1, v2, None


def _try_load_raw_float32_depth(depth_path, orig_size):
    """Try reading a binary file as raw float32 depth (ETH3D format).

    ETH3D stores depth as raw float32 arrays with no header, at the
    original camera resolution (e.g. 6048×4032).
    """
    fsize = osp.getsize(depth_path)
    total_floats = fsize // 4
    if fsize % 4 != 0 or total_floats < 1000:
        return None

    orig_H, orig_W = orig_size
    # Check if it matches the original image dimensions
    if total_floats == orig_H * orig_W:
        data = np.fromfile(depth_path, dtype=np.float32)
        return data.reshape(orig_H, orig_W)

    # Also accept the transposed size.
    for H_try, W_try in [(orig_H, orig_W), (orig_W, orig_H)]:
        if H_try * W_try == total_floats:
            data = np.fromfile(depth_path, dtype=np.float32)
            return data.reshape(H_try, W_try)

    return None


def _try_load_depth_pts3d(view, depth_path, c2w, K_scaled, orig_size, device):
    """Try to load depth map and back-project to 3D points for GT prompt generation."""
    if not depth_path or not osp.isfile(depth_path):
        return

    try:
        ext = osp.splitext(depth_path)[1].lower()
        if ext == ".npy":
            depth = np.load(depth_path)
        elif ext == ".pfm":
            depth = _read_pfm(depth_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            # Try PIL first; if it fails, try raw float32 binary
            # (ETH3D stores depth as raw float32 with .JPG extension)
            try:
                depth_img = Image.open(depth_path)
                depth = np.array(depth_img, dtype=np.float32)
                # uint16 depth, or depth with values above 100, is in millimeters; convert to meters.
                if depth.dtype == np.uint16 or depth.max() > 100:
                    depth = depth.astype(np.float32) / 1000.0
            except Exception:
                depth = _try_load_raw_float32_depth(depth_path, orig_size)
                if depth is None:
                    return
        else:
            return

        # Center crop and resize to match image
        orig_H, orig_W = orig_size
        short_side = min(orig_H, orig_W)
        top = (orig_H - short_side) // 2
        left = (orig_W - short_side) // 2

        if depth.shape[0] >= orig_H and depth.shape[1] >= orig_W:
            depth = depth[top : top + short_side, left : left + short_side]

        # Resize depth to IMG_SIZE
        from PIL import Image as PILImage

        depth_pil = PILImage.fromarray(depth)
        depth_pil = depth_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST)
        depth = np.array(depth_pil, dtype=np.float32)

        # Back-project to 3D
        H, W = depth.shape
        fx, fy = K_scaled[0, 0].item(), K_scaled[1, 1].item()
        cx, cy = K_scaled[0, 2].item(), K_scaled[1, 2].item()

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        X = (xx - cx) * depth / fx
        Y = (yy - cy) * depth / fy
        Z = depth

        pts_cam = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
        pts_cam_t = torch.from_numpy(pts_cam).float().to(device)

        # Camera to world
        R = c2w[:3, :3].to(device)
        t = c2w[:3, 3].to(device)
        pts_world = (R @ pts_cam_t.reshape(-1, 3).T).T + t
        pts3d = pts_world.reshape(H, W, 3)

        view["pts3d"] = pts3d.unsqueeze(0)  # (1, H, W, 3)
        view["depthmap"] = depth
    except Exception:
        pass


def _read_pfm(path):
    """Read a PFM file and return a float32 numpy array."""
    with open(path, "rb") as f:
        f.readline()  # PFM header; dimensions and scale follow.
        dims = f.readline().decode("ascii").rstrip().split()
        W, H = int(dims[0]), int(dims[1])
        scale = float(f.readline().decode("ascii").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), endian + "f")
    data = data.reshape(H, W)
    data = np.flipud(data)
    return data.astype(np.float32)
