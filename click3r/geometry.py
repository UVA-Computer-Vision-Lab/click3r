"""Dataset construction, camera projection, and prompt sampling for Click3R training.

Camera poses use the DUSt3R camera-to-world convention.
"""

import torch

from dust3r.datasets import get_data_loader


# build_dataset is adapted from DUSt3R dust3r/training.py (https://github.com/naver/dust3r).
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only). Modified by the Click3R authors.
def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test,
    )
    print(f"{split} dataset length: ", len(loader))
    return loader


def _to_hom(X):
    """(...,3) -> (...,4) homogeneous coordinates."""
    ones = torch.ones_like(X[..., :1])
    return torch.cat([X, ones], dim=-1)


def _ensure_true_shape(view: dict) -> torch.Tensor:
    if "true_shape" in view and view["true_shape"] is not None:
        return view["true_shape"]
    B, _, H, W = view["img"].shape
    return torch.tensor([H, W], device=view["img"].device).view(1, 2).repeat(B, 1)


def _get_K_and_w2c(view: dict):
    """Extract camera params from DUSt3R's standard format.

    DUSt3R datasets provide:
        camera_intrinsics: (B,3,3) or (3,3)
        camera_pose:       (B,4,4) or (4,4) -- assumed c2w (camera-to-world).
    Returns (K(B,3,3), w2c(B,4,4)) or (None, None) if missing.
    """
    K = view.get("camera_intrinsics", None)
    if K is None:
        return None, None
    if K.dim() == 2:
        K = K.unsqueeze(0)

    c2w = view.get("camera_pose", None)
    if c2w is None:
        return None, None
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)

    def _to44(M):
        if M.shape[-2:] == (3, 4):
            tail = torch.zeros(M.shape[0], 1, 4, dtype=M.dtype, device=M.device)
            tail[:, 0, 3] = 1.0
            return torch.cat([M, tail], dim=1)
        return M

    c2w = _to44(c2w)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3:4]
    Rinv = R.transpose(1, 2)
    tinv = -Rinv @ t
    w2c = (
        torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        .unsqueeze(0)
        .repeat(c2w.shape[0], 1, 1)
    )
    w2c[:, :3, :3] = Rinv
    w2c[:, :3, 3:4] = tinv

    return K, w2c


def _project_points(K, w2c, pts3d_world):
    """K:(B,3,3), w2c:(B,4,4) or (B,3,4), pts3d_world:(B,M,3) -> uv:(B,M,2), z:(B,M)."""
    if w2c.shape[-2:] == (3, 4):
        w2c44 = torch.cat([w2c, w2c.new_zeros(w2c.shape[0], 1, 4)], dim=1)
        w2c44[:, 3, 3] = 1.0
        w2c = w2c44
    Xw = _to_hom(pts3d_world)
    Xc = (w2c @ Xw.transpose(1, 2)).transpose(1, 2)
    x, y, z = Xc[..., 0], Xc[..., 1], Xc[..., 2]
    z = z.clamp(min=1e-6)
    xn = x / z
    yn = y / z
    ones = torch.ones_like(xn)
    uvh = torch.stack([xn, yn, ones], dim=-1)
    uv = (K @ uvh.transpose(1, 2)).transpose(1, 2)[..., :2]
    return uv, z


def _valid_mask(uv, z, shape_b2, margin: int = 0):
    """uv:(B,M,2) pixels, z:(B,M), shape_b2:(B,2)=(H,W) -> bool mask (B,M)."""
    B, M, _ = uv.shape
    H = shape_b2[:, 0].view(B, 1).float()
    W = shape_b2[:, 1].view(B, 1).float()
    x = uv[..., 0]
    y = uv[..., 1]
    return (
        (z > 0)
        & (x >= margin)
        & (y >= margin)
        & (x <= (W - 1 - margin))
        & (y <= (H - 1 - margin))
    )


def _sample_num_pairs() -> int:
    """Per-sample random number of prompt pairs, distribution:
    30% chance: 0 pairs
    40% chance: 1-4 pairs (uniform)
    30% chance: 5-12 pairs (uniform)
    """
    r = torch.rand(1).item()
    if r < 0.3:
        return 0
    elif r < 0.7:
        return torch.randint(1, 5, (1,)).item()
    else:
        return torch.randint(5, 13, (1,)).item()
