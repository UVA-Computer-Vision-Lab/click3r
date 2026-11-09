"""Table 2 baselines: MASt3R (512), VGGT (224/518) and Pi3 (224/518).

Adapters that run these baselines under the Hard3DBench protocol:

- Inputs are square crops at ``Baseline.resolution`` in DUSt3R's [-1, 1] range,
  read by data.py at that size. 2-view samples load with ``center_crop``:
  Out-of-Distribution images use `data.center_crop_view`, In-Distribution views
  the DUSt3R dataset crop at that resolution (the Click3R reader, cropped at R
  instead of 224), or with ``whole_frame`` (their pointmap metric)
  `data.center_crop_view` with unmasked depth. 5-view tuples use the Click3R
  loaders at that resolution.
- MASt3R predicts pointmaps like DUSt3R and runs through the DUSt3R code: 2-view
  poses from PnP on the symmetrized prediction (intrinsics from `data.pnp_intrinsics`
  for Out-of-Distribution pairs), 5-view tuples through the complete-graph global
  alignment without prompts.
- VGGT and Pi3 predict cameras: `predict_views` returns their camera-to-world
  poses and pointmaps in the first camera's frame. The 2-view VGGT pointmaps come
  from its depth head and predicted intrinsics, the 5-view ones from its point
  head, as in the paper's 2-view and 5-view runs.
- VGGT and Pi3 run with TF32 matmuls off: their Table 2 runs never imported CroCo,
  whose import turns TF32 on for the whole process.
- ``Baseline.precision`` overrides the shared autocast dtype on GPUs for a
  baseline whose paper runs used another one (None keeps the shared
  bfloat16/float16 rule).
"""

import json
import sys
from typing import NamedTuple

import numpy as np
import torch

from click3r.metrics import rotation_angle_single, translation_angle_single
from evaluation.hard3dbench import common

PINS = common.REPO_ROOT / "click3r/configs/baseline_checkpoints.json"
MAST3R = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


class Baseline(NamedTuple):
    family: str
    resolution: int
    weights: str
    precision: str | None = None


BASELINES = {
    "mast3r_512": Baseline("mast3r", 512, MAST3R, "float16"),
    "vggt_224": Baseline("vggt", 224, "facebook/VGGT-1B"),
    "vggt_518": Baseline("vggt", 518, "facebook/VGGT-1B"),
    "pi3_224": Baseline("pi3", 224, "yyfz233/Pi3"),
    "pi3_518": Baseline("pi3", 518, "yyfz233/Pi3"),
}
CAMERA_FAMILIES = ("vggt", "pi3")


def load_baseline(key, device):
    """Load a Table 2 baseline at its pinned Hugging Face revision, in eval mode."""
    family, weights = BASELINES[key].family, BASELINES[key].weights
    if family == "mast3r":
        mast3r = str(common.REPO_ROOT / "third_party" / "mast3r")
        if mast3r not in sys.path:
            sys.path.insert(0, mast3r)
        from evaluation.baselines.loading import load_mast3r_model

        return load_mast3r_model(weights, device)
    if family == "vggt":
        from evaluation.baselines.vggt import VGGTWrapper

        return VGGTWrapper(weights, device).model
    from evaluation.baselines.pi3 import load_pi3_model

    return load_pi3_model(weights, device)


def weights_metadata(key):
    """Result metadata of a baseline's weights: the pinned revision and sha256."""
    weights = BASELINES[key].weights
    pin = json.loads(PINS.read_text())[weights]
    return {
        "checkpoint": weights,
        "checkpoint_revision": pin["revision"],
        "checkpoint_sha256": pin["files"]["model.safetensors"]["sha256"],
    }


def precision_for(key, device):
    """Autocast dtype: the baseline's Table 2 override on GPUs, else the shared rule."""
    override = BASELINES[key].precision
    if override and device.type == "cuda":
        return override
    return common.precision_for(device)


def _forward_cameras(family, model, images, precision, depth_head=False):
    """Native multi-view forward. images: (B, N, 3, R, R) in [0, 1].

    Returns camera-to-world poses (B, N, 4, 4) as a CPU tensor and world points
    (B, N, R, R, 3) as a float32 array: the point head, or for VGGT with
    ``depth_head`` its depth map unprojected with the predicted intrinsics.
    """
    device = next(model.parameters()).device
    images = images.to(device)
    dtype = getattr(torch, precision)
    autocast = torch.amp.autocast(
        device.type, dtype=dtype, enabled=dtype != torch.float32
    )
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad(), autocast:
        pred = model(images)
    if family == "pi3":
        cam2world = pred["camera_poses"].float().cpu()
        torch.backends.cuda.matmul.allow_tf32 = tf32
        return cam2world, pred["points"].float().cpu().numpy()
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    height, width = images.shape[-2:]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pred["pose_enc"].float(), image_size_hw=(height, width)
    )
    world_to_camera = torch.zeros(
        *extrinsics.shape[:2], 4, 4, device=extrinsics.device, dtype=extrinsics.dtype
    )
    world_to_camera[:, :, :3, :] = extrinsics
    world_to_camera[:, :, 3, 3] = 1.0
    cam2world = torch.linalg.inv(world_to_camera).cpu()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if not depth_head:
        return cam2world, pred["world_points"].float().cpu().numpy()
    depth = pred["depth"][..., 0].float().cpu().numpy().astype(np.float64)
    K = intrinsics.float().cpu().numpy()
    rows, cols = np.mgrid[:height, :width]
    x = (cols - K[..., 0, 2, None, None]) * depth / K[..., 0, 0, None, None]
    y = (rows - K[..., 1, 2, None, None]) * depth / K[..., 1, 1, None, None]
    camera = np.stack([x, y, depth], axis=-1)
    pose = cam2world.numpy().astype(np.float64)
    world = np.einsum("bnij,bnhwj->bnhwi", pose[..., :3, :3], camera)
    world = world + pose[..., None, None, :3, 3]
    return cam2world, world.astype(np.float32)


def predict_views(key, model, images, precision, depth_head):
    """Predict one sample with VGGT or Pi3. images: (N, 3, R, R) in [-1, 1].

    Returns ``cam2world`` (N, 4, 4) and ``pts3d`` (N, R, R, 3), all views in the
    first camera's frame (float32); VGGT points come from its depth head with
    ``depth_head``, else from its point head.
    """
    family = BASELINES[key].family
    cam2world, world = _forward_cameras(
        family, model, (images * 0.5 + 0.5)[None], precision, depth_head=depth_head
    )
    cam2world, world = cam2world[0].numpy(), world[0]
    first = np.linalg.inv(cam2world[0])
    points = [view @ first[:3, :3].T + first[:3, 3] for view in world]
    return {"cam2world": cam2world, "pts3d": np.stack(points)}


def camera_pose_error(cam2world, gt_c2w_1, gt_c2w_2):
    """RRE and RTE (degrees) of a predicted camera pair, in float32 as for Table 2."""
    pred = torch.as_tensor(cam2world, dtype=torch.float32)
    gt_rel = torch.inverse(torch.as_tensor(gt_c2w_1, dtype=torch.float32)) @ (
        torch.as_tensor(gt_c2w_2, dtype=torch.float32)
    )
    pred_rel = torch.inverse(pred[0]) @ pred[1]
    rre = rotation_angle_single(gt_rel[:3, :3], pred_rel[:3, :3]).item()
    rte = translation_angle_single(
        gt_rel[:3, 3], pred_rel[:3, 3], handle_ambiguity=True
    ).item()
    return rre, rte
