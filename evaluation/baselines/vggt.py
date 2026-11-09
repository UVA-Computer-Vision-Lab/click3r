"""
VGGT wrapper with DUSt3R's (view1, view2) -> (res1, res2) interface.

Used for the VGGT rows of Table 1 and in the demo; Table 2 runs VGGT through
evaluation/hard3dbench/baselines.py.
"""

import torch
import torch.nn as nn
import sys
import os

# VGGT is fetched into third_party/vggt by scripts/setup_environment.py --baselines.
vggt_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "third_party", "vggt"
)
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTWrapper(nn.Module):
    """
    Wrapper to make VGGT compatible with DUSt3R interface.

    Key differences:
    - VGGT expects images in [0, 1] range with shape [B, S, 3, H, W] or [S, 3, H, W]
    - DUSt3R uses ImgNorm (mean=0.5, std=0.5) resulting in [-1, 1] range
    - VGGT outputs dict with 'world_points' and 'world_points_conf'
    - DUSt3R outputs tuple (res1, res2) with 'pts3d' and 'conf'
    """

    def __init__(self, model_name="facebook/VGGT-1B", device="cuda"):
        super().__init__()
        self.device = device

        print(f"Loading VGGT model: {model_name}")
        from evaluation.baselines.checkpoints import hub_options

        self.model = VGGT.from_pretrained(model_name, **hub_options(model_name)).to(
            device
        )
        self.model.eval()

    def denormalize_image(self, img_tensor):
        """
        Denormalize from DUSt3R's ImgNorm format to VGGT's [0, 1] range.

        DUSt3R uses: Normalize(mean=0.5, std=0.5) which gives [-1, 1] range
        VGGT expects: [0, 1] range

        Conversion: (img * 0.5) + 0.5
        """
        return (img_tensor * 0.5) + 0.5

    def forward(self, view1, view2):
        """
        Forward pass compatible with DUSt3R interface.

        Args:
            view1 (dict): First view with keys:
                - 'img': torch.Tensor [B, 3, H, W] in [-1, 1] range (ImgNorm)
                - 'true_shape': Original image shape
                - 'idx': Image index
                - 'instance': Image identifier
            view2 (dict): Second view with same structure

        Returns:
            tuple: (res1, res2) where each res is a dict with:
                - 'pts3d': torch.Tensor [B, H, W, 3] - 3D points
                - 'conf': torch.Tensor [B, H, W] - confidence scores
        """
        img1 = view1["img"]  # [B, 3, H, W]
        img2 = view2["img"]  # [B, 3, H, W]

        B, C, H, W = img1.shape

        img1_denorm = self.denormalize_image(img1)
        img2_denorm = self.denormalize_image(img2)

        images = torch.stack([img1_denorm, img2_denorm], dim=1)  # [B, 2, 3, H, W]

        with torch.no_grad():
            predictions = self.model(images)

        world_points = predictions["world_points"]  # [B, 2, H, W, 3]
        world_points_conf = predictions["world_points_conf"]  # [B, 2, H, W]
        pose_enc = predictions["pose_enc"]  # [B, 2, 9]

        # VGGT's extrinsics are camera-from-world (world->camera), shape [B, S, 3, 4];
        # they are inverted below to cam2world.
        extrinsics, _ = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W), build_intrinsics=False
        )

        batch_size, seq_len = extrinsics.shape[:2]
        extrinsics_4x4 = torch.zeros(
            batch_size, seq_len, 4, 4, device=extrinsics.device, dtype=extrinsics.dtype
        )
        extrinsics_4x4[:, :, :3, :] = extrinsics
        extrinsics_4x4[:, :, 3, 3] = 1.0

        camera_poses = torch.linalg.inv(extrinsics_4x4)  # [B, S, 4, 4]

        camera_pose_1 = camera_poses[:, 0]  # [B, 4, 4] cam1-to-world
        camera_pose_2 = camera_poses[:, 1]  # [B, 4, 4] cam2-to-world

        # DUSt3R's loss expects pts3d in each view's camera frame, not the world frame.
        from dust3r.utils.geometry import geotrf

        world_to_cam1 = torch.linalg.inv(camera_pose_1)  # [B, 4, 4]
        world_to_cam2 = torch.linalg.inv(camera_pose_2)  # [B, 4, 4]

        pts3d_1_world = world_points[:, 0]  # [B, H, W, 3] in world frame
        pts3d_2_world = world_points[:, 1]  # [B, H, W, 3] in world frame

        pts3d_1 = geotrf(world_to_cam1, pts3d_1_world)  # Transform to cam1 frame
        pts3d_2 = geotrf(world_to_cam2, pts3d_2_world)  # Transform to cam2 frame

        conf_1 = world_points_conf[:, 0]  # [B, H, W]
        conf_2 = world_points_conf[:, 1]  # [B, H, W]

        # Relative pose camera 2 -> camera 1: it maps view 2's points into camera 1's frame
        # (pts3d_in_other_view), the points DUSt3R's loss compares.
        cam2_to_cam1 = world_to_cam1 @ camera_pose_2

        # Transform view2 points to view1 frame for pose metrics
        pts3d_2_in_cam1 = geotrf(cam2_to_cam1, pts3d_2)

        res1 = {
            "pts3d": pts3d_1,
            "conf": conf_1,
            "camera_pose": camera_pose_1,
            "camera_pose_predicted": camera_pose_1,
            "pts3d_world": pts3d_1_world,
        }

        res2 = {
            "pts3d": pts3d_2,
            "conf": conf_2,
            "pts3d_in_other_view": pts3d_2_in_cam1,
            "camera_pose": cam2_to_cam1,
            "camera_pose_predicted": camera_pose_2,
            "pts3d_world": pts3d_2_world,
        }

        return res1, res2

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self


def load_vggt_model(model_name="facebook/VGGT-1B", device="cuda"):
    """
    Convenience function to load VGGT model with DUSt3R-compatible wrapper.

    Args:
        model_name (str): Hugging Face model name or path
        device (str): Device to load model on

    Returns:
        VGGTWrapper: Wrapped VGGT model
    """
    return VGGTWrapper(model_name=model_name, device=device)
