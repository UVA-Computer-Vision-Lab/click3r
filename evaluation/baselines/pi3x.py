"""
Pi3X wrapper with DUSt3R's (view1, view2) -> (res1, res2) interface.

Used for the Pi3X rows of Table 1 and in the demo.
"""

import torch
import torch.nn as nn
import sys
import os

# scripts/setup_environment.py --baselines fetches Pi3 and Pi3X into third_party/Pi3.
pi3_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "third_party", "Pi3"
)
if pi3_path not in sys.path:
    sys.path.insert(0, pi3_path)

# Pi3X is a separate architecture from Pi3 (Table 2, evaluation/baselines/pi3.py).
from pi3.models.pi3x import Pi3X


def patch_curope_for_pi3x():
    """
    Patch cuRoPE2D class to ensure tensor contiguity for Pi3.

    Pi3's FlashAttentionRope applies q_norm() and .to(dtype) operations before RoPE,
    which can create non-contiguous tensors. The cuRoPE2D module does tokens.transpose(1,2)
    which preserves non-contiguity, causing the CUDA kernel to fail.

    This patches the cuRoPE2D class method before model instantiation.
    """
    # Pi3 imports cuRoPE2D as models.curope, from CroCo inside third_party/dust3r.
    dust3r_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "third_party",
        "dust3r",
    )
    if dust3r_path not in sys.path:
        sys.path.insert(0, dust3r_path)

    from models.curope.curope2d import cuRoPE2D, cuRoPE2D_func

    if hasattr(cuRoPE2D, "_pi3_patched"):
        return

    # Kept so that loading.restore_curope_class_patch can undo the patch.
    cuRoPE2D._original_class_forward = cuRoPE2D.forward

    def contiguous_forward(self, tokens, positions):
        """Ensure tokens are contiguous before calling cuRoPE2D kernel."""
        # The transpose is a non-contiguous view; the kernel needs a contiguous tensor.
        tokens_transposed = tokens.transpose(1, 2).contiguous()

        cuRoPE2D_func.apply(tokens_transposed, positions, self.base, self.F0)

        return tokens_transposed.transpose(1, 2)

    cuRoPE2D.forward = contiguous_forward
    cuRoPE2D._pi3_patched = True

    print("  Patched cuRoPE2D class for Pi3 contiguity safety")


class Pi3XWrapper(nn.Module):
    """
    Wrapper to make Pi3X compatible with DUSt3R interface.

    Key differences from DUSt3R:
    - Pi3X expects images in [0, 1] range with shape [B, N, 3, H, W]
    - DUSt3R uses ImgNorm (mean=0.5, std=0.5) resulting in [-1, 1] range
    - Pi3X outputs dict with 'local_points', 'conf' (logits), 'camera_poses'
    - DUSt3R outputs tuple (res1, res2) with 'pts3d' and 'conf'
    - Pi3X is permutation-equivariant (no reference view needed)
    """

    def __init__(self, model_name="yyfz233/Pi3X", device="cuda"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.patched_modules = []  # Track patched modules for cleanup

        patch_curope_for_pi3x()

        from evaluation.baselines.checkpoints import hub_options

        self.model = Pi3X.from_pretrained(model_name, **hub_options(model_name)).to(
            device
        )
        self.model.eval()

        self._patch_model_curope_instances()

    def _patch_model_curope_instances(self):
        """Patch all cuRoPE2D module instances in the loaded model."""
        dust3r_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "third_party",
            "dust3r",
        )
        if dust3r_path not in sys.path:
            sys.path.insert(0, dust3r_path)
        from models.curope.curope2d import cuRoPE2D_func

        # Find and patch all cuRoPE2D instances (check by class name)
        patched_count = 0
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "cuRoPE2D":
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward

                def make_contiguous_forward(curope_module):
                    def contiguous_forward(tokens, positions):
                        # Pi3's q_norm() and .to(dtype) can leave both non-contiguous.
                        tokens_transposed = tokens.transpose(1, 2).contiguous()
                        positions_contiguous = positions.contiguous()
                        cuRoPE2D_func.apply(
                            tokens_transposed,
                            positions_contiguous,
                            curope_module.base,
                            curope_module.F0,
                        )
                        return tokens_transposed.transpose(1, 2)

                    return contiguous_forward

                module.forward = make_contiguous_forward(module)
                patched_count += 1
                self.patched_modules.append(module)

        if patched_count > 0:
            print(
                f"  Patched {patched_count} cuRoPE2D module instance(s) for contiguity safety"
            )

    def __del__(self):
        """
        Cleanup method to restore original cuRoPE2D forward methods.
        This prevents "free(): invalid pointer" errors during Python exit.
        """
        for module in self.patched_modules:
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                delattr(module, "_original_forward")

    def denormalize_image(self, img_tensor):
        """
        Denormalize from DUSt3R's ImgNorm format to Pi3's [0, 1] range.

        DUSt3R uses: Normalize(mean=0.5, std=0.5) which gives [-1, 1] range
        Pi3 expects: [0, 1] range

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
                - 'pts3d': torch.Tensor [B, H, W, 3] - 3D points in camera frame
                - 'conf': torch.Tensor [B, H, W] - confidence scores [0, 1]
                - 'camera_pose': Camera pose (cam2world for view1, relative for view2)
                - 'camera_pose_predicted': Absolute camera pose from Pi3
                - 'pts3d_world': Global 3D points in world frame
        """
        img1 = view1["img"]  # [B, 3, H, W]
        img2 = view2["img"]  # [B, 3, H, W]

        B, C, H, W = img1.shape

        img1_denorm = self.denormalize_image(img1)
        img2_denorm = self.denormalize_image(img2)

        images = torch.stack([img1_denorm, img2_denorm], dim=1)  # [B, 2, 3, H, W]

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
            else torch.float16
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = self.model(images)

        # Pi3X output:
        #   - points: [B, N, H, W, 3] global world points
        #   - local_points: [B, N, H, W, 3] per-view points in each camera's frame
        #   - conf: [B, N, H, W, 1] raw confidence logits
        #   - camera_poses: [B, N, 4, 4] cam2world transformation matrices

        local_points = predictions["local_points"]  # [B, 2, H, W, 3]
        conf_logits = predictions["conf"]  # [B, 2, H, W, 1]
        camera_poses = predictions["camera_poses"]  # [B, 2, 4, 4]
        global_points = predictions["points"]  # [B, 2, H, W, 3]

        conf_probs = torch.sigmoid(conf_logits).squeeze(-1)  # [B, 2, H, W]

        pts3d_1 = local_points[:, 0]  # [B, H, W, 3] in cam1 frame
        pts3d_2 = local_points[:, 1]  # [B, H, W, 3] in cam2 frame

        conf_1 = conf_probs[:, 0]  # [B, H, W]
        conf_2 = conf_probs[:, 1]  # [B, H, W]

        camera_pose_1 = camera_poses[:, 0]  # [B, 4, 4] cam1-to-world
        camera_pose_2 = camera_poses[:, 1]  # [B, 4, 4] cam2-to-world

        pts3d_1_world = global_points[:, 0]  # [B, H, W, 3] in world frame
        pts3d_2_world = global_points[:, 1]  # [B, H, W, 3] in world frame

        # Relative pose camera 2 -> camera 1: it maps view 2's points into camera 1's frame
        # (pts3d_in_other_view), the points DUSt3R's loss compares.
        world_to_cam1 = torch.linalg.inv(camera_pose_1)
        cam2_to_cam1 = world_to_cam1 @ camera_pose_2

        # Transform view2 points to view1 frame for pose metrics
        from dust3r.utils.geometry import geotrf

        pts3d_2_in_cam1 = geotrf(cam2_to_cam1, pts3d_2)

        res1 = {
            "pts3d": pts3d_1,  # In cam1 frame (for loss)
            "conf": conf_1,
            "camera_pose": camera_pose_1,  # Absolute cam2world for reference
            "camera_pose_predicted": camera_pose_1,  # Direct pose prediction from Pi3
            "pts3d_world": pts3d_1_world,  # World-frame points for visualization
        }

        res2 = {
            "pts3d": pts3d_2,  # In cam2 frame (for loss computation)
            "conf": conf_2,
            "pts3d_in_other_view": pts3d_2_in_cam1,  # In cam1 frame (for pose metrics)
            "camera_pose": cam2_to_cam1,  # Relative transform for loss
            "camera_pose_predicted": camera_pose_2,  # Direct pose prediction from Pi3
            "pts3d_world": pts3d_2_world,  # World-frame points for visualization
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


def load_pi3x_model(model_name="yyfz233/Pi3X", device="cuda"):
    """Load the pinned Pi3X checkpoint with a DUSt3R-compatible interface."""
    return Pi3XWrapper(model_name=model_name, device=device)
