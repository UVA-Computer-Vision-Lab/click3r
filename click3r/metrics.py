# Portions adapted from DUSt3R (https://github.com/naver/dust3r),
# dust3r/cloud_opt/init_im_poses.py (pixel_grid, fast_pnp).
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Portions adapted from PoseDiffusion (https://github.com/facebookresearch/PoseDiffusion),
# pose_diffusion/util/metric.py (calculate_auc_np, compare_translation_by_angle).
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under CC BY-NC 4.0.
#
# Modified by the Click3R authors.
"""
Pose evaluation metrics adapted from PoseDiffusion for DUSt3R/Click3R.

Metrics:
- RRA@K: Relative Rotation Accuracy @ K degrees
- RTA@K: Relative Translation Accuracy @ K degrees
- mAA@K: mean Average Accuracy @ K degrees (AUC of max(RRA, RTA))

Pose estimation uses PnP (Perspective-n-Point) to estimate camera poses
from predicted 3D points, following the approach in dust3r/cloud_opt/init_im_poses.py
"""

import numpy as np
import torch
import cv2
from functools import lru_cache


@lru_cache(maxsize=128)
def _pixel_grid(H, W):
    """Cached pixel grid generation."""
    return np.mgrid[:W, :H].T.astype(np.float32)


def unrotate_pixel_grid_90deg(H_rot, W_rot, k):
    """
    Generate pixel grid in original (un-rotated) coordinates.

    When image was rotated k * 90° CCW:
    - Current image has dimensions (H_rot, W_rot)
    - For each pixel (x', y') in rotated image, compute (x, y) in original image

    Args:
        H_rot: Height of rotated image
        W_rot: Width of rotated image
        k: Number of 90-degree CCW rotations applied (1-3)

    Returns:
        pixels_orig: (H_rot, W_rot, 2) array of (x, y) coordinates in original frame
        H_orig, W_orig: Original image dimensions before rotation
    """
    k = k % 4
    if k == 0:
        return _pixel_grid(H_rot, W_rot), H_rot, W_rot

    # Generate pixel coordinates in rotated image
    y_rot, x_rot = np.mgrid[0:H_rot, 0:W_rot].astype(np.float32)

    # Compute original dimensions (un-rotate dimensions)
    # After k CCW rotations, dimensions swap k%2 times
    if k % 2 == 1:
        H_orig, W_orig = W_rot, H_rot
    else:
        H_orig, W_orig = H_rot, W_rot

    # Un-rotate pixel coordinates by applying k CW rotations
    # (inverse of k CCW rotations = k CW rotations = (4-k) CCW rotations)
    x, y = x_rot, y_rot
    cur_H, cur_W = H_rot, W_rot

    for _ in range(k):
        # One CCW step maps (x, y) -> (y, W-1-x), where W = cur_H is the width
        # before the step; its inverse is (x, y) = (cur_H-1-y', x').
        x, y = cur_H - 1 - y, x
        cur_H, cur_W = cur_W, cur_H

    # Stack into (H_rot, W_rot, 2) array
    pixels_orig = np.stack([x, y], axis=-1)

    return pixels_orig, H_orig, W_orig


def rotate_camera_pose_90deg(camera_pose, k, as_tensor=True):
    """
    Rotate camera pose by k * 90 degrees CCW to match image rotation augmentation.

    When an image is rotated 90° CCW:
    - Pixel (x, y) -> (y, W-1-x)
    - Camera's coordinate frame rotates with the image
    - Need to apply rotation to camera's orientation

    Args:
        camera_pose: (4, 4) camera-to-world transformation matrix
        k: Number of 90-degree CCW rotations (0-3)
        as_tensor: If True, return torch.Tensor; otherwise return numpy array

    Returns:
        Rotated camera pose (4, 4)
    """
    k = k % 4
    if k == 0:
        return camera_pose

    # Convert to numpy if needed
    if isinstance(camera_pose, torch.Tensor):
        pose = camera_pose.cpu().numpy()
        was_tensor = True
    else:
        pose = camera_pose.copy()
        was_tensor = False

    # Rotation matrix for k * 90 degrees CCW around Z-axis (optical axis)
    # This rotates the camera's X and Y axes while keeping Z fixed
    angle = k * np.pi / 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R_img = np.array(
        [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
    )

    # Apply rotation to camera orientation: R_new = R_old @ R_img
    # (rotating in camera frame to match image rotation)
    pose_rotated = pose.copy()
    pose_rotated[:3, :3] = pose[:3, :3] @ R_img

    if as_tensor or was_tensor:
        return torch.from_numpy(pose_rotated).float()
    return pose_rotated


def estimate_pose_from_pts3d(
    pts3d, focal=None, pp=None, conf=None, niter_PnP=10, K=None, augment_rotation_k=0
):
    """
    Estimate camera pose from predicted 3D points using PnP+RANSAC.
    Adapted from dust3r/cloud_opt/init_im_poses.py:fast_pnp

    Args:
        pts3d: (H, W, 3) predicted 3D points (in rotated frame if augmented)
        focal: focal length (used if K is None)
        pp: principal point (cx, cy), defaults to image center (used if K is None)
        conf: (H, W) confidence map, optional
        niter_PnP: number of RANSAC iterations
        K: (3, 3) camera intrinsics of the original (un-rotated) image
        augment_rotation_k: Number of 90° CCW rotations applied to image (0-3).
                           If non-zero, pixel coordinates will be un-rotated before PnP,
                           and resulting pose will be rotated to match augmented frame.

    Returns:
        camera_pose: (4, 4) cam-to-world transformation (in rotated frame if augmented), or None if failed
    """
    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.detach().cpu().numpy()
    if conf is not None and isinstance(conf, torch.Tensor):
        conf = conf.detach().cpu().numpy()

    H, W = pts3d.shape[:2]

    # Create mask from confidence or use all valid points
    if conf is not None:
        msk = conf > conf.mean()  # Use above-average confidence points
    else:
        msk = np.isfinite(pts3d).all(axis=-1)

    if msk.sum() < 4:
        return None  # Need at least 4 points for PnP

    # Get pixel coordinates (un-rotated if needed)
    if augment_rotation_k > 0:
        # Un-rotate pixel coordinates to original frame
        pixels, H_orig, W_orig = unrotate_pixel_grid_90deg(H, W, augment_rotation_k)
    else:
        pixels = _pixel_grid(H, W)
        H_orig, W_orig = H, W

    # Use provided K matrix or build from focal/pp
    if K is None:
        if focal is None:
            raise ValueError("Must provide either K matrix or focal length")
        if pp is None:
            pp = (W_orig / 2, H_orig / 2)  # Use original dimensions for pp
        # Pinhole intrinsics with one focal length
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])
    else:
        # Convert K to numpy if needed
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
        K = np.float32(K)

    try:
        success, R, T, inliers = cv2.solvePnPRansac(
            pts3d[msk],
            pixels[msk],
            K,
            None,
            iterationsCount=niter_PnP,
            reprojectionError=5,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success or inliers is None or len(inliers) < 4:
            return None

        # Convert Rodrigues to rotation matrix
        R = cv2.Rodrigues(R)[0]  # world-to-cam

        # Build world-to-cam pose
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.ravel()

        # Invert to get cam-to-world (in original frame)
        c2w = np.linalg.inv(w2c)

        # If augmentation was applied, rotate the pose to match rotated frame
        if augment_rotation_k > 0:
            c2w = rotate_camera_pose_90deg(c2w, augment_rotation_k, as_tensor=False)

        return torch.from_numpy(c2w).float()

    except Exception:
        return None


def _extract_item(data, idx):
    """Extract item from data, handling tensors, lists, and numpy arrays."""
    if isinstance(data, (list, tuple)):
        return data[idx]
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        return data[idx]
    else:
        return data


def compute_pose_metrics_from_dust3r(
    view1, view2, pred1, pred2, pair_idx=None, augment_rotation_k=0, return_poses=False
):
    """
    Compute pose metrics from DUSt3R/Click3R/VGGT predictions.

    For models that directly predict poses (like VGGT), uses predicted poses directly.
    For models that only predict 3D points (like DUSt3R/Click3R), estimates poses via PnP.

    Args:
        view1, view2: Dict containing ground truth:
            - 'camera_pose': (B, 4, 4) or list of (4, 4) camera-to-world transformations
            - 'camera_intrinsics': (B, 3, 3) or list of (3, 3) camera intrinsics
        pred1, pred2: Dict containing predictions:
            - 'pts3d': (B, H, W, 3) or (H, W, 3) predicted 3D points
            - 'conf': (B, H, W) or (H, W) confidence (optional)
            - 'camera_pose_predicted': (B, 4, 4) or (4, 4) predicted cam2world (optional, for VGGT)
        pair_idx: Optional index to extract from batch (if inputs are batched)
        augment_rotation_k: Number of 90-degree CCW rotations applied to view2 (0-3)
                           If non-zero, GT camera pose for view2 will be rotated to match
        return_poses: If True, return predicted camera poses as well

    Returns:
        If return_poses=False:
            rotation_error (float): Relative rotation angle error in degrees
            translation_error (float): Relative translation angle error in degrees
        If return_poses=True:
            rotation_error (float): Relative rotation angle error in degrees
            translation_error (float): Relative translation angle error in degrees
            pred_c2w_1 (torch.Tensor): Predicted cam1-to-world pose (4x4)
            pred_c2w_2 (torch.Tensor): Predicted cam2-to-world pose (4x4)
    """
    # Extract single pair if batched
    # pred1 has 'pts3d', pred2 has 'pts3d_in_other_view'
    if pair_idx is not None:
        gt_c2w_1 = _extract_item(view1["camera_pose"], pair_idx)
        gt_c2w_2 = _extract_item(view2["camera_pose"], pair_idx)
        # Use original intrinsics if available (for augmented data), else use current intrinsics
        gt_K1 = _extract_item(
            view1.get("camera_intrinsics_original", view1["camera_intrinsics"]),
            pair_idx,
        )
        gt_K2 = _extract_item(
            view2.get("camera_intrinsics_original", view2["camera_intrinsics"]),
            pair_idx,
        )
        pred_pts3d_1 = _extract_item(pred1["pts3d"], pair_idx)
        pred_pts3d_2 = _extract_item(pred2["pts3d_in_other_view"], pair_idx)
        pred_conf_1 = (
            _extract_item(pred1["conf"], pair_idx) if "conf" in pred1 else None
        )
        pred_conf_2 = (
            _extract_item(pred2["conf"], pair_idx) if "conf" in pred2 else None
        )
    else:
        gt_c2w_1 = view1["camera_pose"]
        gt_c2w_2 = view2["camera_pose"]
        # Use original intrinsics if available (for augmented data), else use current intrinsics
        gt_K1 = view1.get("camera_intrinsics_original", view1["camera_intrinsics"])
        gt_K2 = view2.get("camera_intrinsics_original", view2["camera_intrinsics"])
        pred_pts3d_1 = pred1["pts3d"]
        pred_pts3d_2 = pred2["pts3d_in_other_view"]
        pred_conf_1 = pred1.get("conf")
        pred_conf_2 = pred2.get("conf")

    # Convert GT poses to tensors if needed
    if isinstance(gt_c2w_1, np.ndarray):
        gt_c2w_1 = torch.from_numpy(gt_c2w_1).float()
    if isinstance(gt_c2w_2, np.ndarray):
        gt_c2w_2 = torch.from_numpy(gt_c2w_2).float()

    # Ensure on CPU for consistency
    if gt_c2w_1.is_cuda:
        gt_c2w_1 = gt_c2w_1.cpu()
    if gt_c2w_2.is_cuda:
        gt_c2w_2 = gt_c2w_2.cpu()

    # Apply rotation to view2's GT pose if augmentation was used
    # This aligns the GT pose with the rotated image/predictions
    if augment_rotation_k > 0:
        gt_c2w_2 = rotate_camera_pose_90deg(
            gt_c2w_2, augment_rotation_k, as_tensor=True
        )

    # Ensure intrinsics are on CPU
    if isinstance(gt_K1, torch.Tensor):
        gt_K1 = gt_K1.cpu()
    if isinstance(gt_K2, torch.Tensor):
        gt_K2 = gt_K2.cpu()

    # Check if model provides direct pose predictions (like VGGT)
    has_direct_poses = (
        "camera_pose_predicted" in pred1 and "camera_pose_predicted" in pred2
    )

    if has_direct_poses:
        # Use direct pose predictions from model (VGGT path)
        pred_c2w_1 = (
            _extract_item(pred1["camera_pose_predicted"], pair_idx)
            if pair_idx is not None
            else pred1["camera_pose_predicted"]
        )
        pred_c2w_2 = (
            _extract_item(pred2["camera_pose_predicted"], pair_idx)
            if pair_idx is not None
            else pred2["camera_pose_predicted"]
        )

        # Convert to tensors if needed
        if isinstance(pred_c2w_1, np.ndarray):
            pred_c2w_1 = torch.from_numpy(pred_c2w_1).float()
        if isinstance(pred_c2w_2, np.ndarray):
            pred_c2w_2 = torch.from_numpy(pred_c2w_2).float()

        # Ensure on CPU
        if pred_c2w_1.is_cuda:
            pred_c2w_1 = pred_c2w_1.cpu()
        if pred_c2w_2.is_cuda:
            pred_c2w_2 = pred_c2w_2.cpu()
    else:
        # Estimate camera poses from predicted 3D points with PnP (DUSt3R/Click3R path).
        # For a rotated view 2, PnP un-rotates the pixel grid and rotates the pose back.
        pred_c2w_1 = estimate_pose_from_pts3d(
            pred_pts3d_1, conf=pred_conf_1, K=gt_K1, augment_rotation_k=0
        )  # view1 never rotated
        pred_c2w_2 = estimate_pose_from_pts3d(
            pred_pts3d_2,
            conf=pred_conf_2,
            K=gt_K2,
            augment_rotation_k=augment_rotation_k,
        )  # view2 may be rotated

        if pred_c2w_1 is None or pred_c2w_2 is None:
            raise ValueError("Failed to estimate camera poses from predicted 3D points")

    # Compute relative poses from camera1 to camera2
    # rel_pose = c2w_1^{-1} @ c2w_2: camera 2's pose in camera 1's frame (cam2-to-cam1)
    gt_rel_pose = torch.inverse(gt_c2w_1) @ gt_c2w_2
    pred_rel_pose = torch.inverse(pred_c2w_1) @ pred_c2w_2

    # Extract rotation and translation components
    gt_R_rel = gt_rel_pose[:3, :3]
    gt_t_rel = gt_rel_pose[:3, 3]  # Translation is in first 3 rows, last column

    pred_R_rel = pred_rel_pose[:3, :3]
    pred_t_rel = pred_rel_pose[:3, 3]

    # Compute rotation angle error
    rotation_error = rotation_angle_single(gt_R_rel, pred_R_rel)

    # Compute translation angle error
    translation_error = translation_angle_single(
        gt_t_rel, pred_t_rel, handle_ambiguity=True
    )

    if return_poses:
        return rotation_error.item(), translation_error.item(), pred_c2w_1, pred_c2w_2
    else:
        return rotation_error.item(), translation_error.item()


def closed_form_inverse_single(se3):
    """
    Computes the inverse of a single 4x4 SE3 matrix.

    For a transformation matrix:
    [R  t]
    [0  1]

    The inverse is:
    [R^T  -R^T @ t]
    [0    1       ]

    Args:
        se3 (Tensor): 4x4 SE3 transformation matrix

    Returns:
        Tensor: 4x4 inverted SE3 matrix
    """
    R = se3[:3, :3]
    t = se3[:3, 3]  # Translation is in the last column, first 3 rows

    # Compute the transpose of the rotation
    R_T = R.T

    # Compute the translation part of the inverse: -R^T @ t
    t_inv = -R_T @ t

    # Construct the inverse matrix
    inverted = torch.eye(4, dtype=se3.dtype, device=se3.device)
    inverted[:3, :3] = R_T
    inverted[:3, 3] = t_inv

    return inverted


def rotation_angle_single(rot_gt, rot_pred):
    """
    Compute rotation angle error between two 3x3 rotation matrices.
    Uses the formula: angle = arccos((trace(R1^T @ R2) - 1) / 2)

    Args:
        rot_gt (Tensor): 3x3 ground truth rotation matrix
        rot_pred (Tensor): 3x3 predicted rotation matrix

    Returns:
        Tensor: Rotation angle error in degrees
    """
    # Compute relative rotation: R_rel = R_gt^T @ R_pred
    R_rel = rot_gt.T @ rot_pred

    # Compute trace
    trace = torch.trace(R_rel)

    # Compute angle using: angle = arccos((trace(R) - 1) / 2)
    # Clamp to avoid numerical issues with arccos
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle_rad = torch.acos(cos_angle)
    angle_deg = angle_rad * 180.0 / np.pi

    return angle_deg


def translation_angle_single(tvec_gt, tvec_pred, eps=1e-15, handle_ambiguity=True):
    """
    Compute translation angle error between two translation vectors.
    Normalizes vectors and computes angle between them.

    Args:
        tvec_gt (Tensor): 3D ground truth translation vector
        tvec_pred (Tensor): 3D predicted translation vector
        eps (float): Small value to avoid division by zero
        handle_ambiguity (bool): If True, handle translation direction ambiguity
                                 by taking min(angle, 180-angle). This accounts
                                 for the fact that translation direction can be
                                 ambiguous in relative pose estimation. Default: True.

    Returns:
        Tensor: Translation angle error in degrees
    """
    # Normalize translation vectors
    t_norm = torch.norm(tvec_pred)
    t_normalized = tvec_pred / (t_norm + eps)

    t_gt_norm = torch.norm(tvec_gt)
    t_gt_normalized = tvec_gt / (t_gt_norm + eps)

    # Compute angle between normalized vectors
    cos_angle = torch.clamp(torch.sum(t_normalized * t_gt_normalized), -1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    angle_deg_raw = angle_rad * 180.0 / np.pi

    # Handle translation direction ambiguity (following VGGT's evaluation)
    # Translation direction can be ambiguous, so we take the minimum of
    # the angle and its supplement (180 - angle)
    if handle_ambiguity:
        angle_deg = torch.min(angle_deg_raw, torch.abs(180.0 - angle_deg_raw))
    else:
        angle_deg = angle_deg_raw

    return angle_deg


def calculate_auc_np(r_errors, t_errors, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for pose errors.
    This is the mAA metric: mean Average Accuracy.

    Args:
        r_errors: List or array of rotation errors in degrees
        t_errors: List or array of translation errors in degrees
        max_threshold: Maximum threshold for binning (default: 30)

    Returns:
        float: AUC value (0-1 range, multiply by 100 for percentage)
    """
    r_errors = np.array(r_errors)
    t_errors = np.array(t_errors)

    # Concatenate errors and take maximum for each pair
    error_matrix = np.stack((r_errors, t_errors), axis=1)
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins [0, 1, 2, ..., max_threshold]
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute cumulative sum (this is the accuracy curve)
    cumulative_accuracy = np.cumsum(normalized_histogram)

    # Return the mean (area under the curve)
    return np.mean(cumulative_accuracy)


def compute_accuracy_at_threshold(errors, threshold):
    """
    Compute accuracy at a specific threshold.

    Args:
        errors: List or array of errors in degrees
        threshold: Threshold value in degrees

    Returns:
        float: Percentage of samples with error <= threshold
    """
    errors = np.array(errors)
    return np.mean(errors <= threshold) * 100
