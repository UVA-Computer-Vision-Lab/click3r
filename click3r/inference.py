# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Modified by the Click3R authors: adapted from dust3r/inference.py.
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf


def _interleave_imgs(img1, img2):
    res = {}

    for key, value1 in img1.items():
        # camera_intrinsics_original (rotation augmentation metadata): a view
        # without it uses its camera_intrinsics.
        if key == "camera_intrinsics_original":
            if "camera_intrinsics_original" in img2:
                value2 = img2["camera_intrinsics_original"]
            else:
                value2 = img2["camera_intrinsics"]
            if isinstance(value1, torch.Tensor):
                res[key] = torch.stack((value1, value2), dim=1).flatten(0, 1)
            else:
                res[key] = value1  # Fallback
            continue

        # Skip scalar augmentation metadata (not tensors)
        if key == "augment_rotation_k":
            res[key] = value1  # Keep as scalar, don't interleave
            continue

        # Skip keys that don't exist in img2
        if key not in img2:
            res[key] = value1
            continue

        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        elif isinstance(value1, dict):
            # Handle nested dicts (e.g., sparse_prompts)
            value = {}
            for sub_key in value1.keys():
                sub_val1 = value1[sub_key]
                sub_val2 = value2[sub_key]
                if isinstance(sub_val1, torch.Tensor):
                    value[sub_key] = torch.stack((sub_val1, sub_val2), dim=1).flatten(
                        0, 1
                    )
                else:
                    value[sub_key] = [
                        x
                        for pair in zip(sub_val1, sub_val2, strict=False)
                        for x in pair
                    ]
        else:
            value = [x for pair in zip(value1, value2, strict=False) for x in pair]
        res[key] = value

    # Add keys from img2 that aren't in img1 (e.g., camera_intrinsics_original only in view2)
    for key in img2.keys():
        if key not in res:
            value2 = img2[key]
            if key == "camera_intrinsics_original":
                # Only img2 has it; img1 uses its camera_intrinsics.
                value1 = img1["camera_intrinsics"]
                if isinstance(value1, torch.Tensor) and isinstance(
                    value2, torch.Tensor
                ):
                    res[key] = torch.stack((value1, value2), dim=1).flatten(0, 1)
                else:
                    res[key] = value2  # Fallback
            elif key == "augment_rotation_k":
                res[key] = value2  # Keep as scalar
            else:
                res[key] = value2

    return res


def make_batch_symmetric(batch, flip_image_ids=True):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))

    # Fix image_ids for swapped pairs (odd indices).
    # _interleave_imgs alternates [orig, swapped, orig, swapped, ...].
    # Swapped pairs have their view roles reversed, so image_ids must flip 0<->1.
    # The released checkpoint was trained without this flip (flip_image_ids=False).
    if flip_image_ids:
        for view in (view1, view2):
            if "sparse_prompts" in view and "image_ids" in view["sparse_prompts"]:
                ids = view["sparse_prompts"]["image_ids"]
                ids[1::2] = 1 - ids[1::2]

    return view1, view2


def loss_of_one_batch(
    batch,
    model,
    criterion,
    device,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    flip_image_ids=True,
):
    view1, view2 = batch
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            # Handle nested dicts like sparse_prompts
            if isinstance(view[name], dict):
                for sub_key, sub_val in view[name].items():
                    if isinstance(sub_val, torch.Tensor):
                        view[name][sub_key] = sub_val.to(device, non_blocking=True)
            elif isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch, flip_image_ids)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=bool(use_amp) and torch.cuda.is_available(),
    ):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                criterion(view1, view2, pred1, pred2) if criterion is not None else None
            )

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f">> Inference with model on {len(pairs)} image pairs")
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(
            collate_with_cat(pairs[i : i + batch_size]), model, None, device
        )
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:
        # pts3d from my camera
        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred["pts3d_in_other_view"]  # return!

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    return scaling
