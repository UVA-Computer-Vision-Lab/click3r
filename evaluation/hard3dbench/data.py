"""Load Hard3DBench samples as square model inputs with ground truth.

In-Distribution frames are read from one `<name>_processed_for_dust3r` folder per
dataset (`indist_folders`): CO3D from the held-out CO3D test sequences of Table 1
(`--co3d-root`), BlendedMVS, Habitat and WildRGB-D from the processed training data
(`--data-root`). Frames are loaded by the names in the list, with the per-view code
of the DUSt3R loaders: image and depth reading, the CO3D/WildRGB-D object mask, and
the dataset crop. Loading by frame reproduces the evaluated views without
rebuilding the loaders' pair indices. With `whole_frame` they are cropped like
Out-of-Distribution frames instead, with depth not masked: the views of the
paper's In-Distribution pointmap column.

Out-of-Distribution frames are read from a prepared root (`--ood-root`) holding
`<dataset>/cameras.json` and the image and depth files it lists; images use the
DUSt3R load-and-crop, and depth and intrinsics follow the same crop.

Views are `size` x `size`: 224 by default, the input of Click3R and DUSt3R. The
2-view baselines load with `center_crop`: their whole frames (Out-of-Distribution
and `whole_frame` views) are resized on the short side and center-cropped
(`center_crop_view`), and every view also holds `pnp_K`, the intrinsics
MASt3R's PnP uses.
"""

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms.functional as tvf
from PIL import Image

from dust3r.utils.image import imread_cv2
from evaluation.hard3dbench.common import (
    IMAGE_SIZE,
    crop_depth,
    crop_intrinsics,
    crop_like_dust3r_dataset,
    load_image,
)

INDIST_ROOTS = {
    "BlendedMVS_val": "blendedmvs_processed_for_dust3r",
    "Co3d_test": "co3d_processed_for_dust3r",
    "Habitat_val": "habitat_processed_for_dust3r",
    "WildRGBD_test": "wildrgbd_processed_for_dust3r",
}

# Files of one In-Distribution frame relative to its dataset folder. `frame` is
# the on-disk stem used by the 2-view list (Habitat views count from 1).
INDIST_FILES = {
    "BlendedMVS_val": {
        "image": "{scene}/{frame}.jpg",
        "depth": "{scene}/{frame}.exr",
        "camera": "{scene}/{frame}.npz",
    },
    "Co3d_test": {
        "image": "{scene}/images/{frame}.jpg",
        "depth": "{scene}/depths/{frame}.jpg.geometric.png",
        "camera": "{scene}/images/{frame}.npz",
        "mask": "{scene}/masks/{frame}.png",
    },
    "WildRGBD_test": {
        "image": "{scene}/rgb/{frame}.jpg",
        "depth": "{scene}/depth/{frame}.png",
        "camera": "{scene}/metadata/{frame}.npz",
        "mask": "{scene}/masks/{frame}.png",
    },
    "Habitat_val": {
        "image": "{scene}/{frame}.jpeg",
        "depth": "{scene}/{frame}_depth.exr",
        "camera": "{scene}/{frame}_camera_params.json",
    },
}


def indist_folders(data_root, co3d_root):
    """Folder of each In-Distribution dataset.

    CO3D samples come from the held-out CO3D test sequences that
    scripts/table1/prepare_data.py builds in the Table 1 data root (`co3d_root`);
    the other datasets from the processed training data (`data_root`).
    """
    return {
        dataset: Path(co3d_root if dataset == "Co3d_test" else data_root) / folder
        for dataset, folder in INDIST_ROOTS.items()
    }


def center_crop_view(image_path, size):
    """Load a whole frame as for the 2-view baselines (`center_crop`).

    Resizes the short side to ``size`` (torchvision bilinear on PIL) and
    center-crops a square. Returns the image (1, 3, size, size) in [-1, 1] and the
    crop in the `load_image` format plus ``scale``; ground-truth depth and
    intrinsics follow from it as for `load_image`.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    x0, y0 = (new_width - size) // 2, (new_height - size) // 2
    image = tvf.resize(image, [new_height, new_width])
    image = tvf.crop(image, y0, x0, size, size)
    crop = {
        "orig_W": width,
        "orig_H": height,
        "resize_W": new_width,
        "resize_H": new_height,
        "crop_x0": x0,
        "crop_y0": y0,
        "final_W": size,
        "final_H": size,
        "scale": scale,
    }
    return (tvf.to_tensor(image) * 2 - 1).unsqueeze(0), crop


def pnp_intrinsics(intrinsics, crop):
    """MASt3R's PnP intrinsics on a `center_crop_view` frame: one scale for both axes.

    This differs from the per-axis ground-truth intrinsics by the rounding of the
    resized size; PnP-RANSAC is sensitive to it, so Table 2 keeps this form.
    """
    K = np.asarray(intrinsics, dtype=np.float64).copy()
    K[0, 0] *= crop["scale"]
    K[1, 1] *= crop["scale"]
    K[0, 2] = K[0, 2] * crop["scale"] - crop["crop_x0"]
    K[1, 2] = K[1, 2] * crop["scale"] - crop["crop_y0"]
    return K


def _load_frame(path, size, center_crop):
    """A whole frame resized and center-cropped, and its crop parameters."""
    if center_crop:
        return center_crop_view(path, size)
    return load_image(path, size)


def _frame_view(image, crop, depth, K, c2w, center_crop):
    """View of a whole frame: ground-truth depth and intrinsics on the image crop."""
    view = {
        "img": image,
        "depth": crop_depth(depth, crop),
        "K": crop_intrinsics(K, crop),
    }
    if center_crop:
        view["pnp_K"] = pnp_intrinsics(K, crop)
    view["c2w"] = c2w
    return view


# _read_indist_frame follows the per-view reading code of the DUSt3R loaders
# dust3r/datasets/{blendedmvs,co3d,wildrgbd,habitat}.py
# (https://github.com/naver/dust3r), bundled in third_party/dust3r.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Modified by the Click3R authors.


def _read_indist_frame(files, dataset, object_mask=True):
    """Image, depth, intrinsics and cam-to-world pose as the DUSt3R loaders read them."""
    if dataset == "Habitat_val":
        image = Image.open(files["image"])
        depth = cv2.imread(files["depth"], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        camera = json.loads(Path(files["camera"]).read_text())
        K = np.float32(camera["camera_intrinsics"])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = camera["R_cam2world"]
        c2w[:3, 3] = camera["t_cam2world"]
        return image, depth, K, c2w
    image = imread_cv2(files["image"])
    camera = np.load(files["camera"])
    if dataset in ("Co3d_test", "WildRGBD_test"):
        K = camera["camera_intrinsics"].astype(np.float32)
        c2w = camera["camera_pose"].astype(np.float32)
        depth = imread_cv2(files["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32)
        if dataset == "Co3d_test":
            depth = depth / 65535 * np.nan_to_num(camera["maximum_depth"])
        else:
            depth = depth / 1000.0
        if object_mask:
            mask = imread_cv2(files["mask"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth *= (mask / 255.0) > 0.1
        return image, depth, K, c2w
    depth = imread_cv2(files["depth"])
    K = np.float32(camera["intrinsics"])
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = camera["R_cam2world"]
    c2w[:3, 3] = camera["t_cam2world"]
    return image, depth, K, c2w


def load_indist_view(
    folders,
    dataset,
    scene,
    frame,
    size=IMAGE_SIZE,
    center_crop=False,
    whole_frame=False,
):
    """One In-Distribution frame: (1, 3, size, size) image, depth, K, cam-to-world.

    `folders` maps each dataset to its folder (`indist_folders`). By default the
    view is the DUSt3R dataset crop with object-masked depth (pose metric; with
    `center_crop`, `pnp_K` is K); with `whole_frame` it is the whole frame cropped
    like an Out-of-Distribution frame, with unmasked depth (pointmap metric).
    """
    root = Path(folders[dataset])
    files = {
        key: str(root / template.format(scene=scene, frame=frame))
        for key, template in INDIST_FILES[dataset].items()
    }
    image, depth, K, c2w = _read_indist_frame(files, dataset, not whole_frame)
    if whole_frame:
        image, crop = _load_frame(files["image"], size, center_crop)
        view = _frame_view(image, crop, depth, K, c2w, center_crop)
    else:
        image, depth, K = crop_like_dust3r_dataset(
            image, depth, K, size, info=files["image"]
        )
        view = {"img": image, "depth": depth, "K": K}
        if center_crop:
            view["pnp_K"] = K
        view["c2w"] = c2w
    if not (np.isfinite(c2w).all() and np.isfinite(view["depth"]).all()):
        raise ValueError(f"Non-finite pose or depth in {files['image']}")
    return view


def _pair(view1, view2):
    pair = {}
    for i, view in ((1, view1), (2, view2)):
        pair.update({f"{key}{i}": value for key, value in view.items()})
    return pair


def load_indist_pair(
    sample, folders, size=IMAGE_SIZE, center_crop=False, whole_frame=False
):
    dataset, scene = sample["dataset"], sample["scene"]
    views = [
        load_indist_view(
            folders,
            dataset,
            scene,
            sample[f"frame{i}"],
            size,
            center_crop,
            whole_frame,
        )
        for i in (1, 2)
    ]
    return _pair(*views)


@lru_cache(maxsize=None)
def ood_cameras(ood_root, dataset):
    """Read `<ood-root>/<dataset>/cameras.json` once per dataset."""
    return json.loads((Path(ood_root) / dataset / "cameras.json").read_text())["frames"]


def read_ood_depth(path, dataset, size):
    """Ground-truth depth of a prepared Out-of-Distribution frame (float32)."""
    if dataset == "eth3d":
        depth = np.fromfile(path, dtype="<f4")
        width, height = size
        if depth.size != width * height:
            raise ValueError(f"{path}: expected {width}x{height} float32 depth")
        return depth.reshape(height, width).astype(np.float32)
    if dataset == "dtu":
        return np.load(path).astype(np.float32)
    depth = np.array(Image.open(path))
    if depth.dtype != np.uint16:
        raise ValueError(f"{path}: expected a 16-bit depth PNG, got {depth.dtype}")
    return depth.astype(np.float32) / 1000.0


def load_ood_view(ood_root, dataset, scene, frame, size=IMAGE_SIZE, center_crop=False):
    """One Out-of-Distribution frame: (1, 3, size, size) image, depth, K, cam-to-world.

    The image uses DUSt3R's load-and-crop, or with `center_crop` `center_crop_view`
    (the view then also holds `pnp_K`); depth and intrinsics follow the crop.
    """
    record = ood_cameras(str(ood_root), dataset)[f"{scene}/{frame}"]
    image, crop = _load_frame(Path(ood_root) / record["image"], size, center_crop)
    if [crop["orig_W"], crop["orig_H"]] != list(record["size"]):
        raise ValueError(f"{record['image']}: size differs from cameras.json")
    depth = read_ood_depth(Path(ood_root) / record["depth"], dataset, record["size"])
    c2w = np.array(record["cam2world"], dtype=np.float64)
    if not (np.isfinite(c2w).all() and np.isfinite(depth).all()):
        raise ValueError(f"Non-finite pose or depth for {dataset} {scene}/{frame}")
    return _frame_view(image, crop, depth, record["K"], c2w, center_crop)


def load_ood_pair(sample, ood_root, size=IMAGE_SIZE, center_crop=False):
    views = []
    for i in (1, 2):
        frame = sample[f"frame{i}"]
        record = ood_cameras(str(ood_root), sample["dataset"])[
            f"{sample['scene']}/{frame}"
        ]
        for kind in ("image", "depth"):
            if record[kind] != sample[f"{kind}{i}_path"]:
                raise ValueError(f"{sample['sample_id']}: {kind}{i}_path disagrees")
        views.append(
            load_ood_view(
                ood_root, sample["dataset"], sample["scene"], frame, size, center_crop
            )
        )
    return _pair(*views)


def load_pair(
    sample,
    split,
    folders=None,
    ood_root=None,
    size=IMAGE_SIZE,
    center_crop=False,
    whole_frame=False,
):
    """Model inputs and ground truth of one 2-view sample (keys img1, depth1, ...).

    In-Distribution pairs are read from `folders` (`indist_folders`); `whole_frame`
    selects their pointmap-metric views. Out-of-Distribution pairs use whole-frame
    views for both metrics.
    """
    if split == "indist":
        return load_indist_pair(sample, folders, size, center_crop, whole_frame)
    return load_ood_pair(sample, ood_root, size, center_crop)
