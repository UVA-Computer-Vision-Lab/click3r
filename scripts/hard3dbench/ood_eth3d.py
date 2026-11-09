"""Prepare the ETH3D frames of the Hard3DBench Out-of-Distribution lists.

``raw`` is the ETH3D high-res multi-view training data, with
``multi_view_training_dslr_jpg.7z`` and every ``<scene>_dslr_depth.7z``
extracted into it::

    <raw>/<scene>/dslr_calibration_jpg/cameras.txt, images.txt
    <raw>/<scene>/images/dslr_images/DSC_<NNNN>.JPG
    <raw>/<scene>/ground_truth_depth/dslr_images/DSC_<NNNN>.JPG

Each frame is undistorted with OpenCV's fisheye model onto the same pinhole
intrinsics. The model uses k1-k4 of the THIN_PRISM_FISHEYE calibration and
ignores p1, p2, sx1 and sy1. Depth is splatted to the nearest undistorted pixel.
"""

# Adapted from Pi3 datasets/preprocess/prepare_eth3d.py, evaluation branch
# 97ce97b (https://github.com/yyfz/Pi3/tree/evaluation). Modified by the Click3R
# authors. Pi3 license ("Pi3 for non-commercial purposes"):
#
# Copyright (c) 2025 the authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# DEPTH_FORMAT is written to cameras.json, whose sha256 is in ood_checksums.json:
# any change to the text, even to its spelling, makes the check fail.
DEPTH_FORMAT = (
    "raw little-endian float32 dump without header despite the .JPG extension, "
    "4032x6048 row-major, meters, invalid = 0"
)


def _read_cameras(path):
    """COLMAP cameras.txt -> {camera_id: (width, height, params)}."""
    cameras = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            params = [float(value) for value in parts[4:]]
            cameras[int(parts[0])] = (int(parts[2]), int(parts[3]), params)
    return cameras


def _read_images(path):
    """COLMAP images.txt -> {name: (qvec, tvec, camera_id)}, world-to-camera."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    images = {}
    for line in lines[0::2]:
        parts = line.split()
        qvec = [float(value) for value in parts[1:5]]
        tvec = [float(value) for value in parts[5:8]]
        images[parts[9]] = (qvec, tvec, int(parts[8]))
    return images


def _cam2world(qvec, tvec):
    """OpenCV camera-to-world matrix from a COLMAP (qw, qx, qy, qz) and t."""
    # Plain float arithmetic keeps cameras.json identical across NumPy versions.
    norm = math.sqrt(
        qvec[0] * qvec[0] + qvec[1] * qvec[1] + qvec[2] * qvec[2] + qvec[3] * qvec[3]
    )
    w, x, y, z = (value / norm for value in qvec)
    rot = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    c2w = []
    for i in range(3):
        center = -(rot[0][i] * tvec[0] + rot[1][i] * tvec[1] + rot[2][i] * tvec[2])
        c2w.append([rot[0][i], rot[1][i], rot[2][i], center])
    return c2w + [[0.0, 0.0, 0.0, 1.0]]


def _undistorter(camera):
    """Float32 K, remap tables and undistorted normalized coordinates of every pixel."""
    width, height, params = camera
    fx, fy, cx, cy, k1, k2, _, _, k3, k4 = params[:10]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)
    D = np.array([k1, k2, k3, k4])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K.copy(), (width, height), cv2.CV_16SC2
    )
    v, u = np.indices((height, width))
    pixels = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)
    rays = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), K, D)
    return K, map1, map2, rays.reshape(-1, 2)


def _undistort_depth(depth, K, rays):
    """Splat valid depth to the rounded undistorted pixel; later pixels overwrite."""
    height, width = depth.shape
    z = depth.ravel()
    valid = (z > 0) & np.isfinite(z)
    x = rays[:, 0][valid] * z[valid]
    y = rays[:, 1][valid] * z[valid]
    z = z[valid]
    u = np.round(x * K[0, 0] / z + K[0, 2]).astype(int)
    v = np.round(y * K[1, 1] / z + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    undistorted = np.zeros((height, width), dtype=np.float32)
    undistorted[v[inside], u[inside]] = z[inside]
    return undistorted


def prepare(raw: Path, out: Path, frames: dict[str, list[str]]) -> dict:
    """Undistort the listed frames into ``out/eth3d`` and return their cameras."""
    records = {}
    for scene, names in sorted(frames.items()):
        print(f"eth3d {scene}: {len(names)} frames")
        src = raw / scene
        cameras = _read_cameras(src / "dslr_calibration_jpg/cameras.txt")
        images = _read_images(src / "dslr_calibration_jpg/images.txt")
        undistorters = {}
        for frame in names:
            qvec, tvec, camera_id = images[f"dslr_images/{frame}.JPG"]
            with Image.open(src / f"images/dslr_images/{frame}.JPG") as image:
                rgb = np.array(image)
            height, width = rgb.shape[:2]
            depth = np.fromfile(
                src / f"ground_truth_depth/dslr_images/{frame}.JPG", dtype="<f4"
            ).reshape(height, width)
            if camera_id not in undistorters:
                undistorters[camera_id] = _undistorter(cameras[camera_id])
            K, map1, map2, rays = undistorters[camera_id]

            image_rel = f"eth3d/{scene}/images/custom_undistorted/{frame}.JPG"
            depth_rel = (
                f"eth3d/{scene}/ground_truth_depth/custom_undistorted/{frame}.JPG"
            )
            for rel in (image_rel, depth_rel):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
            rgb = cv2.remap(
                rgb,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            # Pillow's default JPEG settings (quality 75), as in the paper's data.
            Image.fromarray(rgb).save(out / image_rel, format="JPEG")
            _undistort_depth(depth, K, rays).astype("<f4").tofile(out / depth_rel)

            records[f"{scene}/{frame}"] = {
                "image": image_rel,
                "depth": depth_rel,
                "K": K.astype(np.float64).tolist(),
                "cam2world": _cam2world(qvec, tvec),
                "size": [width, height],
            }
    return {"depth_format": DEPTH_FORMAT, "frames": records}
