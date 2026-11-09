"""ScanNetv2 for Hard3DBench Out-of-Distribution, from the scenes' ``.sens`` files.

``raw`` is the ``-o`` folder of the official ``download-scannet.py``, which saves
test scenes as ``scans_test/<scene>/<scene>.sens``. List frame ``frame_<i>`` is
``.sens`` frame ``3 * i`` (the first 270 frames with stride 3). As in the SensReader
export the paper used, color is decoded and re-encoded as Pillow JPEG quality 75
(the ``.sens`` JPEG bytes themselves would give different pixels) and depth is
written as a 16-bit PNG. Only the frames up to the last listed one are read from
each file.
"""

# The .sens reader follows ScanNet SensReader/python/SensorData.py
# (https://github.com/ScanNet/ScanNet). Copyright 2017 Angela Dai, Angel X. Chang,
# Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner.
# Licensed under the MIT License, whose permission notice is in the repository's
# LICENSE file. Modified by the Click3R authors.

import io
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

STRIDE = 3
JPEG_QUALITY = 75
# DEPTH_FORMAT is written to cameras.json, whose sha256 is in ood_checksums.json:
# any change to the text, even to its spelling, makes the check fail.
DEPTH_FORMAT = (
    "uint16 PNG, 640x480 (resize to the 1296x968 image with nearest neighbour), "
    "millimetres (divide by 1000 for metres), 0 where no depth"
)


def _read_header(handle):
    """Read the .sens header up to the first frame; return the color K and sizes."""
    name_length = struct.unpack("<4xQ", handle.read(12))[0]  # skip the version
    handle.seek(name_length, 1)
    calibration = np.frombuffer(handle.read(4 * 64), dtype="<f4").reshape(4, 4, 4)
    # Color width and height, depth width and height; skip the two compression
    # types before them and the depth shift and frame count after them.
    sizes = struct.unpack("<8x4I12x", handle.read(36))
    return calibration[0][:3, :3], sizes


def _export_scene(sens, scene, names, out):
    """Write the listed frames of one scene and return their camera entries."""
    wanted = {STRIDE * int(name.removeprefix("frame_")): name for name in names}
    folder = out / "scannetv2" / scene
    (folder / "color_90").mkdir(parents=True, exist_ok=True)
    (folder / "depth_90").mkdir(parents=True, exist_ok=True)
    cameras = {}
    with sens.open("rb") as handle:
        K, (width, height, depth_width, depth_height) = _read_header(handle)
        for index in range(max(wanted) + 1):
            cam2world = np.frombuffer(handle.read(64), dtype="<f4").reshape(4, 4)
            color_size, depth_size = struct.unpack("<16xQQ", handle.read(32))
            if index not in wanted:
                handle.seek(color_size + depth_size, 1)
                continue
            name = wanted[index]
            color = Image.open(io.BytesIO(handle.read(color_size))).convert("RGB")
            image_rel = f"scannetv2/{scene}/color_90/{name}.jpg"
            Image.fromarray(np.asarray(color)).save(
                out / image_rel, format="JPEG", quality=JPEG_QUALITY
            )
            depth = np.frombuffer(zlib.decompress(handle.read(depth_size)), "<u2")
            depth_rel = f"scannetv2/{scene}/depth_90/{name}.png"
            cv2.imwrite(str(out / depth_rel), depth.reshape(depth_height, depth_width))
            cameras[f"{scene}/{name}"] = {
                "image": image_rel,
                "depth": depth_rel,
                "K": K.tolist(),
                "cam2world": cam2world.tolist(),
                "size": [width, height],
            }
    return cameras


def prepare(raw: Path, out: Path, frames: dict[str, list[str]]) -> dict:
    """Export the listed ScanNetv2 frames to ``out/scannetv2`` with their cameras."""
    sens = {scene: raw / "scans_test" / scene / f"{scene}.sens" for scene in frames}
    for scene, path in sens.items():
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing. --raw must be the download-scannet.py -o folder; "
                f"download the file with download-scannet.py -o {raw} --id {scene} "
                "--type .sens"
            )
    cameras = {}
    for scene, path in sens.items():
        cameras.update(_export_scene(path, scene, frames[scene], out))
        print(f"scannetv2 {scene}: {len(frames[scene])} frames")
    return {"depth_format": DEPTH_FORMAT, "frames": cameras}
