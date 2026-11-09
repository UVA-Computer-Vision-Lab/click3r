"""NRGBD for Hard3DBench Out-of-Distribution, from ``neural_rgbd_data.zip``.

``raw`` is the zip itself. Images and depth maps are copied unchanged. Cameras
come from ``trainval_poses.txt`` (one OpenGL camera-to-world 4x4 per frame, block
``N`` for frame ``img<N>``) and ``focal.txt``, as in the Hard3DBench evaluation;
``poses.txt`` is not used. The principal point is not in the data and is taken as
the center of the 640x480 images.
"""

import zipfile
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 640, 480
# DEPTH_FORMAT is written to cameras.json, whose sha256 is in ood_checksums.json:
# any change to the text, even to its spelling, makes the check fail.
DEPTH_FORMAT = "uint16 PNG, millimetres (divide by 1000 for metres), 0 where no depth"


def _pose_blocks(text):
    """Split trainval_poses.txt into one 16-number string per frame."""
    lines = [line for line in text.splitlines() if line.strip()]
    return [" ".join(lines[i : i + 4]) for i in range(0, len(lines) - 3, 4)]


def prepare(raw: Path, out: Path, frames: dict[str, list[str]]) -> dict:
    """Copy the listed NRGBD frames to ``out/nrgbd`` and return their cameras."""
    files = {
        (scene, frame): (
            f"{scene}/images/{frame}.png",
            f"{scene}/depth/depth{frame.removeprefix('img')}.png",
        )
        for scene, names in frames.items()
        for frame in names
    }
    needed = [rel for group in files.values() for rel in group]
    for scene in frames:
        needed += [f"{scene}/focal.txt", f"{scene}/trainval_poses.txt"]
    names = set(zipfile.ZipFile(raw).namelist()) if zipfile.is_zipfile(raw) else set()
    missing = [rel for rel in needed if rel not in names]
    if missing:
        raise SystemExit(
            f"NRGBD: --raw must be neural_rgbd_data.zip; {raw} lacks "
            f"{len(missing)} needed files (first: {missing[0]})"
        )
    archive = zipfile.ZipFile(raw)
    scenes = {}
    for scene in frames:
        focal = float(archive.read(f"{scene}/focal.txt").decode())
        K = [[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]]
        blocks = _pose_blocks(archive.read(f"{scene}/trainval_poses.txt").decode())
        scenes[scene] = K, blocks
    cameras = {}
    for (scene, frame), (image, depth) in files.items():
        K, blocks = scenes[scene]
        # Unlisted frames can have no pose ("-nan(ind)"), so parse only this block.
        block = blocks[int(frame.removeprefix("img"))]
        cam2world = np.array([float(word) for word in block.split()]).reshape(4, 4)
        cam2world[:3, 1:3] *= -1  # OpenGL to OpenCV: flip the camera y and z axes
        for rel in (image, depth):
            target = out / "nrgbd" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(rel))
        cameras[f"{scene}/{frame}"] = {
            "image": f"nrgbd/{image}",
            "depth": f"nrgbd/{depth}",
            "K": K,
            "cam2world": cam2world.tolist(),
            "size": [WIDTH, HEIGHT],
        }
    return {"depth_format": DEPTH_FORMAT, "frames": cameras}
