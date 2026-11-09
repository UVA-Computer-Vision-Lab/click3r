"""DTU for Hard3DBench Out-of-Distribution, from Spann3R's ``dtu_test_mvsnet.zip``.

``raw`` is the zip itself. Images and Spann3R's rendered depth maps are copied
unchanged; cameras come from the MVSNet ``cams/<frame>_cam.txt`` files, whose
translations are in millimeters.
"""

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

TOP = "dtu_test_mvsnet_release"
# DEPTH_FORMAT is written to cameras.json, whose sha256 is in ood_checksums.json:
# any change to the text, even to its spelling, makes the check fail.
DEPTH_FORMAT = (
    "float32 .npy, millimetres (same unit as the camera translations), "
    "0 where Spann3R's rendered mesh has no surface"
)


def _read_cam(text):
    """Parse an MVSNet cam.txt into the world-to-camera 4x4 and the 3x3 K."""
    words = text.split()
    w2c = np.array([float(word) for word in words[1:17]]).reshape(4, 4)
    K = np.array([float(word) for word in words[18:27]]).reshape(3, 3)
    return w2c, K


def prepare(raw: Path, out: Path, frames: dict[str, list[str]]) -> dict:
    """Copy the listed DTU frames to ``out/dtu`` and return their cameras."""
    files = {
        (scene, frame): (
            f"{scene}/images/{frame}.jpg",
            f"{scene}/depths/{frame}.npy",
            f"{scene}/cams/{frame}_cam.txt",
        )
        for scene, names in frames.items()
        for frame in names
    }
    names = set(zipfile.ZipFile(raw).namelist()) if zipfile.is_zipfile(raw) else set()
    missing = [
        rel for group in files.values() for rel in group if f"{TOP}/{rel}" not in names
    ]
    if missing:
        raise SystemExit(
            f"DTU: --raw must be Spann3R's dtu_test_mvsnet.zip; {raw} lacks "
            f"{len(missing)} needed files (first: {TOP}/{missing[0]})"
        )
    archive = zipfile.ZipFile(raw)
    cameras = {}
    for (scene, frame), (image, depth, cam) in files.items():
        for rel in (image, depth):
            target = out / "dtu" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(f"{TOP}/{rel}"))
        w2c, K = _read_cam(archive.read(f"{TOP}/{cam}").decode())
        cameras[f"{scene}/{frame}"] = {
            "image": f"dtu/{image}",
            "depth": f"dtu/{depth}",
            "K": K.tolist(),
            "cam2world": np.linalg.inv(w2c).tolist(),
            "size": list(Image.open(out / "dtu" / image).size),
        }
    return {"depth_format": DEPTH_FORMAT, "frames": cameras}
