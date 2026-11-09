"""Prepare the 7-Scenes frames of the Hard3DBench Out-of-Distribution lists.

``raw`` is the folder holding the seven scene zips from the 7-Scenes project page
(``chess.zip`` ... ``stairs.zip``). Color images are copied unchanged; Kinect
depth is registered to the color camera.
"""

# register_depth and its constants (COLOR_FOCAL, DEPTH_FOCAL, DEPTH_TO_COLOR) are
# adapted from SimpleRecon data_scripts/7scenes_preprocessing.py
# (https://github.com/nianticlabs/simplerecon). Modified by the Click3R authors.
# Copyright © Niantic Spatial, Inc. 2022. All rights reserved.
# Licensed under the SimpleRecon License (non-commercial use only), a copy of which
# is in LICENSE_SimpleRecon; this file remains under that license.
#
# SimpleRecon's script is based on setup_7scenes.py of
# https://github.com/tsattler/visloc_pseudo_gt_limitations, licensed as follows:
#
# BSD 3-Clause License
#
# Parts of the evaluation and dataset scripts are based on DSAC*
# https://github.com/vislearn/dsacstar
#
# Copyright (c) 2020, University Heidelberg
# Copyright (c) 2022, Niantic
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np

# DEPTH_FORMAT is written to cameras.json, whose sha256 is in ood_checksums.json:
# any change to the text, even to its spelling, makes the check fail.
DEPTH_FORMAT = (
    "16-bit PNG, millimetres, registered to the colour camera, 0 = no depth; raw "
    "invalid depth (65535) is kept as 65031-65535 or, past 65535, wrapped to 0-478"
)
SIZE = [640, 480]
# The paper's evaluation gives every frame the 7-Scenes default depth-camera
# intrinsics, although depth is registered to the color camera with focal 525.
K = [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]]

# The paper's depth maps come from two implementations of the registration below:
# these frames from SimpleRecon's original loop, which rounds the projected pixel
# coordinates, and all other frames from a vectorized rewrite that truncates them.
ROUNDED = frozenset(
    [f"chess/seq-01/frame-{i:06d}" for i in (119, 560, 843, 913)]
    + [f"fire/seq-01/frame-{i:06d}" for i in (51, 102, 460, 505, 565, 606, 660, 870)]
    + ["heads/seq-02/frame-000083", "heads/seq-02/frame-000497"]
    + ["pumpkin/seq-02/frame-000226"]
)


# Adapted from SimpleRecon (license notice at the top of this file).
COLOR_FOCAL = 525.0
DEPTH_FOCAL = 585.0
# Kinect depth-to-RGB transform, from
# https://projet.liris.cnrs.fr/voir/activities-dataset/kinect-calibration.html
DEPTH_TO_COLOR = np.array(
    [
        [
            9.9996518012567637e-01,
            2.6765126468950343e-03,
            -7.9041012313000904e-03,
            -2.5558943178152542e-02,
        ],
        [
            -2.7409311281316700e-03,
            9.9996302803027592e-01,
            -8.1504520778013286e-03,
            1.0109636268061706e-04,
        ],
        [
            7.8819942130445332e-03,
            8.1718328771890631e-03,
            9.9993554558014031e-01,
            2.0318321729487039e-03,
        ],
    ]
)


def register_depth(depth, rounded=False):
    """Register uint16 Kinect depth (mm) to the color camera, nearest point wins."""
    h, w = depth.shape
    z = depth.astype(np.float32) / 1000.0
    # SimpleRecon casts rays in float64 and rounds the projected pixel; the
    # vectorized rewrite casts them in float32 and truncates.
    dtype = np.float64 if rounded else np.float32
    u, v = np.meshgrid(np.arange(w, dtype=dtype) + 0.5, np.arange(h, dtype=dtype) + 0.5)
    keep = (z > 0) & (z < 100)
    z = z[keep]
    rays = [(u[keep] - w / 2) / DEPTH_FOCAL * z, (v[keep] - h / 2) / DEPTH_FOCAL * z, z]
    x, y, z = [ray.astype(np.float64) for ray in rays]
    # Elementwise products instead of a matrix product keep the result BLAS-independent.
    x, y, z = [m[0] * x + m[1] * y + m[2] * z + m[3] for m in DEPTH_TO_COLOR]
    px = x / z * COLOR_FOCAL + w / 2
    py = y / z * COLOR_FOCAL + h / 2
    if rounded:
        px, py = np.rint(px), np.rint(py)
    px, py = px.astype(np.int64), py.astype(np.int64)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    index, z = py[inside] * w + px[inside], z[inside]
    order = np.lexsort((z, index))
    index, z = index[order], z[order]
    nearest = np.ones(len(index), dtype=bool)
    nearest[1:] = index[1:] != index[:-1]
    registered = np.zeros(h * w, dtype=np.float32)
    registered[index[nearest]] = z[nearest]
    # Values above 65535 mm wrap around, as the original uint16 cast did on x86.
    return (registered * 1000).astype(np.int64).astype(np.uint16).reshape(h, w)


def _write_frame(out, scene, frame, archive):
    """Copy one frame's color image, register its depth and return its camera."""
    color = archive.read(f"{frame}.color.png")
    raw_depth = np.frombuffer(archive.read(f"{frame}.depth.png"), np.uint8)
    depth = cv2.imdecode(raw_depth, cv2.IMREAD_UNCHANGED)
    cam2world = np.loadtxt(io.BytesIO(archive.read(f"{frame}.pose.txt"))).reshape(4, 4)
    registered = register_depth(depth, f"{scene}/{frame}" in ROUNDED)
    stem = f"7scenes/{scene}/{frame}"
    (out / stem).parent.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.color.png").write_bytes(color)
    (out / f"{stem}.depth.proj.png").write_bytes(cv2.imencode(".png", registered)[1])
    return {
        "image": f"{stem}.color.png",
        "depth": f"{stem}.depth.proj.png",
        "K": K,
        "cam2world": cam2world.tolist(),
        "size": SIZE,
    }


def prepare(raw: Path, out: Path, frames: dict[str, list[str]]) -> dict:
    """Write images, registered depth and cameras for {scene: [seq-XX/frame-XXXXXX]}."""
    missing = [
        f"{scene}.zip" for scene in frames if not (raw / f"{scene}.zip").is_file()
    ]
    if missing:
        raise SystemExit(
            f"7-Scenes: --raw must be the folder holding the official scene zips; "
            f"{raw} has no {', '.join(missing)}"
        )
    records = {}
    for scene, names in sorted(frames.items()):
        sequences = {}
        for frame in names:
            sequences.setdefault(frame.split("/")[0], []).append(frame)
        with zipfile.ZipFile(raw / f"{scene}.zip") as scene_zip:
            for seq, seq_frames in sorted(sequences.items()):
                # One sequence zip at a time, in memory (up to 0.6 GB).
                data = io.BytesIO(scene_zip.read(f"{scene}/{seq}.zip"))
                with zipfile.ZipFile(data) as archive:
                    for frame in seq_frames:
                        record = _write_frame(out, scene, frame, archive)
                        records[f"{scene}/{frame}"] = record
                print(f"7scenes {scene}/{seq}: {len(seq_frames)} frames")
    return {"depth_format": DEPTH_FORMAT, "frames": records}
