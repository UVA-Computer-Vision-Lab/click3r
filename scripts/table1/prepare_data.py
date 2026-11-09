"""Prepare the Table 1 evaluation data from the official dataset releases.

Table 1 evaluates on the CO3D v2 test sequences, the ARKitScenes Validation fold and
the ScanNet++ nvs_sem_val scenes. ``click3r/configs/table1/`` fixes the sequences,
frames and image pairs. The script converts them with DUSt3R's converters in
``third_party/dust3r/datasets_preprocess/`` into ``<out>/<dataset>_processed_for_dust3r``
and then checks the result against ``data_checksums.json``.

- co3d: reads the 499 sequences of ``co3d_sequences.json`` from the CO3D v2 zips
  with HTTP range requests, or from a full CO3D v2 download given as ``--raw``.
- arkitscenes: reads the frames in ``arkitscenes_pairs.npz`` from Apple's
  per-video zips with HTTP range requests.
- scannetpp: decodes the iPhone frames in ``scannetpp_pairs.npz`` from your
  ScanNet++ download (``--raw``) with ffmpeg; the converter renders depth with EGL.

    python scripts/table1/prepare_data.py --dataset co3d --out /path/to/table1_data
    python scripts/table1/prepare_data.py --dataset scannetpp \\
        --raw /path/to/scannetpp --out /path/to/table1_data
    python scripts/table1/prepare_data.py --dataset co3d --check-only \\
        --out /path/to/table1_data
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "click3r/configs/table1"
PREPROCESS = REPO / "third_party/dust3r/datasets_preprocess"
ARKIT_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
    "/raw/Validation"
)
CO3D_FILES = (
    "images/frame{:06d}.jpg",
    "images/frame{:06d}.npz",
    "depths/frame{:06d}.jpg.geometric.png",
    "masks/frame{:06d}.png",
)
CO3D_FOLDERS = ("images", "masks", "depths")  # the converter reads these
SHOWN = 10  # file names listed per problem
TIMEOUT = 300  # seconds without data before a request fails


def download(url):
    """Return the bytes of url."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def http_range(url, start, end):
    """Open bytes [start, end) of url with an HTTP range request."""
    headers = {"Range": f"bytes={start}-{end - 1}"}
    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, timeout=TIMEOUT)
    assert response.status == 206, (url, response.status)
    return response


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file for ``zipfile``."""

    def __init__(self, url):
        super().__init__()
        self.url, self.pos = url, 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size)
        if end <= self.pos:
            return 0
        with http_range(self.url, self.pos, end) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def zip_members(url, start, end):
    """Yield (name, data) of the zip members stored in bytes [start, end) of url.

    The span starts at a local file header and ends where the member after the
    last one starts. Every member is inflated and checked against its CRC-32.
    """
    with http_range(url, start, end) as stream:
        while start < end:
            fields = struct.unpack("<IHHHHHIIIHH", stream.read(30))
            sig, _, flag, method, _, _, crc, size, _, name_len, extra_len = fields
            assert sig == 0x04034B50 and not flag & 0x9 and size != 0xFFFFFFFF, url
            name = stream.read(name_len).decode()
            stream.read(extra_len)
            data = stream.read(size)
            if method == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            assert method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), (url, name)
            assert zlib.crc32(data) == crc, (url, name)
            start += 30 + name_len + extra_len + size
            yield name, data


def fetch_members(url, names, folder):
    """Write the members ``names`` of the remote zip at url under folder.

    Only the central directory and the listed members are read; members that
    follow each other in the zip are read with one request.
    """
    with zipfile.ZipFile(RemoteFile(url)) as archive:
        infos = sorted(archive.infolist(), key=lambda info: info.header_offset)
        ends = [info.header_offset for info in infos[1:]] + [archive.start_dir]
    spans = []
    for info, end in zip(infos, ends, strict=True):
        if info.filename in names:
            if spans and spans[-1][1] == info.header_offset:
                spans[-1][1] = end
            else:
                spans.append([info.header_offset, end])
    written = 0
    for start, end in spans:
        for name, data in zip_members(url, start, end):
            (folder / name).write_bytes(data)
            written += 1
    assert written == len(names), (url, written, len(names))


def co3d_category(task):
    """Fetch (unless --raw) and convert the listed sequences of one category."""
    category, sequences, url, raw, root, tmp = task
    done = root / category / "selected_seqs_test.json"
    if done.is_file():
        return category
    fetch = raw is None
    if fetch:
        raw = tmp / category
        shutil.rmtree(raw, ignore_errors=True)
        names = ["set_lists/set_lists_fewview_dev.json", "frame_annotations.jgz"]
        names = [f"{category}/{name}" for name in [*names, "sequence_annotations.jgz"]]
        with zipfile.ZipFile(io.BytesIO(download(f"{url}{category}_000.zip"))) as zf:
            zf.extractall(raw, members=names)
        for zip_name, start, end in sequences.values():
            for name, data in zip_members(url + zip_name, start, end):
                # <category>/<sequence>/<folder>/<file>
                if not name.endswith("/") and name.split("/")[2] in CO3D_FOLDERS:
                    (raw / name).parent.mkdir(parents=True, exist_ok=True)
                    (raw / name).write_bytes(data)
    os.environ["TQDM_DISABLE"] = "1"
    sys.path.insert(0, str(PREPROCESS))
    import preprocess_co3d

    selected = preprocess_co3d.prepare_sequences(
        category=category,
        co3d_dir=str(raw),
        output_dir=str(root),
        img_size=512,
        split="test",
        min_quality=0.5,
        max_num_sequences_per_object=50,
        seed=42 + preprocess_co3d.CATEGORIES_IDX[category],
        set_list="fewview_dev",
        sequences=list(sequences),
    )
    done.write_text(json.dumps(selected))
    if fetch:
        shutil.rmtree(raw)
    return category


def prepare_co3d(raw, out, workers):
    config = json.loads((CONFIGS / "co3d_sequences.json").read_text())
    root, tmp = out / "co3d_processed_for_dust3r", out / "co3d_tmp"
    url = config["metadata"]["url"]
    tasks = [
        (category, sequences, url, raw, root, tmp)
        for category, sequences in config["sequences"].items()
    ]
    with Pool(workers) as pool:
        for category in pool.imap_unordered(co3d_category, tasks):
            print(f"co3d: converted {category}", flush=True)
    selected = {
        category: json.loads((root / category / "selected_seqs_test.json").read_text())
        for category in config["sequences"]
    }
    (root / "selected_seqs_test.json").write_text(json.dumps(selected))
    shutil.rmtree(tmp, ignore_errors=True)


def scenes_and_pairs(name):
    """Return [(scene, frames, pairs with scores)] from a compact pair file."""
    with np.load(CONFIGS / name) as data:
        arrays = {key: data[key] for key in data.files}
    frames = np.split(arrays["frames"], np.cumsum(arrays["frame_counts"])[:-1])
    split = np.cumsum(arrays["pair_counts"])[:-1]
    pairs = np.split(arrays["pairs"], split)
    scores = np.split(arrays["scores"], split)
    rows = zip(arrays["scenes"].tolist(), frames, pairs, scores, strict=True)
    return [
        (scene, names.tolist(), np.column_stack([ij, score]).astype(np.float64))
        for scene, names, ij, score in rows
    ], arrays


def write_pairs(folder, scenes, **extra):
    """Write DUSt3R's pair format: scene_list.json and <scene>/selected_pairs.npz."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene_list.json").write_text(json.dumps([s for s, _, _ in scenes]))
    for k, (scene, frames, pairs) in enumerate(scenes):
        (folder / scene).mkdir(exist_ok=True)
        values = {key: value[k : k + 1] for key, value in extra.items()}
        path = folder / scene / "selected_pairs.npz"
        np.savez(path, selection=np.array(frames), pairs=pairs, **values)


def arkit_video(task):
    """Fetch the listed frames of one video in the raw layout the converter reads."""
    video, frames, raw, root = task
    folder = raw / "Test" / video
    if folder.is_dir() or (root / "Test" / video / "scene_metadata.npz").is_file():
        return video
    part = folder.with_name(video + ".part")
    shutil.rmtree(part, ignore_errors=True)
    for name in ("vga_wide", "lowres_depth", "vga_wide_intrinsics"):
        (part / name).mkdir(parents=True)
    url = f"{ARKIT_URL}/{video}"
    (part / "lowres_wide.traj").write_bytes(download(f"{url}/lowres_wide.traj"))
    intrinsics = zipfile.ZipFile(io.BytesIO(download(f"{url}/vga_wide_intrinsics.zip")))
    present = set(intrinsics.namelist())
    for frame in frames:
        # the converter takes the intrinsics of the frame's timestamp or of +-1 ms
        stamp = frame[len(video) + 1 : -len(".png")]
        nearby = (f"{float(stamp) - 0.001:.3f}", f"{float(stamp) + 0.001:.3f}")
        for value in (stamp, *nearby):
            name = f"vga_wide_intrinsics/{video}_{value}.pincam"
            if name in present:
                (part / name).write_bytes(intrinsics.read(name))
    for asset in ("vga_wide", "lowres_depth"):
        names = {f"{asset}/{frame}" for frame in frames}
        fetch_members(f"{url}/{asset}.zip", names, part)
    part.rename(folder)
    return video


def prepare_arkitscenes(raw, out, workers):
    videos, arrays = scenes_and_pairs("arkitscenes_pairs.npz")
    root, tmp = out / "arkitscenes_processed_for_dust3r", out / "arkitscenes_tmp"
    tasks = [(video, frames, tmp / "raw", root) for video, frames, _ in videos]
    with Pool(workers) as pool:
        for video in pool.imap_unordered(arkit_video, tasks):
            print(f"arkitscenes: fetched {video}", flush=True)
    sky = arrays["sky_direction_scene"]
    write_pairs(tmp / "pairs/Test", videos, sky_direction_scene=sky)
    converter = PREPROCESS / "preprocess_arkitscenes.py"
    command = [sys.executable, converter, "--arkitscenes_dir", tmp / "raw"]
    command += ["--precomputed_pairs", tmp / "pairs"]
    command += ["--output_dir", root]
    subprocess.run(command, check=True)
    shutil.rmtree(tmp)


def decode_frames(video, numbers, folder, ext, options):
    """Decode frames ``numbers`` of video to folder/frame_<n>.<ext> with ffmpeg.

    The options are the ScanNet++ toolkit's. A select filter keeps the listed
    frames (a balanced sum, as ffmpeg limits the expression depth), and setpts
    numbers them consecutively so that every ffmpeg version writes each once.
    """

    def select(terms):
        if len(terms) <= 8:
            return "+".join(terms)
        half = len(terms) // 2
        return f"({select(terms[:half])})+({select(terms[half:])})"

    folder.mkdir(parents=True)
    expression = select([f"eq(n,{n})" for n in numbers])
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", video]
    command += ["-vf", f"select='{expression}',setpts=N/FRAME_RATE/TB", *options]
    subprocess.run([*command, folder / f"%06d.{ext}"], check=True)
    decoded = sorted(folder.iterdir())
    assert len(decoded) == len(numbers), (video, len(decoded), len(numbers))
    for path, number in zip(decoded, sorted(numbers), strict=True):
        path.rename(folder / f"frame_{number:06d}.{ext}")


def scannetpp_scene(task):
    """Link the scene's download and decode its listed iPhone frames."""
    scene, frames, raw, data = task
    folder = data / scene
    if folder.is_dir():
        return scene
    part = folder.with_name(scene + ".part")
    shutil.rmtree(part, ignore_errors=True)
    source = raw / "data" / scene
    (part / "iphone").mkdir(parents=True)
    for name in ("dslr", "scans", "iphone/colmap"):
        (part / name).symlink_to(source / name)
    numbers = [int(frame[len("frame_") :]) for frame in frames if "frame_" in frame]
    iphone = source / "iphone"
    decode_frames(
        iphone / "rgb.mp4", numbers, part / "iphone/rgb", "jpg", ["-q:v", "1"]
    )
    masks = ["-pix_fmt", "gray"]
    decode_frames(
        iphone / "rgb_mask.mkv", numbers, part / "iphone/rgb_masks", "png", masks
    )
    part.rename(folder)
    return scene


def prepare_scannetpp(raw, out, workers):
    scenes, _ = scenes_and_pairs("scannetpp_pairs.npz")
    tmp = out / "scannetpp_tmp"
    tasks = [
        (scene, frames, raw.resolve(), tmp / "data") for scene, frames, _ in scenes
    ]
    with Pool(workers) as pool:
        for scene in pool.imap_unordered(scannetpp_scene, tasks):
            print(f"scannetpp: decoded the iPhone frames of {scene}", flush=True)
    write_pairs(tmp / "pairs", scenes)
    path = [str(REPO / "third_party/dust3r"), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, path)))
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    converter = PREPROCESS / "preprocess_scannetpp.py"
    command = [sys.executable, converter, "--scannetpp_dir", tmp]
    command += ["--precomputed_pairs", tmp / "pairs"]
    command += ["--output_dir", out / "scannetpp_processed_for_dust3r"]
    subprocess.run(command, check=True, env=env)
    shutil.rmtree(tmp)


DATASETS = {
    "co3d": ("co3d_processed_for_dust3r", "selected_seqs_test.json", prepare_co3d),
    "arkitscenes": (
        "arkitscenes_processed_for_dust3r",
        "Test/all_metadata.npz",
        prepare_arkitscenes,
    ),
    "scannetpp": (
        "scannetpp_processed_for_dust3r",
        "all_metadata.npz",
        prepare_scannetpp,
    ),
}


def array_sha256(key, array):
    """sha256 of a metadata array, by the rule in data_checksums.json."""
    if key == "pairs":
        array = array[:, :2].astype(np.int64)
    if array.dtype.kind == "U":
        data = "\n".join(array.tolist()).encode()
    elif array.dtype.kind in "iu":
        data = array.astype("<i8").tobytes()
    else:
        data = (np.round(array.astype("<f8"), 4) + 0.0).tobytes()
    return hashlib.sha256(data).hexdigest()


def digest(dataset, root):
    """Return the metadata checksums and the frame files the metadata lists."""
    path = root / DATASETS[dataset][1]
    if dataset == "co3d":
        listed = json.loads(path.read_text()).items()
        files = [
            f"{category}/{sequence}/{name.format(frame)}"
            for category, sequences in listed
            for sequence, frames in sequences.items()
            for frame in frames
            for name in CO3D_FILES
        ]
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}, files
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    checksums = {key: array_sha256(key, arrays[key]) for key in sorted(arrays)}
    scenes = arrays["scenes"][arrays["sceneids"]].tolist()
    if dataset == "arkitscenes":
        names = ("Test/{}/vga_wide/{}.jpg", "Test/{}/lowres_depth/{}.png")
        frames = [name[: -len(".png")] for name in arrays["images"].tolist()]
    else:
        names = ("{}/images/{}.jpg", "{}/depth/{}.png")
        frames = arrays["images"].tolist()
    pairs = zip(scenes, frames, strict=True)
    files = [name.format(scene, frame) for scene, frame in pairs for name in names]
    return checksums, files


def check(dataset, out):
    """Compare the metadata and file list under ``out`` with data_checksums.json."""
    expected = json.loads((CONFIGS / "data_checksums.json").read_text())[dataset]
    root = out / DATASETS[dataset][0]
    if not (root / DATASETS[dataset][1]).is_file():
        raise SystemExit(f"{dataset}: FAILED, {root / DATASETS[dataset][1]} is missing")
    checksums, files = digest(dataset, root)
    present = {
        os.path.relpath(os.path.join(folder, name), root)
        for folder, _, names in os.walk(root)
        for name in names
    }
    missing = [rel for rel in files if rel not in present]
    problems = [
        f"{key} differs"
        for key in expected["metadata"]
        if checksums.get(key) != expected["metadata"][key]
    ]
    if len(files) != expected["files"]:
        problems.append(
            f"the metadata lists {len(files)} files, not {expected['files']}"
        )
    if missing:
        listed = "".join(f"\n  {rel}" for rel in missing[:SHOWN])
        more = (
            f"\n  ... and {len(missing) - SHOWN} more" if len(missing) > SHOWN else ""
        )
        problems.append(
            f"{len(missing)} of {len(files)} files are missing:{listed}{more}"
        )
    for problem in problems:
        print(f"{dataset}: {problem}")
    if problems:
        raise SystemExit(
            f"{dataset}: FAILED, the data differs from data_checksums.json"
        )
    print(f"{dataset}: OK, the metadata matches and all {len(files)} files are present")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--raw",
        type=Path,
        help="scannetpp: the download root with data/<scene>; co3d (optional): "
        "a full CO3D v2 download instead of range requests",
    )
    parser.add_argument("--out", type=Path, required=True, help="the Table 1 data root")
    parser.add_argument("--workers", type=int, default=8, help="parallel processes")
    parser.add_argument(
        "--check-only", action="store_true", help="only check an existing --out"
    )
    args = parser.parse_args()
    if not args.check_only:
        if args.dataset == "scannetpp" and args.raw is None:
            parser.error("--raw is required for scannetpp")
        if args.dataset == "arkitscenes" and args.raw is not None:
            parser.error("arkitscenes reads Apple's zips; it takes no --raw")
        DATASETS[args.dataset][2](args.raw, args.out.resolve(), args.workers)
    check(args.dataset, args.out)


if __name__ == "__main__":
    main()
