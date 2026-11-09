"""Fetch checkpoints from official upstream URLs or a supplied Click3R URL."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

RELEASE_CHECKPOINTS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "click3r/configs/checkpoints.json"
    ).read_text()
)

URLS = {
    "dust3r": (
        "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
        "dust3r/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    ),
    "croco": (
        "https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth",
        "croco/CroCo_V2_ViTLarge_BaseDecoder.pth",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(source, output, checksum=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        if checksum and sha256(output) != checksum:
            raise ValueError(
                f"Existing checkpoint checksum mismatch: {output.name}; move it aside before downloading"
            )
        print(f"Already present: {output}")
        return
    temp = output.with_suffix(output.suffix + ".part")
    try:
        if source.startswith(("https://", "http://")):
            with (
                urllib.request.urlopen(source, timeout=60) as response,
                temp.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        else:
            shutil.copyfile(source, temp)
        if checksum:
            if sha256(temp) != checksum:
                raise ValueError(f"Checksum mismatch: {output.name}")
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--upstream", nargs="*", choices=list(URLS), default=[])
    p.add_argument(
        "--click3r-source",
        nargs="?",
        const=RELEASE_CHECKPOINTS["click3r"]["source"],
        help="URL or local exported checkpoint; without a value, the published URL",
    )
    p.add_argument("--click3r-sha256", default=RELEASE_CHECKPOINTS["click3r"]["sha256"])
    p.add_argument(
        "--dust3r-retrained-source",
        nargs="?",
        const=RELEASE_CHECKPOINTS["dust3r_retrained"]["source"],
        help="URL or local exported checkpoint; without a value, the published URL",
    )
    p.add_argument(
        "--dust3r-retrained-sha256",
        default=RELEASE_CHECKPOINTS["dust3r_retrained"]["sha256"],
    )
    args = p.parse_args()
    if (
        not args.upstream
        and not args.click3r_source
        and not args.dust3r_retrained_source
    ):
        p.error("Choose --upstream, --click3r-source, and/or --dust3r-retrained-source")
    for name in args.upstream:
        source, relative = URLS[name]
        acquire(source, args.output_dir / relative)
    if args.click3r_source:
        acquire(
            args.click3r_source, args.output_dir / "click3r.pth", args.click3r_sha256
        )
    if args.dust3r_retrained_source:
        acquire(
            args.dust3r_retrained_source,
            args.output_dir / "dust3r/retrained.pth",
            args.dust3r_retrained_sha256,
        )


if __name__ == "__main__":
    main()
