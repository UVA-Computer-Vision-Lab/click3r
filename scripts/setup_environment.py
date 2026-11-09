"""Create the local runtime environment and fetch pinned external dependencies."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
SAM_REVISION = "dca509fe793f601edb92606367a655c15ac00fdf"
BASELINES = (
    ("naver/mast3r", "mast3r", "f5209afc300cec36239a7ac992263f36847bbba0"),
    ("facebookresearch/vggt", "vggt", "44b3afbd1869d8bde4894dd8ea1e293112dd5eba"),
    # Pi3 and Pi3X inference code as used for the paper.
    ("yyfz/Pi3", "Pi3", "db0ea536b5a67912e8b486b8a1622f04ea8ffddc"),
)


def requirements_satisfied():
    """Check the active interpreter without importing CUDA or the demo."""
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return False
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            actual = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if actual not in requirement.specifier:
            return False
    return True


def clone_at(repository, destination, revision):
    path = ROOT / "third_party" / destination
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        subprocess.run(
            ["git", "clone", f"https://github.com/{repository}.git", str(path)],
            check=True,
        )
        subprocess.run(["git", "-C", str(path), "checkout", revision], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(
            f"{destination} is at {actual}; expected {revision}. Use a fresh dependency directory."
        )
    return path


def environment_python():
    path = ROOT / ".venv"
    python = path / "bin/python"
    if not python.is_file():
        # Isolated: inherited binary packages built against NumPy 1 break NumPy 2.
        print("Creating isolated .venv.", flush=True)
        venv.EnvBuilder(with_pip=True).create(path)
    return python


def check_imports(baselines=False):
    """Fail early, naming the module, if the installed runtime cannot import."""
    modules = [
        "torch",
        "torchvision",
        "cv2",
        "scipy",
        "matplotlib",
        "timm",
        "gradio",
        "trimesh",
        "roma",
        "segment_anything",
    ]
    if baselines:
        modules += ["sklearn", "safetensors"]
    for module in modules:
        subprocess.run([sys.executable, "-c", f"import {module}"], check=True)


def install(baselines=False):
    sam = clone_at(
        "facebookresearch/segment-anything", "segment-anything", SAM_REVISION
    )
    if not requirements_satisfied():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements.txt"),
            ],
            check=True,
        )
    spec = importlib.util.find_spec("segment_anything")
    expected = sam / "segment_anything/__init__.py"
    if spec is None or Path(spec.origin).resolve() != expected.resolve():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(sam)],
            check=True,
        )
    if baselines:
        for repository, destination, revision in BASELINES:
            dependency = clone_at(repository, destination, revision)
            if destination == "mast3r":
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(dependency),
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                    ],
                    check=True,
                )
        # safetensors already comes with timm from requirements.txt.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "scikit-learn==1.6.1"],
            check=True,
        )
    check_imports(baselines)


def setup(baselines=False):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11 for the tested release environment.")
    python = environment_python()
    command = [str(python), str(Path(__file__).resolve()), "--install"]
    if baselines:
        command.append("--baselines")
    subprocess.run(command, cwd=ROOT, check=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also fetch optional Table 1 comparison models",
    )
    parser.add_argument("--install", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.install:
        install(args.baselines)
    else:
        print(f"Runtime ready: {setup(args.baselines)}")


if __name__ == "__main__":
    main()
