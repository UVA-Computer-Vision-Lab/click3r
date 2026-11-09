"""Launch Click3R training with environment setup and checkpoint acquisition."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from launch import main

if __name__ == "__main__":
    main("train")
