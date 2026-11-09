"""Click3R: correspondence-guided stereo reconstruction."""

from pathlib import Path
import sys

# Resolve bundled DUSt3R when importing the architecture from a source checkout.
_dust3r = Path(__file__).resolve().parents[1] / "third_party/dust3r"
if str(_dust3r) not in sys.path:
    sys.path.insert(0, str(_dust3r))
