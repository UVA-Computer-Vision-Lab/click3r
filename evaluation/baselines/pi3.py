"""Pi3 loader for the Hard3DBench (Table 2) baselines.

Pi3 (https://github.com/yyfz/Pi3) is fetched into ``third_party/Pi3`` by
``scripts/setup_environment.py --baselines``; the ``yyfz233/Pi3`` weights come from
Hugging Face at the revision pinned in ``click3r/configs/baseline_checkpoints.json``.
Pi3's source and weights are licensed under "Pi3 for non-commercial purposes"
(``third_party/Pi3/LICENSE``, copyright (c) 2025 the Pi3 authors); this module only
imports and calls them.

Table 2 runs Pi3 with its pure-PyTorch RoPE: the loader hides the CroCo CUDA RoPE
extension, which rejects the non-contiguous query/key tensors Pi3 passes
(``pi3x.py`` patches the extension for Pi3X instead), so the results do not
depend on whether the extension has been compiled.
"""

import os
import sys

from evaluation.baselines.checkpoints import hub_options

PI3_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "third_party", "Pi3"
)
if PI3_PATH not in sys.path:
    sys.path.insert(0, PI3_PATH)

POS_EMBED = "pi3.models.layers.pos_embed"


def import_pi3():
    """Import the Pi3 class, binding Pi3's pure-PyTorch RoPE."""
    if POS_EMBED not in sys.modules:
        # A None entry makes `from models.curope import cuRoPE2D` raise ImportError,
        # which Pi3 answers with its PyTorch RoPE2D.
        hidden = sys.modules.get("models.curope")
        sys.modules["models.curope"] = None
        __import__(POS_EMBED)
        if hidden is None:
            del sys.modules["models.curope"]
        else:
            sys.modules["models.curope"] = hidden
    rope = sys.modules[POS_EMBED].RoPE2D
    if rope.__module__ != POS_EMBED:
        raise RuntimeError(
            f"Pi3 was already imported with {rope.__module__}.{rope.__name__}; "
            "load Pi3 in a fresh process to use its PyTorch RoPE as in Table 2"
        )
    from pi3.models.pi3 import Pi3

    return Pi3


def load_pi3_model(model_name="yyfz233/Pi3", device="cuda"):
    """Load the pinned Pi3 checkpoint in eval mode."""
    pi3 = import_pi3()
    return pi3.from_pretrained(model_name, **hub_options(model_name)).to(device).eval()
