"""Resolve the pinned Hugging Face revisions of the baseline checkpoints."""

import json
from pathlib import Path


def hub_options(model_name):
    manifest = (
        Path(__file__).resolve().parents[2]
        / "click3r/configs/baseline_checkpoints.json"
    )
    checkpoints = json.loads(manifest.read_text())
    entry = checkpoints.get(str(model_name))
    return {"revision": entry["revision"]} if entry else {}
