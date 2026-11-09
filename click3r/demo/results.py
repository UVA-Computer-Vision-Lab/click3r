"""Named reconstruction outputs for the interactive demo."""

from typing import NamedTuple


class ReconstructionResult(NamedTuple):
    """GLB paths for each comparison and the user-facing reconstruction status."""

    original: str | None
    retrained: str | None
    vggt: str | None
    pi3: str | None
    overlap: str | None
    prompted: str | None
    message: str
