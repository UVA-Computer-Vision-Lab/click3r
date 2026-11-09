"""Reconstruction viewers and display controls for the interactive demo."""

from functools import lru_cache
import json
from pathlib import Path
import re

import gradio as gr
from gradio.data_classes import FileData

DEFAULT_POINT_SIZE = 3  # Pixels; size 1 leaves visible gaps between points.


@lru_cache(maxsize=1)
def viewer_javascript():
    """Use the Babylon viewer distributed with the pinned Gradio installation."""
    assets = Path(gr.__file__).parent / "templates/frontend/assets"
    library = next(
        asset
        for asset in assets.glob("lib-*.js")
        if "CreateViewerForCanvas" in asset.read_text()
    )
    dependencies = re.findall(r'from["\']\./([^"\']+\.js)["\']', library.read_text())
    return (
        Path(__file__)
        .with_name("viewer.js")
        .read_text()
        .replace("__VIEWER_LIBRARY__", json.dumps(library.name))
        .replace("__VIEWER_DEPENDENCIES__", json.dumps(dependencies))
        .replace("__DEFAULT_POINT_SIZE__", json.dumps(DEFAULT_POINT_SIZE))
    )


class ReconstructionViewer(gr.HTML):
    """A downloadable GLB viewer with explicit WebGL point-size support.

    Babylon's default engine selection can choose WebGPU, whose point primitives
    have fixed size. Use its public viewer API with WebGL and retain Gradio's
    FileData processing and the framing metadata embedded in the GLB.
    """

    data_model = FileData

    def __init__(self, *, label, visible=True, **props):
        super().__init__(
            value=None,
            label=label,
            visible=visible,
            scale=1,
            min_width=160,
            container=True,
            padding=False,
            apply_default_css=False,
            elem_classes="click3r-reconstruction",
            # Gradio's inner HTML wrapper adds padding independently of this component.
            head="""
                <style>
                    .block.click3r-reconstruction .html-container { padding: 0; }
                </style>
            """,
            html_template="""
                <div class="viewer">
                    <div class="toolbar">
                        <span class="title">${label}</span>
                        <button class="reset" type="button" aria-label="Reset view">Reset view</button>
                        <a class="download" aria-label="Download reconstruction" download>Download</a>
                        <button class="fullscreen" type="button" aria-label="Fullscreen">Fullscreen</button>
                    </div>
                    <canvas aria-label="${label} reconstruction"></canvas>
                    <div class="notice" role="status">Run reconstruction to view the scene.</div>
                </div>
            """,
            css_template="""
                .viewer { position: relative; width: 100%; aspect-ratio: 1;
                    overflow: hidden;
                    border-radius: var(--block-radius); background: white; }
                .viewer:fullscreen { width: 100vw; height: 100vh; aspect-ratio: auto; }
                canvas { position: absolute; inset: 0; display: block;
                    width: 100%; height: 100%; touch-action: none; }
                .toolbar { position: absolute; top: 0; left: 0; right: 0; z-index: 1;
                    display: flex; flex-wrap: wrap; align-items: center;
                    gap: 4px 8px; padding: 5px 8px;
                    background: var(--background-fill-primary); font-size: 12px; }
                .title { flex: 1; }
                button, a { cursor: pointer; color: var(--body-text-color);
                    background: transparent; border: 0; padding: 2px;
                    font: inherit; text-decoration: none; }
                a:not([href]) { pointer-events: none; opacity: 0.5; }
                .notice { position: absolute; inset: 40px 16px 16px; display: grid;
                    place-items: center; text-align: center; pointer-events: none;
                    color: var(--body-text-color-subdued); }
                .notice[hidden] { display: none; }
            """,
            js_on_load=viewer_javascript(),
            **props,
        )

    def preprocess(self, payload):
        return payload.path if payload is not None else None

    def postprocess(self, value):
        if value is None:
            return None
        return FileData(path=str(value), orig_name=Path(value).name)

    def api_info(self):
        return gr.components.Component.api_info(self)


def point_size_control():
    """Update both WebGL viewers without rerunning reconstruction."""
    slider = gr.Slider(
        1,
        10,
        value=DEFAULT_POINT_SIZE,
        step=0.5,
        label="Point size (px)",
        info="Larger points fill visual gaps in both 3D views.",
    )
    controller = gr.HTML(
        value=DEFAULT_POINT_SIZE,
        html_template="",
        js_on_load="""
            function update() {
                document.getElementById("click3r-point-size").dataset.size = props.value;
                document.dispatchEvent(new CustomEvent("click3r:point-size", {detail: props.value}));
            }
            watch("value", update);
            update();
        """,
        elem_id="click3r-point-size",
        visible="hidden",
    )
    slider.input(None, [slider], [controller], js="(size) => size", queue=False)
    return slider
