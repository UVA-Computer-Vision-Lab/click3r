"""Per-browser state for the interactive demo's callbacks."""

import copy
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

_SESSION = ContextVar("click3r_demo_session", default=None)
# Image lists: never edited in place (new images replace old ones), so a copied session can share the images.
_IMAGE_KEYS = ("uploaded_images", "current_image_pair", "display_images")


class DemoSession(dict):
    """A Gradio-compatible dictionary whose keys can also be read as attributes."""

    def __deepcopy__(self, memo):
        # Each edit copies the session; deep-copying two full-resolution photos would add tens of milliseconds per click.
        return DemoSession(
            (key, list(value) if key in _IMAGE_KEYS else copy.deepcopy(value, memo))
            for key, value in self.items()
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def new_session():
    """Create an independent state and output namespace for a browser session."""
    return DemoSession(
        correspondence_points=[],
        uploaded_images=[],
        current_image_pair=[None, None],
        display_images=[None, None],
        click_count=0,
        image_augmented=False,
        id=uuid.uuid4().hex,
        example=None,
    )


def current_session():
    state = _SESSION.get()
    if state is None:
        raise RuntimeError("Demo callback requires a session")
    return state


@contextmanager
def session_scope(state):
    """Bind state for one callback and restore the calling context on exit."""
    token = _SESSION.set(state)
    try:
        yield state
    finally:
        _SESSION.reset(token)


def session_edit(state, callback, *args):
    """Return display updates and a copied state, leaving the input untouched."""
    state = copy.deepcopy(state) if state else new_session()
    with session_scope(state):
        result = callback(*args)
    return (*result, state)
