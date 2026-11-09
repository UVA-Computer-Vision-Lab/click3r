"""
Load Click3R, DUSt3R and MASt3R checkpoints; undo the Pi3X cuRoPE2D patch.
"""

import torch
import gc


def _checkpoint_state(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    return checkpoint.get("model", checkpoint)


def load_dust3r_model(checkpoint_path, model, device):
    """Load a DUSt3R baseline and bypass correspondence adapters."""
    if hasattr(model, "prompts_enabled"):
        model.prompts_enabled = False
    incompatible = model.load_state_dict(
        _checkpoint_state(checkpoint_path), strict=False
    )
    missing_backbone = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(("correspondence_adapter1.", "correspondence_adapter2."))
    ]
    if missing_backbone or incompatible.unexpected_keys:
        raise RuntimeError(
            f"DUSt3R checkpoint mismatch: missing={missing_backbone}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def load_click3r_model(checkpoint_path, model, device):
    """Load every backbone and correspondence tensor from a Click3R checkpoint."""
    model.prompts_enabled = True
    model.load_state_dict(_checkpoint_state(checkpoint_path), strict=True)
    return model.to(device).eval()


def load_mast3r_model(model_name, device):
    """
    Load MASt3R model from HuggingFace.

    Args:
        model_name: HuggingFace model name (e.g., 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
        device: torch device

    Returns:
        Loaded MASt3R model on device in eval mode
    """
    from mast3r.model import AsymmetricMASt3R

    from evaluation.baselines.checkpoints import hub_options

    model = (
        AsymmetricMASt3R.from_pretrained(model_name, **hub_options(model_name))
        .to(device)
        .eval()
    )
    return model


def cleanup_model(model):
    """
    Delete model and clear GPU cache.

    Special handling for Pi3XWrapper to restore cuRoPE2D patches
    and prevent "free(): invalid pointer" errors during exit.

    Args:
        model: Model to cleanup
    """
    if model.__class__.__name__ == "Pi3XWrapper":
        cleanup_pi3_wrapper(model)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def cleanup_pi3_wrapper(pi3_model):
    """
    Clean up Pi3XWrapper by restoring original cuRoPE2D forward methods.

    This prevents "free(): invalid pointer" errors caused by CUDA extension
    cleanup when cuRoPE2D has been patched.

    Args:
        pi3_model: Pi3XWrapper instance to cleanup
    """
    # Restore instance-level patches
    if hasattr(pi3_model, "patched_modules"):
        for module in pi3_model.patched_modules:
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                delattr(module, "_original_forward")


def restore_curope_class_patch():
    """
    Undo patch_curope_for_pi3x so later models use the stock cuRoPE2D forward.

    The patch is applied to the class loaded as models.curope.curope2d, the
    name CroCo and Pi3 import it under, so look that module up. Importing the
    file again under another name (croco.models.curope.curope2d) gives a
    different class, and with the in-place build it loads the compiled kernel
    a second time, which aborts the interpreter at exit with
    "free(): invalid pointer".
    """
    import sys

    curope2d = sys.modules.get("models.curope.curope2d")
    if curope2d is None:
        # cuRoPE2D was never imported, so nothing was patched
        return
    cuRoPE2D = curope2d.cuRoPE2D
    if hasattr(cuRoPE2D, "_original_class_forward"):
        cuRoPE2D.forward = cuRoPE2D._original_class_forward
        delattr(cuRoPE2D, "_original_class_forward")
        if hasattr(cuRoPE2D, "_pi3_patched"):
            delattr(cuRoPE2D, "_pi3_patched")
