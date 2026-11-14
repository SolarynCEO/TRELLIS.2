import copy
import os
import warnings
from typing import Any

from PIL import Image

from .BiRefNet import *
from .preprocess_utils import (
    frame_and_premultiply_rgba,
    image_has_nontrivial_alpha,
    resize_image_for_preprocess,
)

ALLOW_NONCOMMERCIAL_RMBG_ENV = "TRELLIS2_ALLOW_NONCOMMERCIAL_RMBG"


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_rembg_config(rembg_config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve rembg configuration with a commercial-safe default.

    The upstream pipeline config may still point to briaai/RMBG-2.0. Unless the
    caller explicitly opts in via TRELLIS2_ALLOW_NONCOMMERCIAL_RMBG, we switch
    BiRefNet to a pinned MIT-licensed checkpoint.
    """
    resolved = copy.deepcopy(rembg_config)
    resolved.setdefault("args", {})
    if resolved.get("name") != "BiRefNet":
        return resolved

    args = resolved["args"]
    model_name = args.get("model_name")
    allow_noncommercial = _is_truthy(os.getenv(ALLOW_NONCOMMERCIAL_RMBG_ENV))

    if model_name in (None, "", DEFAULT_BIREFNET_MODEL_NAME):
        args["model_name"] = DEFAULT_BIREFNET_MODEL_NAME
        args.setdefault("revision", DEFAULT_BIREFNET_REVISION)
        return resolved

    if model_name == NON_COMMERCIAL_RMBG_MODEL_NAME and not allow_noncommercial:
        warnings.warn(
            f"Blocked default use of non-commercial model '{NON_COMMERCIAL_RMBG_MODEL_NAME}'. "
            f"Falling back to MIT-licensed '{DEFAULT_BIREFNET_MODEL_NAME}@{DEFAULT_BIREFNET_REVISION}'. "
            f"Set {ALLOW_NONCOMMERCIAL_RMBG_ENV}=1 to enable RMBG-2.0 for research-only comparisons.",
            RuntimeWarning,
            stacklevel=2,
        )
        args["model_name"] = DEFAULT_BIREFNET_MODEL_NAME
        args["revision"] = DEFAULT_BIREFNET_REVISION
        return resolved

    return resolved


def build_rembg_model(rembg_config: dict[str, Any]):
    resolved = resolve_rembg_config(rembg_config)
    return globals()[resolved["name"]](**resolved.get("args", {}))


def preprocess_with_rembg(
    input_image: Image.Image,
    rembg_model: Any,
    *,
    device: str,
    low_vram: bool,
    warning_context: str,
) -> Image.Image:
    """
    Apply TRELLIS foreground framing with a safe fallback for implausibly small masks.
    """
    resized = resize_image_for_preprocess(input_image)
    if image_has_nontrivial_alpha(resized):
        rgba = resized.convert("RGBA")
    else:
        rgb = resized.convert("RGB")
        if low_vram:
            rembg_model.to(device)
        rgba = rembg_model(rgb)
        if low_vram:
            rembg_model.cpu()
    output_rgb, _ = frame_and_premultiply_rgba(rgba, warning_context=warning_context)
    return output_rgb
