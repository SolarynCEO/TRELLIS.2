from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from PIL import Image

MAX_PREPROCESS_EDGE = 1024
BBOX_ALPHA_THRESHOLD = 0.8
MASK_COVERAGE_THRESHOLD = 0.02
MASK_OCCUPANCY_THRESHOLD = 0.5
MIN_CROP_SIZE_PX = 128


def image_has_nontrivial_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.array(image, copy=False)[:, :, 3]
    return not np.all(alpha == 255)


def resize_image_for_preprocess(image: Image.Image, max_edge: int = MAX_PREPROCESS_EDGE) -> Image.Image:
    scale = min(1.0, max_edge / max(image.size))
    if scale >= 1.0:
        return image
    return image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )


def _bbox_from_alpha(alpha: np.ndarray, threshold: float = BBOX_ALPHA_THRESHOLD) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(alpha > threshold * 255)
    if coords.size == 0:
        return None
    x0 = int(np.min(coords[:, 1]))
    y0 = int(np.min(coords[:, 0]))
    x1 = int(np.max(coords[:, 1]))
    y1 = int(np.max(coords[:, 0]))
    return x0, y0, x1, y1


def premultiply_rgba_to_rgb(image_rgba: Image.Image) -> Image.Image:
    rgba = np.array(image_rgba).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4]
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def frame_and_premultiply_rgba(
    image_rgba: Image.Image,
    *,
    warning_context: str = "",
    bbox_alpha_threshold: float = BBOX_ALPHA_THRESHOLD,
    occupancy_threshold: float = MASK_OCCUPANCY_THRESHOLD,
    min_mask_coverage: float = MASK_COVERAGE_THRESHOLD,
    min_crop_size_px: int = MIN_CROP_SIZE_PX,
) -> tuple[Image.Image, dict[str, Any]]:
    framed_rgba, metadata = frame_rgba_with_fallback(
        image_rgba,
        warning_context=warning_context,
        bbox_alpha_threshold=bbox_alpha_threshold,
        occupancy_threshold=occupancy_threshold,
        min_mask_coverage=min_mask_coverage,
        min_crop_size_px=min_crop_size_px,
    )
    output_rgb = premultiply_rgba_to_rgb(framed_rgba)
    metadata["output_size"] = list(output_rgb.size)
    return output_rgb, metadata


def frame_rgba_with_fallback(
    image_rgba: Image.Image,
    *,
    warning_context: str = "",
    bbox_alpha_threshold: float = BBOX_ALPHA_THRESHOLD,
    occupancy_threshold: float = MASK_OCCUPANCY_THRESHOLD,
    min_mask_coverage: float = MASK_COVERAGE_THRESHOLD,
    min_crop_size_px: int = MIN_CROP_SIZE_PX,
) -> tuple[Image.Image, dict[str, Any]]:
    rgba_image = image_rgba.convert("RGBA")
    alpha = np.array(rgba_image, copy=False)[:, :, 3]
    mask_coverage = float(np.mean(alpha > occupancy_threshold * 255))

    bbox = _bbox_from_alpha(alpha, threshold=bbox_alpha_threshold)
    fallback_reason = None
    crop_box: tuple[float, float, float, float] | None = None

    if mask_coverage < min_mask_coverage:
        fallback_reason = (
            f"mask coverage {mask_coverage:.4f} is below {min_mask_coverage:.4f}; "
            "using full resized image instead of tiny foreground crop"
        )
    elif bbox is None:
        fallback_reason = (
            f"no foreground pixels above alpha threshold {bbox_alpha_threshold:.2f}; "
            "using full resized image"
        )
    else:
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1)
        if size < min_crop_size_px:
            fallback_reason = (
                f"crop size {size}px is below minimum {min_crop_size_px}px; "
                "using full resized image"
            )
        else:
            crop_box = (
                center[0] - size // 2,
                center[1] - size // 2,
                center[0] + size // 2,
                center[1] + size // 2,
            )

    if fallback_reason is not None:
        prefix = f"{warning_context}: " if warning_context else ""
        warnings.warn(prefix + fallback_reason, RuntimeWarning, stacklevel=2)
        cropped = rgba_image
    else:
        cropped = rgba_image.crop(crop_box)  # type: ignore[arg-type]

    metadata = {
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "mask_coverage": mask_coverage,
        "bbox_alpha_threshold": bbox_alpha_threshold,
        "occupancy_threshold": occupancy_threshold,
        "bbox_xyxy": list(bbox) if bbox is not None else None,
        "crop_box_xyxy": list(crop_box) if crop_box is not None else None,
        "output_size": list(cropped.size),
    }
    return cropped, metadata

