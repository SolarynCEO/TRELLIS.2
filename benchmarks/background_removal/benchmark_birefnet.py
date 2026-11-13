#!/usr/bin/env python3
import argparse
import json
import os
import platform
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw

from trellis2.pipelines.rembg import BiRefNet, NON_COMMERCIAL_RMBG_MODEL_NAME


THRESHOLD_ALPHA = int(0.8 * 255)


@dataclass
class RunResult:
    rgba: Image.Image
    framed_rgb: Image.Image
    framed_rgba: Image.Image
    runtime_s: float
    peak_rss_mb: float
    alpha_coverage: float
    soft_edge_fraction: float
    source_bbox_xyxy: list[int]
    frame_bbox_xyxy: list[int]
    frame_center_offset_px: list[float]


def _bbox_from_alpha(alpha_uint8: np.ndarray, threshold: int = THRESHOLD_ALPHA) -> list[int]:
    indices = np.argwhere(alpha_uint8 > threshold)
    if indices.size == 0:
        h, w = alpha_uint8.shape
        return [0, 0, w - 1, h - 1]
    x0 = int(np.min(indices[:, 1]))
    y0 = int(np.min(indices[:, 0]))
    x1 = int(np.max(indices[:, 1]))
    y1 = int(np.max(indices[:, 0]))
    return [x0, y0, x1, y1]


def _center_from_bbox(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _apply_trellis_framing(rgba_image: Image.Image) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    output_np = np.array(rgba_image)
    alpha = output_np[:, :, 3]
    bbox = _bbox_from_alpha(alpha, threshold=THRESHOLD_ALPHA)
    center = _center_from_bbox(bbox)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    crop_box = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    framed_rgba = rgba_image.crop(crop_box)

    framed_np = np.array(framed_rgba).astype(np.float32) / 255.0
    framed_rgb_np = framed_np[:, :, :3] * framed_np[:, :, 3:4]
    framed_rgb = Image.fromarray((framed_rgb_np * 255).astype(np.uint8))

    frame_alpha = np.array(framed_rgba)[:, :, 3]
    frame_bbox = _bbox_from_alpha(frame_alpha, threshold=THRESHOLD_ALPHA)
    frame_center = _center_from_bbox(frame_bbox)
    image_center = ((framed_rgba.width - 1) / 2.0, (framed_rgba.height - 1) / 2.0)

    stats = {
        "source_bbox_xyxy": bbox,
        "frame_bbox_xyxy": frame_bbox,
        "frame_center_offset_px": [
            round(frame_center[0] - image_center[0], 4),
            round(frame_center[1] - image_center[1], 4),
        ],
    }
    return framed_rgb, framed_rgba, stats


def _compute_mask_stats(alpha_uint8: np.ndarray) -> tuple[float, float]:
    alpha = alpha_uint8.astype(np.float32) / 255.0
    coverage = float(np.mean(alpha > 0.5))
    soft_edge_fraction = float(np.mean((alpha > 0.05) & (alpha < 0.95)))
    return coverage, soft_edge_fraction


def _infer_with_monitor(model: BiRefNet, image: Image.Image) -> tuple[Image.Image, float, float]:
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    stop_flag = threading.Event()

    def _monitor() -> None:
        nonlocal peak_rss
        while not stop_flag.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(0.01)

    watcher = threading.Thread(target=_monitor, daemon=True)
    watcher.start()
    t0 = time.perf_counter()
    rgba = model(image.copy())
    runtime_s = time.perf_counter() - t0
    stop_flag.set()
    watcher.join(timeout=1.0)
    peak_rss_mb = (peak_rss - start_rss) / (1024 * 1024)
    return rgba, runtime_s, peak_rss_mb


def _run_pipeline_preprocess(model: BiRefNet, image: Image.Image) -> RunResult:
    rgba, runtime_s, peak_rss_mb = _infer_with_monitor(model, image)
    rgba_np = np.array(rgba)
    alpha = rgba_np[:, :, 3]
    coverage, soft_edge_fraction = _compute_mask_stats(alpha)
    framed_rgb, framed_rgba, frame_stats = _apply_trellis_framing(rgba)

    return RunResult(
        rgba=rgba,
        framed_rgb=framed_rgb,
        framed_rgba=framed_rgba,
        runtime_s=runtime_s,
        peak_rss_mb=peak_rss_mb,
        alpha_coverage=coverage,
        soft_edge_fraction=soft_edge_fraction,
        source_bbox_xyxy=frame_stats["source_bbox_xyxy"],
        frame_bbox_xyxy=frame_stats["frame_bbox_xyxy"],
        frame_center_offset_px=frame_stats["frame_center_offset_px"],
    )


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def _preprocess_tensor_stats(image: Image.Image, image_size: int) -> dict[str, Any]:
    arr = np.array(image.resize((image_size, image_size), Image.Resampling.LANCZOS).convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "input_range": [float(tensor.min().item()), float(tensor.max().item())],
        "normalized_range": [float(normalized.min().item()), float(normalized.max().item())],
    }


def _paste_on_checker(image: Image.Image, cell: int = 24) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, (220, 220, 220))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, rgba.height, cell):
        for x in range(0, rgba.width, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                draw.rectangle([x, y, x + cell, y + cell], fill=(185, 185, 185))
    canvas.paste(rgba, mask=rgba.split()[3])
    return canvas


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (255, 255, 255))
    img = image.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    out.paste(img, (x, y))
    return out


def _contact_sheet(
    original: Image.Image,
    new_rgba: Image.Image,
    new_framed: Image.Image,
    baseline_rgba: Optional[Image.Image],
    output_path: Path,
) -> None:
    panel_size = (360, 360)
    header_h = 24
    titles = ["input", "new RGBA", "new framed RGB", "baseline/old"]
    panels = [
        _fit_panel(original.convert("RGB"), panel_size),
        _fit_panel(_paste_on_checker(new_rgba), panel_size),
        _fit_panel(new_framed.convert("RGB"), panel_size),
        _fit_panel(_paste_on_checker(baseline_rgba), panel_size) if baseline_rgba is not None else Image.new("RGB", panel_size, (245, 245, 245)),
    ]

    sheet = Image.new("RGB", (panel_size[0] * 4, panel_size[1] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (title, panel) in enumerate(zip(titles, panels)):
        x = i * panel_size[0]
        draw.text((x + 8, 4), title, fill=(0, 0, 0))
        sheet.paste(panel, (x, header_h))
    sheet.save(output_path)


def _load_warlord_from_zip(zip_path: Path) -> dict[str, Image.Image]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("reference.jpg") as f:
            reference = Image.open(f).convert("RGB")
            reference.load()
        with zf.open("cutout_square_rgba.png") as f:
            baseline_cutout = Image.open(f).convert("RGBA")
            baseline_cutout.load()
        with zf.open("trellis2_preprocessed.png") as f:
            baseline_preprocessed = Image.open(f).convert("RGB")
            baseline_preprocessed.load()
    return {
        "reference": reference,
        "baseline_cutout": baseline_cutout,
        "baseline_preprocessed": baseline_preprocessed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TRELLIS background preprocessing with pinned BiRefNet.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/background_removal/image_manifest.json"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("benchmarks/background_removal/images"),
    )
    parser.add_argument(
        "--warlord-zip",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/opt/cursor/artifacts/birefnet-benchmark"),
    )
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    warlord = _load_warlord_from_zip(args.warlord_zip)
    images: list[dict[str, Any]] = [
        {
            "id": "warlord-reference",
            "title": "Uploaded warlord reference",
            "license": "User-provided reference input",
            "source_url": "uploaded zip",
            "image": warlord["reference"],
            "baseline_rgba": warlord["baseline_cutout"],
            "baseline_preprocessed": warlord["baseline_preprocessed"],
        }
    ]

    for row in manifest["external_images"]:
        image_path = args.images_dir / row["filename"]
        images.append(
            {
                **row,
                "image": Image.open(image_path).convert("RGB"),
                "baseline_rgba": None,
                "baseline_preprocessed": None,
            }
        )

    new_model = BiRefNet()
    new_model.cpu()

    old_model: Optional[BiRefNet] = None
    old_model_error: Optional[str] = None
    try:
        old_model = BiRefNet(model_name=NON_COMMERCIAL_RMBG_MODEL_NAME, revision=None)
        old_model.cpu()
    except Exception as exc:
        old_model_error = f"{type(exc).__name__}: {exc}"

    results: list[dict[str, Any]] = []
    for row in images:
        image: Image.Image = row["image"]
        new_result = _run_pipeline_preprocess(new_model, image)

        baseline_iou = None
        baseline_kind = None
        old_runtime_s = None
        old_peak_rss_mb = None
        old_coverage = None
        old_soft_edges = None
        old_rgba = None

        if old_model is not None:
            old_result = _run_pipeline_preprocess(old_model, image)
            old_rgba = old_result.rgba
            old_runtime_s = old_result.runtime_s
            old_peak_rss_mb = old_result.peak_rss_mb
            old_coverage = old_result.alpha_coverage
            old_soft_edges = old_result.soft_edge_fraction
            old_mask = np.array(old_result.rgba)[:, :, 3] > THRESHOLD_ALPHA
            new_mask = np.array(new_result.rgba)[:, :, 3] > THRESHOLD_ALPHA
            baseline_iou = _mask_iou(new_mask, old_mask)
            baseline_kind = "rmbg_2_0"
        elif row["baseline_rgba"] is not None:
            baseline_mask = np.array(row["baseline_rgba"].resize(new_result.rgba.size, Image.Resampling.BILINEAR))[:, :, 3] > THRESHOLD_ALPHA
            new_mask = np.array(new_result.rgba)[:, :, 3] > THRESHOLD_ALPHA
            baseline_iou = _mask_iou(new_mask, baseline_mask)
            baseline_kind = "provided_cutout_square_rgba"

        row_artifact = args.artifacts_dir / f"contact_sheet_{row['id']}.png"
        _contact_sheet(
            original=image,
            new_rgba=new_result.rgba,
            new_framed=new_result.framed_rgb,
            baseline_rgba=old_rgba if old_rgba is not None else row["baseline_rgba"],
            output_path=row_artifact,
        )

        preproc_stats_512 = _preprocess_tensor_stats(new_result.framed_rgb, image_size=512)
        preproc_stats_1024 = _preprocess_tensor_stats(new_result.framed_rgb, image_size=1024)
        preproc_stats_518 = _preprocess_tensor_stats(new_result.framed_rgb, image_size=518)

        result_row = {
            "id": row["id"],
            "title": row["title"],
            "license": row["license"],
            "source_url": row["source_url"],
            "edge_case": row.get("edge_case"),
            "new_runtime_s": new_result.runtime_s,
            "new_peak_rss_mb": new_result.peak_rss_mb,
            "new_alpha_coverage": new_result.alpha_coverage,
            "new_soft_edge_fraction": new_result.soft_edge_fraction,
            "new_source_bbox_xyxy": new_result.source_bbox_xyxy,
            "new_frame_bbox_xyxy": new_result.frame_bbox_xyxy,
            "new_frame_size": list(new_result.framed_rgb.size),
            "new_frame_center_offset_px": new_result.frame_center_offset_px,
            "old_runtime_s": old_runtime_s,
            "old_peak_rss_mb": old_peak_rss_mb,
            "old_alpha_coverage": old_coverage,
            "old_soft_edge_fraction": old_soft_edges,
            "mask_iou_against_old_or_baseline": baseline_iou,
            "comparison_basis": baseline_kind,
            "contact_sheet": str(row_artifact),
            "preprocess_tensor_512": preproc_stats_512,
            "preprocess_tensor_1024": preproc_stats_1024,
            "preprocess_tensor_518": preproc_stats_518,
        }

        if row["baseline_preprocessed"] is not None:
            result_row["provided_preprocessed_size"] = list(row["baseline_preprocessed"].size)
        results.append(result_row)
        print(
            f"{row['id']}: runtime={new_result.runtime_s:.3f}s peak_delta={new_result.peak_rss_mb:.1f}MB "
            f"coverage={new_result.alpha_coverage:.4f} soft={new_result.soft_edge_fraction:.4f}"
        )

    payload = {
        "model": {
            "repo": "ZhengPeng7/BiRefNet",
            "revision": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
            "license": "MIT",
        },
        "old_rmbg_attempt": {
            "model_name": NON_COMMERCIAL_RMBG_MODEL_NAME,
            "reachable_without_gated_terms": old_model is not None,
            "error": old_model_error,
        },
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "results": results,
    }

    output_json = args.artifacts_dir / "benchmark_results.json"
    output_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
