import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

def _default_image_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "example_image" / "T.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TRELLIS.2 image-to-3D generation with low-VRAM settings and emit JSON metrics.",
    )
    parser.add_argument("--model-id", default="microsoft/TRELLIS.2-4B", help="Model id or local model path.")
    parser.add_argument(
        "--image",
        default=str(_default_image_path()),
        help="Input image path. Use your warlord image path here for laptop validation.",
    )
    parser.add_argument("--output-dir", default="outputs/benchmarks", help="Directory for GLB and JSON output.")
    parser.add_argument(
        "--vram-profile",
        default="8gb",
        choices=["8gb", "none"],
        help="VRAM profile preset. Use 'none' to disable profile behavior.",
    )
    parser.add_argument(
        "--pipeline-type",
        default=None,
        choices=["512", "1024", "1024_cascade", "1536_cascade"],
        help="Override generation pipeline type.",
    )
    parser.add_argument("--max-num-tokens", type=int, default=None, help="Optional token cap override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--attention-backend", default=None, choices=["sdpa", "xformers", "flash_attn"])
    parser.add_argument(
        "--dump-baker-input",
        action="store_true",
        help="Save production to_glb baker inputs (mesh + sparse voxel attrs) for compare tooling.",
    )
    parser.add_argument(
        "--baker-input-path",
        default=None,
        help="Optional explicit output path for baker-input npz dump.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    vram_profile = None if args.vram_profile == "none" else args.vram_profile
    if vram_profile == "8gb":
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    if args.attention_backend is not None:
        os.environ["ATTN_BACKEND"] = args.attention_backend
        os.environ["SPARSE_ATTN_BACKEND"] = args.attention_backend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # Lazy import so --help works in environments missing optional runtime deps.
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.to(torch.device("cuda"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wall_start = time.perf_counter()
    outputs, run_metrics = pipeline.run(
        image=image,
        num_samples=args.num_samples,
        seed=args.seed,
        return_metrics=True,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_num_tokens,
        vram_profile=vram_profile,
    )
    mesh = outputs[0]
    mesh.simplify(16_777_216)  # nvdiffrast face-limit guard
    bake_args_meta = {
        "decimation_target": int(args.decimation_target),
        "texture_size": int(args.texture_size),
        "remesh": True,
        "remesh_band": 1,
        "remesh_project": 0,
    }
    to_glb_kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "attr_volume": mesh.attrs,
        "coords": mesh.coords,
        "attr_layout": mesh.layout,
        "grid_size": int(round(1.0 / mesh.voxel_size)),
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "decimation_target": bake_args_meta["decimation_target"],
        "texture_size": bake_args_meta["texture_size"],
        "remesh": bake_args_meta["remesh"],
        "remesh_band": bake_args_meta["remesh_band"],
        "remesh_project": bake_args_meta["remesh_project"],
        "use_tqdm": True,
    }
    baker_input_path = None
    if args.dump_baker_input:
        if args.baker_input_path is not None:
            baker_input_path = Path(args.baker_input_path)
        else:
            baker_input_path = output_dir / f"low_vram_baker_input_{ts}.npz"
        baker_input_path.parent.mkdir(parents=True, exist_ok=True)

        layout_meta = {
            key: [int(sl.start), int(sl.stop)]
            for key, sl in mesh.layout.items()
        }
        np_payload = {
            "vertices": mesh.vertices.detach().cpu().numpy(),
            "faces": mesh.faces.detach().cpu().numpy(),
            "attrs": mesh.attrs.detach().cpu().numpy(),
            "coords": mesh.coords.detach().cpu().numpy(),
            "aabb": np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32),
            "voxel_size": np.array([mesh.voxel_size], dtype=np.float32),
            "grid_size": np.array([int(round(1.0 / mesh.voxel_size))] * 3, dtype=np.int32),
            "attr_layout_json": np.array([json.dumps(layout_meta)], dtype=np.str_),
            "to_glb_args_json": np.array([json.dumps(bake_args_meta)], dtype=np.str_),
        }
        np.savez_compressed(str(baker_input_path), **np_payload)

    glb = o_voxel.postprocess.to_glb(**to_glb_kwargs)
    glb_path = output_dir / f"low_vram_benchmark_{ts}.glb"
    glb.export(str(glb_path), extension_webp=True)
    wall_total = time.perf_counter() - wall_start

    payload = {
        "timestamp_utc": ts,
        "input_image": str(Path(args.image).resolve()),
        "output_glb_path": str(glb_path.resolve()),
        "total_time_s": wall_total,
        "settings": {
            "model_id": args.model_id,
            "vram_profile": vram_profile,
            "pipeline_type": args.pipeline_type,
            "max_num_tokens": args.max_num_tokens,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "decimation_target": args.decimation_target,
            "texture_size": args.texture_size,
            "attention_backend": os.environ.get("ATTN_BACKEND"),
            "low_vram_env": os.environ.get("ZYREX_LOW_VRAM"),
            "low_vram_dtype_env": os.environ.get("ZYREX_LOW_VRAM_DTYPE"),
            "baker_input_dump_path": str(baker_input_path.resolve()) if baker_input_path is not None else None,
        },
        "run_metrics": run_metrics,
        "cuda": {
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        },
    }
    metrics_path = output_dir / f"low_vram_benchmark_{ts}.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote benchmark report: {metrics_path}")


if __name__ == "__main__":
    main()
