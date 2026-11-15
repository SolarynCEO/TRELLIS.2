"""Measure a real 1024 cascade, extraction and bake; never synthesize GPU data."""
import argparse
import json
import os
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--vram-profile', choices=['none', '8gb_1024'], default='8gb_1024')
    parser.add_argument('--low-vram', choices=['on', 'off'], default='on',
                        help='Legacy model offload for the none baseline; 8gb_1024 always enables it.')
    parser.add_argument('--decoder-chunk-size', type=int, default=32768)
    parser.add_argument('--max-num-tokens', type=int, default=49152)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--decimation-target', type=int, default=500000)
    parser.add_argument('--baker', choices=['nvdiffrast', 'zyrex'], default='nvdiffrast')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    # Must be configured before importing torch / initializing CUDA. Some Windows
    # builds ignore expandable_segments with a warning; report the setting only,
    # not a claim that the allocator supports it.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ['ATTN_BACKEND'] = 'sdpa'
    os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
    os.environ['ZYREX_BAKER'] = args.baker
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from trellis2.utils.memory import StageProfiler

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='failed', settings=vars(args),
                  allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                  allocator_config_override=os.environ.get('PYTORCH_ALLOC_CONF'),
                  low_vram_dtype=os.environ.get('ZYREX_LOW_VRAM_DTYPE'),
                  measurement_scope='PyTorch allocator only; excludes direct native CUDA allocations',
                  target_reserved_gib=6.5, target_met=None, run_metrics=None)
    pipeline = None
    profiler = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real VRAM profiling; no measurements were taken')
        from PIL import Image
        import o_voxel
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        profiler = StageProfiler(torch.device('cuda'), cuda_active=True,
                                 empty_cache=args.vram_profile == '8gb_1024')
        report['cuda'] = dict(device_name=torch.cuda.get_device_name(0),
                              cuda_version=torch.version.cuda, torch_version=torch.__version__)
        with profiler.stage('load_models'):
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
            pipeline.low_vram = args.vram_profile == '8gb_1024' or args.low_vram == 'on'
            pipeline.to(torch.device('cuda'))
        with Image.open(args.image) as image:
            outputs, metrics = pipeline.run(
                image, seed=args.seed, pipeline_type='1024_cascade',
                vram_profile=args.vram_profile, profile_stages=True, return_metrics=True,
                max_num_tokens=args.max_num_tokens, decoder_chunk_size=args.decoder_chunk_size)
        report['run_metrics'] = metrics
        mesh = outputs[0]
        with profiler.stage('mesh_simplify'):
            mesh.simplify(16_777_216)
        with profiler.stage('bake'):
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout,
                grid_size=int(round(1.0 / mesh.voxel_size)),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target, texture_size=args.texture_size,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
        with profiler.stage('export'):
            glb.export(str(output / 'asset.glb'), extension_webp=True)
        report['status'] = 'ok'
    except Exception as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if pipeline is not None:
            report['run_metrics'] = pipeline.last_run_stats
        stages = list((report.get('run_metrics') or {}).get('stage_metrics', []))
        stages += profiler.metrics if profiler is not None else []
        report['stage_metrics'] = stages
        for key in ('peak_vram_allocated_bytes', 'peak_vram_reserved_bytes'):
            values = [s[key] for s in stages if s[key] is not None]
            pipeline_peak = (report.get('run_metrics') or {}).get(key)
            outer_peak = profiler.peaks()[key] if profiler is not None else None
            values.extend(value for value in (pipeline_peak, outer_peak) if value is not None)
            report[key] = max(values, default=None)
        peak = report['peak_vram_reserved_bytes']
        if report['status'] == 'ok' and peak is not None:
            report['target_met'] = (peak <= 6.5 * 1024**3 and
                                    report['run_metrics']['output_resolution'] == 1024)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
