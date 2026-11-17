"""Measure a real 1024 cascade, extraction and bake; never synthesize GPU data."""
import argparse
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--vram-profile', choices=['none', '8gb', '8gb_1024'], default='8gb_1024')
    parser.add_argument('--low-vram', choices=['on', 'off'], default='on',
                        help='Legacy model offload for the none baseline; 8gb_1024 always enables it.')
    parser.add_argument('--decoder-chunk-size', type=int, default=32768)
    parser.add_argument('--max-num-tokens', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--texture-size', type=int, default=2048)
    parser.add_argument('--texture-resolution', type=int, choices=[512, 1024], default=None,
                        help='Opt-in texture flow selection on the unchanged 1024 shape; default is 1024.')
    parser.add_argument('--decimation-target', type=int, default=500000)
    parser.add_argument('--baker', choices=['nvdiffrast', 'zyrex'], default='nvdiffrast')
    parser.add_argument('--texture-diagnostics', action='store_true',
                        help='Report decoded voxel channel statistics on CPU before baking.')
    parser.add_argument('--texture-trace', action='store_true',
                        help='Trace texture conditioning, sampled latents, normalization and raw decoder output; affects timing.')
    parser.add_argument('--texture-decoder-dtype', choices=['checkpoint', 'fp32'], default='checkpoint',
                        help='Diagnostic arithmetic-only decoder precision override; fp32 may exceed 8 GB.')
    parser.add_argument('--texture-flow-autocast', choices=['profile', 'checkpoint'], default='profile',
                        help='Diagnostic: checkpoint disables extra autocast only during texture sampling.')
    parser.add_argument('--texture-compare-only', action='store_true',
                        help='Trace through texture decode, skipping all mesh extraction and baking.')
    args = parser.parse_args(argv)
    if args.texture_resolution is not None and args.vram_profile == '8gb':
        parser.error('--texture-resolution requires 1024_cascade, not the 8gb/512 pipeline')
    if args.texture_compare_only:
        if os.environ.get('ZYREX_LOW_VRAM_DTYPE'):
            parser.error('clear ZYREX_LOW_VRAM_DTYPE for an unmodified texture comparison')
        if args.vram_profile == '8gb' or (args.vram_profile == 'none' and args.low_vram != 'off'):
            parser.error('texture comparison requires none --low-vram off, or 8gb_1024')
        if args.texture_decoder_dtype != 'checkpoint' or args.texture_flow_autocast != 'profile':
            parser.error('texture comparison does not allow precision overrides')
        args.texture_trace = True
    return args


def texture_channel_stats(mesh):
    """Inspect pre-bake voxels, not UV pixels or surface-area-weighted samples.

    Copy bounded chunks to CPU so diagnostics do not allocate large GPU masks.
    Do not clamp or reorder predictions to hide material errors.
    """
    import torch
    result = {}
    for name, channel_slice in mesh.layout.items():
        count = finite_count = near_one = 0
        total = 0.0
        low, high = float('inf'), float('-inf')
        for start in range(0, mesh.attrs.shape[0], 65536):
            values = mesh.attrs[start:start + 65536, channel_slice].detach().cpu().double()
            count += values.numel()
            values = values[torch.isfinite(values)]
            finite_count += values.numel()
            if values.numel():
                total += values.sum().item()
                low = min(low, values.min().item())
                high = max(high, values.max().item())
                near_one += int((values >= 0.99).sum().item())
        result[name] = dict(channel_start=channel_slice.start, channel_stop=channel_slice.stop,
                            value_count=count, finite_count=finite_count,
                            min=low if finite_count else None, max=high if finite_count else None,
                            mean=total / finite_count if finite_count else None,
                            fraction_ge_0_99=near_one / finite_count if finite_count else None)
    return result


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
                  target_reserved_gib=6.5, target_met=None, run_metrics=None,
                  target_scope='pytorch_allocator_reserved_only')
    pipeline = None
    profiler = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for real VRAM profiling; no measurements were taken')
        from PIL import Image
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        profiler = StageProfiler(torch.device('cuda'), cuda_active=True,
                                 empty_cache=args.vram_profile == '8gb_1024')
        report['cuda'] = dict(device_name=torch.cuda.get_device_name(0),
                              cuda_version=torch.version.cuda, torch_version=torch.__version__)
        with profiler.stage('load_models'):
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
            if args.texture_decoder_dtype == 'fp32':
                # Arithmetic-only ablation of the loaded weights, not a replacement
                # checkpoint or recovery of precision lost during initial loading.
                pipeline.models['tex_slat_decoder'].convert_to_fp32()
            pipeline.low_vram = args.vram_profile in ('8gb', '8gb_1024') or args.low_vram == 'on'
            pipeline.to(torch.device('cuda'))
        trace_context = nullcontext()
        flow_context = nullcontext()
        if args.texture_flow_autocast == 'checkpoint':
            from scripts.texture_diagnostics import checkpoint_texture_flow_precision
            flow_context = checkpoint_texture_flow_precision(pipeline)
        if args.texture_trace:
            from scripts.texture_diagnostics import trace_texture
            report['texture_trace'] = {}
            trace_context = trace_texture(pipeline, report['texture_trace'])
        with Image.open(args.image) as image, flow_context, trace_context:
            outputs, metrics = pipeline.run(
                image, seed=args.seed, pipeline_type='512' if args.vram_profile == '8gb' else '1024_cascade',
                vram_profile=args.vram_profile, profile_stages=args.vram_profile != '8gb', return_metrics=True,
                max_num_tokens=args.max_num_tokens, decoder_chunk_size=args.decoder_chunk_size,
                texture_compare_only=args.texture_compare_only, texture_resolution=args.texture_resolution)
        report['run_metrics'] = metrics
        if args.texture_compare_only:
            report.update(status='ok', diagnostic_only=True, stopped_after='texture_decode')
            return
        import o_voxel
        mesh = outputs[0]
        if args.texture_diagnostics or args.texture_trace:
            with profiler.stage('texture_diagnostics'):
                report['decoded_voxel_channels'] = texture_channel_stats(mesh)
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
        if args.texture_trace:
            from scripts.texture_diagnostics import baked_metallic_stats
            report['baked_metallic'] = baked_metallic_stats(output / 'asset.glb')
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
        if report['status'] == 'ok' and peak is not None and not args.texture_compare_only:
            report['target_met'] = (peak <= 6.5 * 1024**3 and
                                    report['run_metrics'].get('output_resolution') == 1024)
        (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
