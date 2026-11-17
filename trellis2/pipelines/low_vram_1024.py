"""Explicit, experimental 1024 cascade execution and laptop profiling.

No resolution fallback. The 6.5 GiB target must be validated on real hardware.
"""
from contextlib import nullcontext
import os
import time

import torch

from ..representations import MeshWithVoxel
from ..utils.memory import StageProfiler, chunk_ranges, fp32_conditioning


def run_1024(pipeline, image, *, profile_name, pipeline_type, num_samples, seed,
             preprocess_image, return_latent, return_metrics, max_num_tokens,
             decoder_chunk_size, sparse_params, shape_params, tex_params, texture_compare_only=False,
             texture_resolution=None):
    optimized = profile_name == '8gb_1024'
    if texture_resolution is not None and (isinstance(texture_resolution, bool) or texture_resolution not in (512, 1024)):
        raise ValueError('texture_resolution must be 512 or 1024')
    texture_resolution = 1024 if texture_resolution is None else texture_resolution
    texture_model = f'tex_slat_flow_model_{texture_resolution}'
    if texture_model not in pipeline.models:
        raise ValueError(f'Missing requested texture flow model: {texture_model}')
    if texture_compare_only and return_latent:
        raise ValueError('texture_compare_only returns diagnostics, not retained latents')
    if profile_name not in (None, '8gb_1024'):
        raise ValueError('Detailed 1024 profiling supports none or 8gb_1024')
    if pipeline_type not in (None, '1024_cascade'):
        raise ValueError('8gb_1024 / profile_stages requires 1024_cascade; no resolution fallback')
    if optimized and num_samples != 1:
        raise ValueError('8gb_1024 supports one sample per run')
    token_limit = 49152 if max_num_tokens is None else max_num_tokens
    if isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit <= 0:
        raise ValueError('max_num_tokens must be a positive integer')
    if decoder_chunk_size is None:
        decoder_chunk_size = int(os.environ.get('ZYREX_DECODER_CHUNK_SIZE', '32768'))
    chunk_ranges(0, decoder_chunk_size)  # Validate before moving any model.
    cfg = pipeline.vram_profiles.get(profile_name, {})
    dtype = pipeline._resolve_autocast_dtype(cfg)
    pipeline._configure_attention_backend_for_profile(cfg)
    cuda_active = pipeline._is_cuda_runtime_active()
    profiler = StageProfiler(pipeline.device, cuda_active=cuda_active, empty_cache=optimized)
    stats = dict(vram_profile=profile_name, pipeline_type='1024_cascade',
                 texture_resolution=texture_resolution, texture_flow_model=texture_model,
                 conditioning_precision='fp32_autocast_off' if optimized else 'upstream',
                 output_resolution=None, low_vram=bool(optimized or pipeline.low_vram),
                 device=str(pipeline.device), max_num_tokens=token_limit,
                 decoder_chunk_size=decoder_chunk_size if optimized else None,
                 autocast_dtype=pipeline._dtype_name(dtype), status='failed',
                 stage_metrics=profiler.metrics, measurement_scope='pytorch_allocator',
                 target_reserved_gib=6.5, target_met=None,
                 target_scope='pytorch_allocator_reserved_only')
    pipeline.last_run_stats = stats
    started = time.perf_counter()
    previous_low_vram = pipeline.low_vram
    saved = []
    missing = object()

    def temporary(obj, key, value):
        saved.append((obj, key, getattr(obj, key, missing)))
        setattr(obj, key, value)

    def offload_all():
        for model in [*pipeline.models.values(), pipeline.image_cond_model, pipeline.rembg_model]:
            if model is not None:
                model.cpu()

    def release_cache():
        if optimized and cuda_active:
            torch.cuda.empty_cache()

    def stage(name, fn, autocast=False):
        with profiler.stage(name):
            context = (torch.autocast('cuda', dtype=dtype)
                       if autocast and cuda_active and dtype is not None else nullcontext())
            with context:
                return fn()

    def upsample(slat):
        decoder = pipeline.models['shape_slat_decoder']
        if pipeline.low_vram and not optimized:
            decoder.to(pipeline.device)
        try:
            return decoder.upsample(slat, upsample_times=4)
        finally:
            if pipeline.low_vram:
                decoder.cpu()

    try:
        pipeline.low_vram = bool(previous_low_vram or optimized)
        if not pipeline.low_vram and getattr(pipeline, '_models_offloaded_for_1024', False):
            stage('restore_model_residency', lambda: pipeline.to(pipeline.device))
        if optimized:
            # Also handles callers that previously put the whole pipeline on CUDA.
            stage('offload_idle_models', offload_all)
            for name in ('shape_slat_decoder', 'tex_slat_decoder'):
                decoder = pipeline.models[name]
                temporary(decoder, 'stream_device', pipeline.device)
                temporary(decoder, 'output_chunk_size', decoder_chunk_size)
            sparse_params = {**sparse_params, 'retain_history': False}
            shape_params = {**shape_params, 'retain_history': False}
            tex_params = {**tex_params, 'retain_history': False}
        for name in ('shape_slat_decoder', 'tex_slat_decoder'):
            temporary(pipeline.models[name], 'low_vram', pipeline.low_vram)
        temporary(pipeline.models['shape_slat_decoder'], 'stage_profiler', profiler)
        if preprocess_image:
            image = stage('preprocess_image', lambda: pipeline.preprocess_image(image))
        torch.manual_seed(seed)
        def condition(resolution):
            if optimized:
                with fp32_conditioning(pipeline.image_cond_model, pipeline.device):
                    return pipeline.get_cond([image], resolution)
            return pipeline.get_cond([image], resolution)
        cond_512 = stage('condition_512', lambda: condition(512), not optimized)
        cond_1024 = stage('condition_1024', lambda: condition(1024), not optimized)
        if optimized:
            cond_1024 = {key: value.cpu() for key, value in cond_1024.items()}
            release_cache()
        coords = stage('sparse_structure', lambda: pipeline.sample_sparse_structure(
            cond_512, 32, num_samples, sparse_params), True)
        if not optimized:
            # Profile the original cascade with its original tensor lifetimes.
            temporary(pipeline, '_stage_profiler', profiler)
            shape_slat, _ = stage('shape_slat_1024_cascade', lambda: pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024, pipeline.models['shape_slat_flow_model_512'],
                pipeline.models['shape_slat_flow_model_1024'], 512, 1024,
                coords, shape_params, token_limit), True)
            stats['num_tokens_1024'] = int(shape_slat.coords.shape[0])
        else:
            low_shape = stage('shape_slat_512', lambda: pipeline.sample_shape_slat(
                cond_512, pipeline.models['shape_slat_flow_model_512'], coords, shape_params), True)
            if optimized:
                if texture_resolution == 512:
                    cond_512 = {key: value.cpu() for key, value in cond_512.items()}
                else:
                    del cond_512
                del coords
                release_cache()
            # Sparse convolution requires matching feature/weight dtypes. Let the
            # decoder use its checkpoint precision; flow autocast is not safe here.
            hr_coords = stage('cascade_upsample', lambda: upsample(low_shape))
            if optimized:
                del low_shape
                release_cache()
            with profiler.stage('cascade_coordinates'):
                coords = torch.cat([
                    hr_coords[:, :1],
                    ((hr_coords[:, 1:] + 0.5) / 512 * (1024 // 16)).int(),
                ], dim=1).unique(dim=0)
                stats['num_tokens_1024'] = int(coords.shape[0])
                if optimized and coords.shape[0] > token_limit:
                    raise ValueError(
                        f'1024 cascade needs {coords.shape[0]} tokens, above max_num_tokens={token_limit}. '
                        'No resolution downgrade performed. Increase the cap explicitly only after profiling.')
            if optimized:
                del hr_coords
                cond_1024 = {key: value.to(pipeline.device) for key, value in cond_1024.items()}
                release_cache()
            shape_slat = stage('shape_slat_1024', lambda: pipeline.sample_shape_slat(
                cond_1024, pipeline.models['shape_slat_flow_model_1024'], coords, shape_params), True)
        texture_cond = cond_1024 if texture_resolution == 1024 else {
            key: value.to(pipeline.device) for key, value in cond_512.items()}
        tex_slat = stage(f'tex_slat_{texture_resolution}', lambda: pipeline.sample_tex_slat(
            texture_cond, pipeline.models[texture_model], shape_slat, tex_params), True)
        del texture_cond
        if optimized and texture_resolution == 512:
            del cond_512
        if optimized:
            del cond_1024, coords
            release_cache()
        if texture_compare_only:
            # Decode subdivision guidance using the same shape decoder forward,
            # but bypass its mesh conversion (and every later mesh/bake stage).
            decoder = pipeline.models['shape_slat_decoder']
            decoder.set_resolution(1024)
            _, subs = stage('shape_decode_guidance', lambda: decoder(
                shape_slat, return_subs=True, skip_mesh=True))
            del _
            if optimized:
                del shape_slat
                release_cache()
            stage('texture_decode', lambda: pipeline.decode_tex_slat(tex_slat, subs))
            stats.update(status='ok', output_resolution=1024,
                         diagnostic_only=True, stopped_after='texture_decode')
            return ([], stats) if return_metrics else []
        meshes, subs = stage('shape_decode_and_extract', lambda: pipeline.decode_shape_slat(shape_slat, 1024))
        if optimized and not return_latent:
            del shape_slat
            release_cache()
        tex_voxels = stage('texture_decode', lambda: pipeline.decode_tex_slat(tex_slat, subs))
        if optimized:
            del subs
            if not return_latent:
                del tex_slat
            release_cache()
        with profiler.stage('mesh_finalize'):
            outputs = []
            for mesh, voxels in zip(meshes, tex_voxels):
                mesh.fill_holes()
                outputs.append(MeshWithVoxel(
                    mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / 1024, coords=voxels.coords[:, 1:], attrs=voxels.feats,
                    voxel_shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
                    layout=pipeline.pbr_attr_layout))
        stats.update(status='ok', output_resolution=1024)
    finally:
        try:
            if optimized:
                offload_all()
                pipeline._models_offloaded_for_1024 = True
                release_cache()
        finally:
            for obj, key, value in reversed(saved):
                if value is missing:
                    delattr(obj, key)
                else:
                    setattr(obj, key, value)
            pipeline.low_vram = previous_low_vram
            stats.update(profiler.peaks(), total_time_s=time.perf_counter() - started)
            # Pipeline-only result; the benchmark additionally includes bake/export.
            peak = stats['peak_vram_reserved_bytes']
            stats['target_met'] = (peak <= 6.5 * 1024**3) if peak is not None and stats['status'] == 'ok' and not texture_compare_only else None
    if return_latent and return_metrics:
        return outputs, (shape_slat, tex_slat, 1024), stats
    if return_latent:
        return outputs, (shape_slat, tex_slat, 1024)
    if return_metrics:
        return outputs, stats
    return outputs
