"""Read-only texture diagnostics for paired laptop runs; no model/output edits."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import struct
import hashlib
import inspect


def tensor_fingerprint(value):
    """Exact bytes/dtype/shape fingerprint; supports bfloat16 without numpy casts."""
    import torch
    value = getattr(value, 'feats', value)
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    rows = value.reshape(-1, value.shape[-1])
    for start in range(0, rows.shape[0], 16384):
        chunk = rows[start:start + 16384].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_precision(model):
    input_layer = getattr(model, 'from_latent', getattr(model, 'input_layer', None))
    output_layer = getattr(model, 'output_layer', getattr(model, 'out_layer', None))
    return dict(configured_dtype=str(getattr(model, 'dtype', None)),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                input_projection_dtype=str(input_layer.weight.dtype) if hasattr(input_layer, 'weight') else None,
                output_projection_dtype=str(output_layer.weight.dtype) if hasattr(output_layer, 'weight') else None)


@contextmanager
def checkpoint_texture_flow_precision(pipeline):
    """Diagnostic only: disable outer autocast for the texture sampler.

    The flow's existing manual_cast then selects its configured checkpoint dtype;
    other conditioning/shape stages and all model parameters remain unchanged.
    """
    import torch
    sampler = pipeline.tex_slat_sampler
    original = sampler.sample
    had_override = 'sample' in vars(sampler)
    old_override = vars(sampler).get('sample')
    def sample(*args, **kwargs):
        with torch.autocast(device_type=torch.device(pipeline.device).type, enabled=False):
            return original(*args, **kwargs)
    try:
        sampler.sample = sample
        yield
    finally:
        if had_override:
            sampler.sample = old_override
        else:
            del sampler.sample


def tensor_stats(value, *, scale=1.0, offset=0.0):
    import torch
    value = getattr(value, 'feats', value)
    shape = list(value.shape)
    rows = value.reshape(-1, shape[-1])
    count = torch.zeros(shape[-1], dtype=torch.float64)
    total, squares = count.clone(), count.clone()
    low, high = torch.full_like(count, float('inf')), torch.full_like(count, float('-inf'))
    for start in range(0, rows.shape[0], 16384):
        chunk = rows[start:start + 16384].detach().cpu().double() * scale + offset
        finite = torch.isfinite(chunk)
        count += finite.sum(0)
        clean = torch.where(finite, chunk, 0)
        total += clean.sum(0)
        squares += clean.square().sum(0)
        low = torch.minimum(low, torch.where(finite, chunk, float('inf')).amin(0))
        high = torch.maximum(high, torch.where(finite, chunk, float('-inf')).amax(0))
    channels = []
    for i in range(len(count)):
        n = int(count[i])
        mean = float(total[i] / n) if n else None
        channels.append(dict(finite_count=n, min=float(low[i]) if n else None,
                             max=float(high[i]) if n else None, mean=mean,
                             std=(max(0, float(squares[i] / n) - mean**2)**0.5) if n else None))
    return dict(shape=shape, dtype=str(value.dtype), channels=channels)


@contextmanager
def trace_texture(pipeline, report):
    """Capture the actual sampler boundary and decoder boundary, including scale.

    Hooks copy bounded chunks to CPU. Results are statistics only, with no tensor
    references retained across stages. Tracing changes timings; not for benchmarks.
    """
    sampler = pipeline.tex_slat_sampler
    decoder = pipeline.models['tex_slat_decoder']
    original = sampler.sample
    had_override = 'sample' in vars(sampler)
    old_override = vars(sampler).get('sample')
    report['shape_normalization'] = pipeline.shape_slat_normalization
    report['texture_normalization'] = pipeline.tex_slat_normalization
    report['decode_transform'] = 'raw * 0.5 + 0.5 (no clamp)'
    report['checkpoint_models'] = getattr(pipeline, '_pretrained_args', {}).get('models', {})
    report['schema_version'] = 2
    saved_methods = {}

    def wrap(name, factory):
        original_method = getattr(pipeline, name)
        saved_methods[name] = (name in vars(pipeline), vars(pipeline).get(name))
        setattr(pipeline, name, factory(original_method))

    def preprocessing(original_method):
        def call(*args, **kwargs):
            image = original_method(*args, **kwargs)
            report['preprocessed_image'] = dict(mode=image.mode, size=list(image.size),
                sha256=hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest())
            return image
        return call

    def conditioning(original_method):
        def call(image, resolution, *args, **kwargs):
            import torch
            encoder = getattr(pipeline.image_cond_model, 'model', pipeline.image_cond_model)
            report[f'conditioning_precision_{resolution}'] = dict(
                parameter_dtypes=sorted({str(p.dtype) for p in encoder.parameters()}),
                autocast_enabled=torch.is_autocast_enabled(torch.device(pipeline.device).type))
            result = original_method(image, resolution, *args, **kwargs)
            report[f'conditioning_{resolution}'] = {
                key: dict(sha256=tensor_fingerprint(value), moments=tensor_stats(value))
                for key, value in result.items()}
            return result
        return call

    def texture_sampling(original_method):
        def call(cond, model, shape, *args, **kwargs):
            report['shape_slat'] = dict(coords_sha256=tensor_fingerprint(shape.coords),
                feats_sha256=tensor_fingerprint(shape), moments=tensor_stats(shape))
            return original_method(cond, model, shape, *args, **kwargs)
        return call

    def sample(model, noise, **kwargs):
        import torch
        report['flow_model'] = next((name for name, m in pipeline.models.items() if m is model), 'unknown')
        report['flow_precision'] = model_precision(model)
        report['noise'] = tensor_stats(noise)
        report['noise_sha256'] = tensor_fingerprint(noise)
        report['noise_coords_sha256'] = tensor_fingerprint(noise.coords)
        for key in ('cond', 'neg_cond', 'concat_cond'):
            if key in kwargs:
                report[key] = tensor_stats(kwargs[key])
                report[key + '_sha256'] = tensor_fingerprint(kwargs[key])
        report['sampler_params'] = {k: v for k, v in kwargs.items()
                                    if isinstance(v, (str, int, float, bool, type(None), list, tuple))}
        # Include defaults that the caller did not explicitly pass, without tensors.
        for key, parameter in inspect.signature(original).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                value = kwargs.get(key, parameter.default)
                if isinstance(value, (str, int, float, bool, type(None), list, tuple)):
                    report['sampler_params'][key] = value
        report['sampler_params']['sigma_min'] = getattr(sampler, 'sigma_min', None)
        def flow_before(module, args):
            if 'flow_forward_precision' not in report:
                device_type = args[0].device.type
                report['flow_forward_precision'] = dict(
                    input_dtype=str(args[0].dtype),
                    autocast_enabled=torch.is_autocast_enabled(device_type),
                    autocast_dtype=str(torch.get_autocast_dtype(device_type)))
        handle = model.register_forward_pre_hook(flow_before)
        try:
            result = original(model, noise, **kwargs)
        finally:
            handle.remove()
        report['predicted_normalized_tex_slat'] = tensor_stats(result.samples)
        report['predicted_normalized_tex_slat_sha256'] = tensor_fingerprint(result.samples)
        return result

    def before(module, args):
        report['decoder_input_denormalized_tex_slat'] = tensor_stats(args[0])
        report['decoder_dtype'] = str(module.dtype)
        report['decoder_precision'] = model_precision(module)
        report['decoder_input_sha256'] = tensor_fingerprint(args[0])
        if hasattr(args[0], 'coords'):
            report['decoder_input_coords_sha256'] = tensor_fingerprint(args[0].coords)
        report['decoder_output_chunk_size'] = getattr(module, 'output_chunk_size', None)

    def after(module, args, output):
        report['raw_decoder_output'] = tensor_stats(output)
        report['raw_decoder_output_sha256'] = tensor_fingerprint(output)
        report['scaled_decoder_output'] = tensor_stats(output, scale=0.5, offset=0.5)

    handles = []
    try:
        wrap('preprocess_image', preprocessing)
        wrap('get_cond', conditioning)
        wrap('sample_tex_slat', texture_sampling)
        sampler.sample = sample
        handles.append(decoder.register_forward_pre_hook(before))
        handles.append(decoder.register_forward_hook(after))
        yield
    finally:
        for name, (had_value, value) in saved_methods.items():
            if had_value:
                setattr(pipeline, name, value)
            else:
                delattr(pipeline, name)
        for handle in handles:
            handle.remove()
        if had_override:
            sampler.sample = old_override
        else:
            del sampler.sample


def baked_metallic_stats(path):
    """Blue-channel >= .95 ratio over encoded texture pixels, including padding.

    Reads embedded PNG/JPEG/EXT_texture_webp GLBs. Reports every material and uses
    the maximum material ratio, so low-metal materials cannot hide saturation.
    This is not a surface-area-weighted metric. Factors are deliberately excluded
    to prevent a metallicFactor override from concealing saturated predictions.
    """
    import numpy as np
    from PIL import Image
    data = Path(path).read_bytes()
    if len(data) < 12 or struct.unpack_from('<4sII', data) != (b'glTF', 2, len(data)):
        raise ValueError('Expected a complete GLB 2.0 file')
    offset, document, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        if len(chunk) != length:
            raise ValueError('Truncated GLB chunk')
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += length
    if document is None or binary is None:
        raise ValueError('GLB requires JSON and embedded texture buffer')
    materials = []
    for index, material in enumerate(document.get('materials', [])):
        pbr = material.get('pbrMetallicRoughness', {})
        info = pbr.get('metallicRoughnessTexture')
        if info is None:
            continue
        texture = document['textures'][info['index']]
        source = texture.get('extensions', {}).get('EXT_texture_webp', {}).get('source', texture.get('source'))
        image = document['images'][source]
        if 'bufferView' not in image:
            raise ValueError('Regression metric requires embedded GLB textures')
        view = document['bufferViews'][image['bufferView']]
        start, length = view.get('byteOffset', 0), view['byteLength']
        if view.get('buffer', 0) != 0 or start < 0 or length <= 0 or start + length > len(binary):
            raise ValueError('Invalid embedded image buffer view')
        with Image.open(BytesIO(binary[start:start + length])) as decoded:
            blue = np.asarray(decoded.convert('RGB'))[..., 2].astype(np.float64) / 255
        materials.append(dict(material=index, metallic_factor=pbr.get('metallicFactor', 1),
                              pixel_count=int(blue.size), metallic_high_ratio=float((blue >= 0.95).mean())))
    if not materials:
        raise ValueError('No metallic-roughness texture found; regression must not silently pass')
    return dict(threshold=0.95, scope='encoded_blue_texture_pixels_including_padding',
                metallic_high_ratio=max(m['metallic_high_ratio'] for m in materials), materials=materials)


def assert_warlord_metallic(report):
    ratio = report['baked_metallic']['metallic_high_ratio']
    if not 0 <= ratio <= 0.5:
        raise AssertionError(f'Warlord metallic_high_ratio={ratio!r} exceeds 0.5; inspect texture_trace in metrics.json')


def compare_texture_reports(reports):
    """Compare supplied reports without claiming to have run their GPU workloads."""
    baseline = reports.get('baseline_1024', {}).get('texture_trace', {})
    result = {}
    for label, report in reports.items():
        trace = report['texture_trace']
        raw_metal = trace['raw_decoder_output']['channels'][3]
        keys = ('decoder_input_sha256', 'decoder_input_coords_sha256')
        match = (all(trace[key] == baseline[key] for key in keys)
                 if all(trace.get(key) and baseline.get(key) for key in keys) else None)
        result[label] = dict(raw_decoder_metallic=raw_metal,
                             flow_model=trace['flow_model'], flow_precision=trace.get('flow_precision'),
                             flow_forward_precision=trace.get('flow_forward_precision'),
                             decoder_precision=trace.get('decoder_precision'),
                             same_decoder_input_as_baseline_1024=match,
                             baked_metallic_high_ratio=report['baked_metallic']['metallic_high_ratio'])
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare texture-trace reports; no GPU work.')
    parser.add_argument('--report', action='append', required=True, help='LABEL=metrics.json; use baseline_1024 for the reference')
    args = parser.parse_args()
    reports = {}
    for entry in args.report:
        label, path = entry.split('=', 1)
        reports[label] = json.loads(Path(path).read_text(encoding='utf-8'))
    print(json.dumps(compare_texture_reports(reports), indent=2))
