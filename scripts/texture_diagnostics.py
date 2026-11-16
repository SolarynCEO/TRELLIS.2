"""Read-only texture diagnostics for paired laptop runs; no model/output edits."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import struct


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

    def sample(model, noise, **kwargs):
        report['flow_model'] = next((name for name, m in pipeline.models.items() if m is model), 'unknown')
        report['noise'] = tensor_stats(noise)
        for key in ('cond', 'neg_cond', 'concat_cond'):
            if key in kwargs:
                report[key] = tensor_stats(kwargs[key])
        report['sampler_params'] = {k: v for k, v in kwargs.items()
                                    if isinstance(v, (str, int, float, bool, type(None), list, tuple))}
        result = original(model, noise, **kwargs)
        report['predicted_normalized_tex_slat'] = tensor_stats(result.samples)
        return result

    def before(module, args):
        report['decoder_input_denormalized_tex_slat'] = tensor_stats(args[0])
        report['decoder_dtype'] = str(module.dtype)
        report['decoder_output_chunk_size'] = getattr(module, 'output_chunk_size', None)

    def after(module, args, output):
        report['raw_decoder_output'] = tensor_stats(output)
        report['scaled_decoder_output'] = tensor_stats(output, scale=0.5, offset=0.5)

    handles = []
    try:
        sampler.sample = sample
        handles.append(decoder.register_forward_pre_hook(before))
        handles.append(decoder.register_forward_hook(after))
        yield
    finally:
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
