"""Synthetic CPU checks for diagnostic accounting and the real-image GPU gate."""
from io import BytesIO
import json
import struct

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.texture_diagnostics import tensor_stats, baked_metallic_stats, assert_warlord_metallic


def _glb(path, blue, *, webp=False, factor=1.0, green=255):
    pixels = np.zeros((1, len(blue), 3), dtype=np.uint8)
    pixels[..., 1] = green
    pixels[..., 2] = blue
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format='WEBP' if webp else 'PNG', lossless=True)
    data = stream.getvalue()
    document = dict(asset={'version': '2.0'}, buffers=[{'byteLength': len(data)}],
                    bufferViews=[{'buffer': 0, 'byteOffset': 0, 'byteLength': len(data)}],
                    images=[{'bufferView': 0, 'mimeType': 'image/webp' if webp else 'image/png'}],
                    textures=[{'extensions': {'EXT_texture_webp': {'source': 0}}} if webp else {'source': 0}],
                    materials=[{'pbrMetallicRoughness': {'metallicFactor': factor, 'metallicRoughnessTexture': {'index': 0}}}])
    encoded = json.dumps(document).encode()
    encoded += b' ' * (-len(encoded) % 4)
    data += b'\0' * (-len(data) % 4)
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(data)) +
                     struct.pack('<II', len(encoded), 0x4E4F534A) + encoded +
                     struct.pack('<II', len(data), 0x004E4942) + data)


@pytest.mark.parametrize('webp', [False, True])
def test_baked_metric_uses_blue_not_green_and_detects_saturation(tmp_path, webp):
    path = tmp_path/'synthetic.glb'
    _glb(path, [0, 242, 243, 255], webp=webp)
    stats = baked_metallic_stats(path)
    assert stats['metallic_high_ratio'] == 0.5
    assert_warlord_metallic({'baked_metallic': stats})
    _glb(path, [255]*4, webp=webp, factor=0, green=0)
    with pytest.raises(AssertionError, match='exceeds 0.5'):
        assert_warlord_metallic({'baked_metallic': baked_metallic_stats(path)})


def test_reported_warlord_ratio_fails_gate():
    with pytest.raises(AssertionError):
        assert_warlord_metallic({'baked_metallic': {'metallic_high_ratio': 0.9805}})


def test_channel_moments_and_scaling_are_cpu_only():
    values = torch.tensor([[0., float('nan')], [2., 4.]])
    raw = tensor_stats(values)
    scaled = tensor_stats(values, scale=0.5, offset=0.5)
    assert raw['channels'][0]['mean'] == 1
    assert raw['channels'][0]['std'] == 1
    assert scaled['channels'][0]['mean'] == 1
    assert scaled['channels'][0]['std'] == 0.5
    assert scaled['channels'][1]['finite_count'] == 1
    assert scaled['channels'][1]['mean'] == 2.5


def test_fingerprints_detect_changed_latents_and_support_bf16():
    from scripts.texture_diagnostics import tensor_fingerprint
    values = torch.ones(2, 32, dtype=torch.bfloat16)
    expected = tensor_fingerprint(values)
    assert tensor_fingerprint(values.clone()) == expected
    assert tensor_fingerprint(values.float()) != expected
    values[1, 3] = 2
    assert tensor_fingerprint(values) != expected


def test_checkpoint_flow_precision_disables_only_its_scoped_autocast():
    from types import SimpleNamespace
    from scripts.texture_diagnostics import checkpoint_texture_flow_precision
    class Sampler:
        def sample(self):
            return torch.is_autocast_enabled('cpu')
    sampler = Sampler()
    pipeline = SimpleNamespace(device='cpu', tex_slat_sampler=sampler)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert sampler.sample()
        with checkpoint_texture_flow_precision(pipeline):
            assert not sampler.sample()
            assert torch.is_autocast_enabled('cpu')
        assert sampler.sample()
        with pytest.raises(ValueError):
            with checkpoint_texture_flow_precision(pipeline):
                raise ValueError('failure')
        assert sampler.sample()
    assert 'sample' not in vars(sampler)


def test_comparison_requires_matching_latent_and_coordinate_fingerprints():
    import copy
    from scripts.texture_diagnostics import compare_texture_reports
    report = {'texture_trace': {
        'flow_model': 'tex_slat_flow_model_1024',
        'raw_decoder_output': {'channels': [{}]*3 + [{'mean': 0.999, 'std': 0.0018}]},
        'decoder_input_sha256': 'latent', 'decoder_input_coords_sha256': 'coords'},
        'baked_metallic': {'metallic_high_ratio': 0.9813}}
    changed = copy.deepcopy(report)
    changed['texture_trace']['decoder_input_coords_sha256'] = 'different'
    summary = compare_texture_reports({'baseline_1024': report, 'decoder_fp32': copy.deepcopy(report), 'changed': changed})
    assert summary['decoder_fp32']['same_decoder_input_as_baseline_1024'] is True
    assert summary['changed']['same_decoder_input_as_baseline_1024'] is False
    assert summary['baseline_1024']['raw_decoder_metallic']['mean'] == 0.999


def test_compare_cli_rejects_invalid_modes(monkeypatch):
    from scripts.profile_1024_vram import parse_args
    monkeypatch.delenv('ZYREX_LOW_VRAM_DTYPE', raising=False)
    base = ['--image', 'fixture.png', '--output-dir', 'out', '--texture-compare-only']
    assert parse_args(base).texture_trace
    assert parse_args(base + ['--vram-profile', 'none', '--low-vram', 'off']).texture_compare_only
    for extra in (['--vram-profile', '8gb'], ['--vram-profile', 'none'],
                  ['--texture-decoder-dtype', 'fp32'], ['--texture-flow-autocast', 'checkpoint']):
        with pytest.raises(SystemExit):
            parse_args(base + extra)


def test_prediction_diff_schema_and_missing_data():
    import copy
    from scripts.compare_texture_predictions import compare
    t = dict(preprocessed_image={'sha256': 'image'},
             shape_slat={'coords_sha256': 'coords', 'feats_sha256': 'shape'},
             sampler_params={key: 1 for key in ('steps', 'guidance_strength', 'guidance_rescale', 'guidance_interval', 'rescale_t', 'sigma_min')})
    for resolution in (512, 1024):
        t[f'conditioning_{resolution}'] = {key: {'sha256': key} for key in ('cond', 'neg_cond')}
    for key in ('noise_sha256', 'noise_coords_sha256', 'concat_cond_sha256',
                'predicted_normalized_tex_slat_sha256', 'decoder_input_sha256',
                'decoder_input_coords_sha256', 'raw_decoder_output_sha256'):
        t[key] = key
    a = dict(status='ok', texture_trace=t, stage_metrics=[dict(stage='texture_decode', time_s=1, status='ok')])
    b = copy.deepcopy(a)
    assert compare(a, b)['complete'] and compare(a, b)['first_divergence'] is None
    b['texture_trace']['noise_sha256'] = 'changed'
    result = compare(a, b)
    assert result['first_divergence'] == 'texture_initial_noise'
    assert result['stage_wall_times']['upstream'][0]['time_s'] == 1
    assert json.loads(json.dumps(result)) == result
    del b['texture_trace']['preprocessed_image']
    result = compare(a, b)
    assert not result['complete'] and result['first_divergence'] is None
    assert result['first_observed_divergence'] == 'texture_initial_noise'
