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
