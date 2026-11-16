"""Explicit laptop-only smoke test; never downloads weights or runs by default."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.gpu_smoke
def test_8gb_1024_end_to_end(tmp_path):
    if os.environ.get('TRELLIS_RUN_GPU_SMOKE') != '1':
        pytest.skip('Set TRELLIS_RUN_GPU_SMOKE=1 with local model and image paths')
    image = Path(os.environ['TRELLIS_GPU_SMOKE_IMAGE'])
    model = Path(os.environ['TRELLIS_GPU_SMOKE_MODEL'])
    assert image.is_file()
    assert (model / 'pipeline.json').is_file(), 'Use an existing local weights directory'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # Separate process prevents optional-native CPU test stubs contaminating this run.
    subprocess.run([
        sys.executable, str(root / 'scripts/profile_1024_vram.py'),
        '--image', str(image.resolve()), '--model-id', str(model.resolve()),
        '--output-dir', str(tmp_path), '--vram-profile', '8gb_1024',
        '--baker', os.environ.get('TRELLIS_GPU_SMOKE_BAKER', 'zyrex'),
    ], cwd=root, env=env, check=True)
    report = json.loads((tmp_path / 'metrics.json').read_text())
    assert report['status'] == 'ok'
    assert report['run_metrics']['output_resolution'] == 1024
    assert report['peak_vram_reserved_bytes'] > 0
    assert (tmp_path / 'asset.glb').stat().st_size > 0
    stages = {stage['stage']: stage['status'] for stage in report['stage_metrics']}
    for stage in ('cascade_upsample', 'texture_decode', 'bake', 'export'):
        assert stages[stage] == 'ok'
    # Completion is distinct from the allocator budget and physical-device usage.
    assert report['target_scope'] == 'pytorch_allocator_reserved_only'
