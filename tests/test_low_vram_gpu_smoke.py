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


@pytest.mark.gpu_regression
def test_warlord_seed42_metallic_512_vs_1024(tmp_path):
    if os.environ.get('TRELLIS_RUN_WARLORD_REGRESSION') != '1':
        pytest.skip('Set TRELLIS_RUN_WARLORD_REGRESSION=1 with local warlord image and weights')
    from scripts.texture_diagnostics import assert_warlord_metallic
    image = Path(os.environ['TRELLIS_WARLORD_IMAGE']).resolve()
    model = Path(os.environ['TRELLIS_GPU_SMOKE_MODEL']).resolve()
    assert image.is_file()
    assert (model / 'pipeline.json').is_file()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ZYREX_LOW_VRAM_DTYPE='fp16')
    reports = {}
    # Same seed/settings in independent processes; different sparse layouts mean
    # random tensors need not match elementwise. Do not compare unmatched voxels.
    for profile in ('8gb', '8gb_1024'):
        output = tmp_path / profile
        subprocess.run([
            sys.executable, str(root / 'scripts/profile_1024_vram.py'),
            '--image', str(image), '--model-id', str(model), '--seed', '42',
            '--output-dir', str(output), '--vram-profile', profile, '--texture-trace',
            '--baker', os.environ.get('TRELLIS_GPU_SMOKE_BAKER', 'zyrex'),
        ], cwd=root, env=env, check=True)
        reports[profile] = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        assert reports[profile]['status'] == 'ok'
    (tmp_path / 'comparison.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    assert reports['8gb']['run_metrics']['pipeline_type'] == '512'
    assert reports['8gb_1024']['run_metrics']['output_resolution'] == 1024
    # Both runs complete and retain reports before the quality gate fails.
    for report in reports.values():
        assert_warlord_metallic(report)
