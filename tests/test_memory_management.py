"""CPU tensor tests and explicitly mocked CUDA accounting (not VRAM evidence)."""
import pytest
import torch
import torch.nn.functional as F

from trellis2.utils.memory import StageProfiler, chunk_ranges, chunked_norm_linear, resident_module


@pytest.mark.parametrize('length,size,expected', [
    (0, 3, []), (1, 3, [(0, 1)]), (6, 3, [(0, 3), (3, 6)]),
    (7, 3, [(0, 3), (3, 6), (6, 7)]),
])
def test_chunk_ranges(length, size, expected):
    assert chunk_ranges(length, size) == expected


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_invalid_chunk_size(size):
    with pytest.raises(ValueError):
        chunk_ranges(0, size)


@pytest.mark.parametrize('rows', [0, 1, 8, 19])
def test_chunked_projection_matches_full_cpu(rows):
    torch.manual_seed(17)
    features = torch.randn(rows, 16)
    layer = torch.nn.Linear(16, 7).eval()
    with torch.no_grad():
        expected = layer(F.layer_norm(features, (16,)))
        actual = chunked_norm_linear(features, layer, 4)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == features.dtype


def test_residency_offloads_after_exception():
    calls = []
    class Model:
        def to(self, device):
            calls.append(('to', device))
        def cpu(self):
            calls.append(('cpu',))
    with pytest.raises(RuntimeError, match='forward failed'):
        with resident_module(Model(), 'cpu'):
            calls.append(('forward',))
            raise RuntimeError('forward failed')
    assert calls == [('to', 'cpu'), ('forward',), ('cpu',)]


def test_cpu_profiler_never_calls_cuda_and_records_failure(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU profiling called CUDA')
    for name in ('synchronize', 'empty_cache', 'reset_peak_memory_stats',
                 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    profiler = StageProfiler('cpu', empty_cache=True)
    with pytest.raises(ValueError):
        with profiler.stage('decode'):
            raise ValueError('test failure')
    assert profiler.metrics[0]['status'] == 'failed'
    assert profiler.metrics[0]['peak_vram_reserved_bytes'] is None
    assert profiler.metrics[0]['time_s'] >= 0
    assert profiler.peaks()['peak_vram_allocated_bytes'] is None


def test_mock_cuda_nested_peaks_survive_resets(monkeypatch):
    # These tiny counters test accounting only; they are not GPU measurements.
    counters = {'allocated': 0, 'reserved': 0}
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda d: calls.append('sync'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('free'))
    def reset(device):
        calls.append('reset')
        counters.update(allocated=0, reserved=0)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda d: counters['allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda d: counters['reserved'])
    profiler = StageProfiler('cuda', cuda_active=True, empty_cache=True)
    with profiler.stage('parent'):
        counters.update(allocated=100, reserved=120)
        with profiler.stage('child'):
            counters.update(allocated=20, reserved=30)
        counters.update(allocated=40, reserved=50)
    parent, child = profiler.metrics
    assert parent['peak_vram_reserved_bytes'] == 120
    assert child['peak_vram_reserved_bytes'] == 30
    assert calls[:2] == ['sync', 'reset']
    assert calls.count('free') == 2


def test_profile_script_cpu_failure_writes_null_metrics(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    import sys
    import json
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'profile_1024_vram.py'
    spec = importlib.util.spec_from_file_location('profile_1024_vram', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [str(script), '--image', 'unused.png', '--output-dir', str(tmp_path)])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Isolate environment changes made by the executable's main().
    for key in ('PYTORCH_CUDA_ALLOC_CONF', 'ATTN_BACKEND', 'SPARSE_ATTN_BACKEND', 'ZYREX_BAKER'):
        monkeypatch.setenv(key, 'test')
    with pytest.raises(RuntimeError, match='no measurements were taken'):
        module.main()
    report = json.loads((tmp_path / 'metrics.json').read_text())
    assert report['status'] == 'failed'
    assert report['target_met'] is None
    assert report['stage_metrics'] == []
    assert report['peak_vram_reserved_bytes'] is None
