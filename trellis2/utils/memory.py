"""Opt-in inference memory utilities; no CUDA calls when profiling a CPU run."""
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F


def chunk_ranges(length, chunk_size):
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError('decoder_chunk_size must be a positive integer')
    if length < 0:
        raise ValueError('length must be nonnegative')
    return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]


def chunked_norm_linear(features, layer, chunk_size):
    """Chunk only independent rows, never sparse convolutions or mesh topology.

    Inference only. The output projection retains the input precision; no extra
    decoder autocast is introduced. SparseLinear is an nn.Linear subclass.
    """
    ranges = chunk_ranges(features.shape[0], chunk_size)
    output = features.new_empty((features.shape[0], layer.out_features))
    for start, end in ranges:
        normalized = F.layer_norm(features[start:end], features.shape[-1:])
        output[start:end] = F.linear(normalized, layer.weight, layer.bias)
    return output


@contextmanager
def resident_module(module, device):
    """Stream one module and always offload it, including on a failed forward."""
    try:
        module.to(device)
        yield module
    finally:
        module.cpu()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()


class StageProfiler:
    """Synchronized allocator peaks, including failed and nested stages.

    Each reset first saves the current segment into every active parent, so a
    child reset cannot erase its parent's earlier high-water mark. Measurements
    describe PyTorch's allocator, not all native CUDA allocations.
    """
    def __init__(self, device, *, cuda_active=False, empty_cache=False):
        self.device = device
        self.cuda_active = cuda_active
        self.empty_cache = empty_cache
        self.metrics = []
        self._active = []
        self._peaks = dict(peak_vram_allocated_bytes=None, peak_vram_reserved_bytes=None)

    def _capture(self):
        if self.cuda_active:
            allocated = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
            self._peaks['peak_vram_allocated_bytes'] = max(self._peaks['peak_vram_allocated_bytes'] or 0, allocated)
            self._peaks['peak_vram_reserved_bytes'] = max(self._peaks['peak_vram_reserved_bytes'] or 0, reserved)
            for metric in self._active:
                metric['peak_vram_allocated_bytes'] = max(metric['peak_vram_allocated_bytes'], allocated)
                metric['peak_vram_reserved_bytes'] = max(metric['peak_vram_reserved_bytes'], reserved)

    @contextmanager
    def stage(self, name):
        if self.cuda_active:
            torch.cuda.synchronize(self.device)
            self._capture()
            torch.cuda.reset_peak_memory_stats(self.device)
        metric = dict(stage=name, status='failed', time_s=None,
                      peak_vram_allocated_bytes=0 if self.cuda_active else None,
                      peak_vram_reserved_bytes=0 if self.cuda_active else None)
        self.metrics.append(metric)
        self._active.append(metric)
        started = time.perf_counter()
        try:
            yield
            if self.cuda_active:
                torch.cuda.synchronize(self.device)
            metric['status'] = 'ok'
        finally:
            self._capture()
            metric['time_s'] = time.perf_counter() - started
            self._active.pop()
            if self.cuda_active:
                torch.cuda.reset_peak_memory_stats(self.device)
                if self.empty_cache:
                    torch.cuda.empty_cache()

    def peaks(self):
        self._capture()  # Includes transfers/cleanup between named stages.
        return dict(self._peaks)
