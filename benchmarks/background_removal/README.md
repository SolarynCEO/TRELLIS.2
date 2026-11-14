# Background-removal benchmark (BiRefNet migration)

This folder contains a repeatable benchmark harness for TRELLIS.2 preprocessing.

## Files

- `image_manifest.json`: pinned external test-image URLs with source and license metadata.
- `download_test_images.py`: downloads and downscales those images to `images/`.
- `benchmark_birefnet.py`: runs background removal + TRELLIS framing stats and writes artifacts. It benchmarks standard/dynamic/HR BiRefNet variants by default.

## Usage

```bash
python3 benchmarks/background_removal/download_test_images.py
PYTHONPATH=. python3 benchmarks/background_removal/benchmark_birefnet.py \
  --warlord-zip /home/ubuntu/.cursor/projects/agent-repos-TRELLIS-2/uploads/warlord_reference_inputs_6dbd.zip \
  --artifacts-dir /opt/cursor/artifacts/birefnet-benchmark
```

Optional: benchmark only a subset of variants:

```bash
PYTHONPATH=. python3 benchmarks/background_removal/benchmark_birefnet.py \
  --variant-model ZhengPeng7/BiRefNet \
  --variant-model ZhengPeng7/BiRefNet_dynamic \
  --warlord-zip /home/ubuntu/.cursor/projects/agent-repos-TRELLIS-2/uploads/warlord_reference_inputs_6dbd.zip
```
