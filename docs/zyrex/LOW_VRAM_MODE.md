# TRELLIS.2 Low-VRAM Mode (`8gb`) for Zyrex Fork

This fork adds an opt-in low-VRAM profile intended to make image-to-3D generation practical on GPUs with 8 GB VRAM (for example, RTX 4070 Laptop GPUs), while keeping default behavior unchanged unless explicitly enabled.

## What this mode does

Enable with either:

- environment variable: `ZYREX_LOW_VRAM=1`
- runtime argument: `pipeline.run(..., vram_profile="8gb")`

When `vram_profile="8gb"` is active, the pipeline:

1. **Forces sequential stage offload** (`low_vram=True`) so each stage's model is moved to GPU only while it runs, then moved back to CPU.
2. **Adds per-stage memory/time instrumentation** (`torch.cuda.max_memory_allocated` and `max_memory_reserved`) for:
   - `preprocess_image`
   - `condition_512`
   - `condition_1024` (non-512 pipelines)
   - `sparse_structure`
   - shape stage (pipeline-dependent)
   - texture stage
   - `decode_latent`
3. **Uses a conservative default generation profile** for memory:
   - default pipeline type: `512`
   - default token cap: `32768`
4. **Prefers SDPA attention backend** when no backend is explicitly configured (helps Windows compatibility when flash-attn is unavailable).
5. **Supports optional autocast precision in low-VRAM mode**:
   - default: `fp16`
   - override via `ZYREX_LOW_VRAM_DTYPE=bf16|fp16|none`

## Default behavior compatibility

- If you do **not** set `ZYREX_LOW_VRAM` and do **not** pass `vram_profile`, behavior remains unchanged from this fork's prior default path.
- Existing calls to `pipeline.run(...)` still work.

## Quality / speed tradeoffs

Expected tradeoffs in `8gb` profile:

- **Lower peak VRAM** (primary goal).
- **Longer wall-clock time** due to CPU<->GPU model transfers between stages.
- **Potential quality loss** if you keep `pipeline_type=512` (fewer details than 1024/1536 paths).
- **Potential quality/speed variation** with `fp16`/`bf16` autocast versus full precision.
- **SDPA backend** is generally slower than flash-attn but much easier to run on Windows.

## Stage weight-memory estimate (reasoned, not measured in this CPU VM)

TRELLIS.2 advertises ~4B total parameters. The major flow checkpoints are named as ~1.3B class models (`...dit_1_3B...`), so a practical approximation is:

- Sparse-structure flow: ~1.3B params
- Shape flow: ~1.3B params
- Texture flow: ~1.3B params
- Decoders + DINOv3 + BiRefNet: smaller remainder (order-of-magnitude lower than one large flow stage)

Approximate weight footprint by precision:

- **fp32**: ~16 GB for 4B params
- **fp16/bf16**: ~8 GB for 4B params
- **int8**: ~4 GB for 4B params

Because this mode offloads per stage, **peak VRAM is dominated by the active stage weights + activations**, not the full model set loaded simultaneously.

## Benchmark script (Windows-friendly)

Script:

- `scripts/benchmark_low_vram.py`

What it outputs:

- one `.glb` file
- one `.json` report containing:
  - per-stage peak VRAM/time
  - total time
  - settings used
  - output GLB path

### PowerShell example (recommended for laptop validation)

```powershell
$env:ZYREX_LOW_VRAM="1"
$env:ATTN_BACKEND="sdpa"
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
py -3 scripts/benchmark_low_vram.py `
  --image "assets/example_image/T.png" `
  --vram-profile 8gb `
  --pipeline-type 512 `
  --max-num-tokens 32768 `
  --output-dir "outputs/benchmarks"
```

To benchmark your warlord image, replace `--image` with your absolute or relative path.

### Command Prompt example

```bat
set ZYREX_LOW_VRAM=1
set ATTN_BACKEND=sdpa
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py -3 scripts\benchmark_low_vram.py --image "assets\example_image\T.png" --vram-profile 8gb --pipeline-type 512 --max-num-tokens 32768 --output-dir "outputs\benchmarks"
```

## Important measurement note

This repository's cloud CPU VM cannot produce real CUDA VRAM numbers. Treat all VRAM fit claims as **pending** until you run the benchmark on your target GPU and inspect the emitted JSON.
