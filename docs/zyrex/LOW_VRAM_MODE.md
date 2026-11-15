# TRELLIS.2 Low-VRAM Profiles for Zyrex Fork

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
6. **Sets SDPA attention backend by default in low-VRAM mode**:
   - when `ZYREX_LOW_VRAM=1`, this fork defaults `ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=sdpa` unless already set
   - if flash-attn/xformers are unavailable, runtime attention calls automatically fall back to SDPA

## Default behavior compatibility

- If you do **not** set `ZYREX_LOW_VRAM` or `ZYREX_VRAM_PROFILE` and do **not** pass `vram_profile`, behavior remains unchanged from this fork's prior default path.
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

## One measured target-hardware run (user-provided)

Hardware:

- RTX 4070 Laptop GPU (8 GB VRAM), Windows, CUDA 12.4
- torch 2.6.0+cu124

Run settings:

- seed: `42`
- input: warlord image
- profile: `8gb`
- pipeline: `512`
- token cap: `32768`
- precision: `fp16`
- attention backend: `sdpa`

Observed metrics from that run:

- total wall time: **87.9 s**
- generation time: **72.0 s**
- peak allocated VRAM: **3.35 GB**
- peak reserved VRAM: **4.59 GB**
- output GLB: **498k faces**, **2048 texture**, visual result reported as correct

This is a single run datapoint; expect variance by image complexity, seed, and backend/runtime state.

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

## Dependency notes for Windows/BiRefNet

- Use `transformers>=4.56,<5` (validated with `4.57.6`)
- BiRefNet remote code requires:
  - `einops`
  - `kornia`
  - `timm`

These are included in `requirements.txt` and `setup.sh --basic`.

## Important measurement note

This repository's cloud CPU VM cannot produce real CUDA VRAM numbers. Treat all VRAM fit claims as **pending** until you run the benchmark on your target GPU and inspect the emitted JSON.

## Experimental explicit 1024 cascade profile

`pipeline.run(image, vram_profile="8gb_1024", return_metrics=True)` selects
**1024_cascade without a 512 fallback**. Alternatively set
`ZYREX_VRAM_PROFILE=8gb_1024` before creating/running the pipeline. An explicit
`vram_profile="none"` disables environment profile selection. The existing
`ZYREX_LOW_VRAM=1` / `8gb` behavior still defaults to 512.

For app integration, set legacy offloading **before** moving the pipeline's
execution device, so an all-model upload cannot OOM before `run()` starts:

```python
pipeline.low_vram = True
pipeline.cuda()
meshes, metrics = pipeline.run(image, vram_profile="8gb_1024", return_metrics=True)
assert metrics["output_resolution"] == 1024
```

The goal is at most **6.5 GiB peak reserved**, leaving headroom on an 8 GB GPU.
**This target is unverified. No GPU generation, bake, quality comparison, or
per-stage VRAM measurements were run for this change.** The profile is a candidate
for laptop validation, not a guarantee that all images fit. A token cap or OOM
failure is not a successful 1024 result.

The opt-in profile:

- Evicts idle models to CPU at entry and keeps sequential stage offloading.
  It offloads models even if a caller previously placed them all on CUDA.
- Disables Euler sampler prediction history. Ordinary sampler calls retain history.
- Releases the low-resolution shape and upsampling intermediates before the
  high-resolution flow, moves inactive high-resolution conditioning to CPU, and
  drops conditioning and unrequested latents before later decodes.
- Streams sparse decoder modules (input projection, each block, output projection)
  to the execution device and back to CPU, including failed forwards. This can
  substantially increase CPU transfer time and requires sufficient host RAM.
- Chunks only the independent rows of the final layer-norm/linear projection.
  `decoder_chunk_size=32768` is the default; override with the runtime argument or
  `ZYREX_DECODER_CHUNK_SIZE`. Sparse convolutions, their caches, and mesh extraction
  remain whole to preserve neighborhood/topology correctness.
- Keeps decoder precision as configured in the checkpoint. Flow/conditioning
  autocast defaults to fp16, as in `8gb`; `ZYREX_LOW_VRAM_DTYPE=none|fp16|bf16`
  controls it. Numerical/visual parity still needs real validation.
- Rejects multiple samples, conflicting pipeline types, invalid chunk sizes, and
  a high-resolution token count above `max_num_tokens` (default 49152). It never
  silently lowers resolution or discards tokens. Raising this cap increases risk.
- Releases allocator caches between stages and streamed blocks. Configure
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **before importing torch**.
  Some Windows/PyTorch builds may warn that this is unsupported; the report records
  the requested configuration, not proof it took effect.

The original pipeline settings and temporary decoder attributes are restored on
exit. Optimized runs leave models on CPU for baking; a subsequent run with profile
disabled and `low_vram=False` restores full model residency automatically. For
standalone calls to pipeline submethods after an optimized run, call `.to(device)`
first if using `low_vram=False`. `return_latent=True` retains the requested latents
and can increase peak memory; the benchmark uses the default `False`.

### Stage profiling and failure reports

`profile_stages=True` enables detailed profiling for `1024_cascade` with either
`vram_profile="none"` (original cascade) or `"8gb_1024"`. Records include:
conditioning at 512/1024, sparse structure, shape SLat at 512, cascade upsampling,
shape SLat at 1024, texture SLat, shape decode, mesh extraction, texture decode,
and mesh finalization. The optimized path also records coordinate quantization.
Parent stage peaks include their children; **do not sum peaks or nested timings**.

`scripts/profile_1024_vram.py` additionally measures model loading, mesh
simplification, **bake**, and export. It writes `metrics.json` on success or failure
after torch has imported, including completed stages and the failing stage when
available. CUDA timings are synchronized. CPU counters are null. The report's
`target_met` requires a completed 1024 output and an overall peak reserved at most
6.5 GiB, including bake. `pipeline.last_run_stats` alone covers generation only.
Metrics measure **PyTorch's allocator**, not direct native CUDA allocations or
other processes; also observe total GPU usage on the laptop.

### Exact laptop validation commands (PowerShell)

Run baseline first, then optimized, in separate Python processes. Choose the same
local warlord image when prompted. Neither the image nor generated outputs belong
in a commit. These commands use existing TRELLIS.2 weights; no weights are modified.

```powershell
Set-Location C:\dev\TRELLIS.2
git fetch origin feat/low-vram-1024
git switch --create validate/low-vram-1024 --track origin/feat/low-vram-1024
$warlordImage = Read-Host 'Absolute path to the local warlord image'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
Remove-Item Env:ZYREX_LOW_VRAM_DTYPE -ErrorAction SilentlyContinue
Remove-Item Env:PYTORCH_ALLOC_CONF -ErrorAction SilentlyContinue
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:ZYREX_RASTER_IMPL = 'torch'

# Baseline with legacy low_vram OFF. It may OOM; keep metrics.json if so.
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py `
  --image $warlordImage --vram-profile none --low-vram off `
  --baker zyrex --seed 42 --texture-size 2048 --decimation-target 500000 `
  --output-dir outputs\vram1024-baseline

# Explicit 1024 profile; no implicit 512 downgrade.
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py `
  --image $warlordImage --vram-profile 8gb_1024 `
  --decoder-chunk-size 32768 --max-num-tokens 49152 `
  --baker zyrex --seed 42 --texture-size 2048 --decimation-target 500000 `
  --output-dir outputs\vram1024-optimized

$report = Get-Content outputs\vram1024-optimized\metrics.json -Raw | ConvertFrom-Json
$report | Select-Object status, target_met, peak_vram_allocated_bytes, peak_vram_reserved_bytes
$report.stage_metrics | Select-Object stage, status, time_s, `
  @{Name='AllocatedGiB';Expression={ if ($null -ne $_.peak_vram_allocated_bytes) { $_.peak_vram_allocated_bytes / 1GB } }}, `
  @{Name='ReservedGiB';Expression={ if ($null -ne $_.peak_vram_reserved_bytes) { $_.peak_vram_reserved_bytes / 1GB } }}
```

For the default baker, repeat **both** commands with `--baker nvdiffrast` and new
output directories. Do not compare different bakers as a profile-only comparison.
Repeat multiple times, inspect GLBs visually, check resolution and all failed
stages, and attach reports before declaring the 8 GB/6.5 GiB goal achieved.
The script defaults to the existing nvdiffrast baker; no baker source is changed.

CPU-only regression commands (no model downloads/generation):

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe -m pytest `
  tests\test_memory_management.py tests\test_low_vram_mode.py tests\test_sparse_attention_sdpa.py -q
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
```

The CPU suite uses real small tensors for sampler and projection equivalence, and
mocks optional native dependencies, generation/extraction, and CUDA counters for
control-flow/accounting tests. Mock counters are not measured VRAM. No dependencies
or major dependency versions are added or changed by this profile.
