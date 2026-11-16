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

The goal is at most **6.5 GiB PyTorch peak reserved**. In a user-reported
RTX 4070 Laptop 8 GB run, disabling autocast for cascade upsampling allowed
completion in **256.5 s**, with **5.27 GiB allocated (peak at `bake`) / 5.93 GiB reserved
(peak at `texture_decode`)**. These are laptop measurements supplied by the user, not CPU
test results or a GPU run performed by the agent.

**nvidia-smi peaked at 7,886 MiB** (about 7.70 GiB) while PyTorch reserved peaked
at 5.93 GiB. `target_met` tests only PyTorch allocator reserved memory and completed
1024 generation; it does **not** certify total-device headroom or material quality.
The report explicitly labels `target_scope=pytorch_allocator_reserved_only`.
The differing scopes and sampling times cannot establish an exact allocation
breakdown. This single run does not guarantee other images fit.

The user confirmed that metallic saturation starts in decoded voxels, before
bake: warlord seed 42, channel 3 min 0.985, mean 0.9996, fraction >= 0.99
0.99999; roughness varied with mean 0.82 and base color looked normal. The baked
blue-channel ratio >= 0.95 was 0.9805. **The material issue remains unresolved.**
It is not a GLB packing issue or a reason to swap channels or zero metallic.

Source audit: both profiles use the same normalization functions. Shape SLat
conditioning is normalized once; texture predictions are denormalized once;
the decoder uses `raw * 0.5 + 0.5`, consistent with the PBR training render path.
CPU tests cover nontrivial mean/scale values and both texture flow selections.
Actual 512/1024 GPU latent predictions have not been compared by the agent.

`--texture-trace` now records actual texture conditioning, normalized shape
conditioning, normalized sampler predictions, denormalized decoder inputs,
raw/scaled decoder channel moments, model identifiers, normalization vectors,
decoded voxel statistics and the exported GLB blue-channel ratio. Statistics are
copied to CPU in bounded chunks; model outputs are never edited. Trace runs add
CPU transfers and are **not timing benchmarks**. Use `--vram-profile 8gb` for the
512 reference and `--vram-profile 8gb_1024` for the cascade with the same seed.
Matching seeds do not align individual noise/latent elements across different
token counts and coordinate layouts.

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
  controls it. Cascade upsampling runs without flow autocast to avoid a flex_gemm
  fp32/fp16 operand mismatch. Numerical/visual parity still needs real validation.
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


### Opt-in real GPU smoke test

The `gpu_smoke` pytest marker runs generation, baking and GLB export in an isolated
process. It skips unless explicitly enabled and requires existing local weights;
Hugging Face offline mode prevents downloads. It checks completion, not a universal
VRAM/quality guarantee. It was **not run by the agent**.

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:TRELLIS_RUN_GPU_SMOKE = '1'
$env:TRELLIS_GPU_SMOKE_IMAGE = Read-Host 'Absolute path to local warlord image'
$env:TRELLIS_GPU_SMOKE_MODEL = Read-Host 'Local TRELLIS.2 weights directory containing pipeline.json'
$env:TRELLIS_GPU_SMOKE_BAKER = 'zyrex'
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe -m pytest tests\test_low_vram_gpu_smoke.py -m gpu_smoke -v
Remove-Item Env:TRELLIS_RUN_GPU_SMOKE -ErrorAction SilentlyContinue
```


### Warlord metallic regression (GPU, opt-in)

This test runs **both** 512 and 1024 with seed 42 in independent offline processes,
saves both GLBs/reports and `comparison.json` under the pytest temp directory, then
fails if either `baked_metallic.metallic_high_ratio` exceeds **0.5**. The ratio
uses blue >= 0.95 over encoded metallic-roughness texture pixels (including UV
padding), with the maximum ratio across materials; it is not surface-area-weighted.
Material factors cannot hide saturation. The user-reported 0.9805 fails this gate.
No GPU regression was run by the agent. It is expected to expose the unresolved
bug until a supported fix is validated. No fixture or weights are committed.

```powershell
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
$env:TRELLIS_RUN_WARLORD_REGRESSION = '1'
$env:TRELLIS_WARLORD_IMAGE = Read-Host 'Absolute path to local warlord fixture'
$env:TRELLIS_GPU_SMOKE_MODEL = Read-Host 'Local weights directory containing pipeline.json'
$env:TRELLIS_GPU_SMOKE_BAKER = 'zyrex'
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe -m pytest tests\test_low_vram_gpu_smoke.py -m gpu_regression -v
Remove-Item Env:TRELLIS_RUN_WARLORD_REGRESSION -ErrorAction SilentlyContinue
```

### Isolating texture flow precision from decoder precision

The latest user run reports baked metallic_high_ratio **0.9813** and raw decoder
channel 3 min **0.969**, mean **0.999**, std **0.0018**; the 32-channel texture
latent is not collapsed and roughness varies. This is before output scaling.
The saturation is still unresolved; the 0.5 warlord gate is unchanged.

Inspection of cached Microsoft model config revision
`af44b45f2e35a493886929c6d786e563ec68364d` shows separate 512/1024 texture flow
checkpoints configured for **bf16**, but one shared `tex_dec_next_dc_f16c32_fp16`
decoder (`SparseUnetVaeDecoder`, 32 input channels, 6 output channels, fp16 blocks).
The decoder input/output projections remain fp32. This cached config is evidence
about the model design, not proof of the exact config used by a particular run.
The trace records the actual model paths and live dtypes.

Flow `manual_cast` uses the configured bf16 dtype only when autocast is disabled;
the profile normally requests fp16 autocast. This is a precision hypothesis, not
evidence that fp16 causes the saturation. A concrete conversion bug was found:
decoder `convert_to_fp32()` changed block weights but did not update `self.dtype`,
so forward still cast features to fp16. Conversion methods now keep activation
dtype and `use_fp16` synchronized with the block weights. Constructor defaults
and the normal generation path are unchanged.

Two benchmark-only switches isolate these questions:

- `--texture-decoder-dtype fp32`: changes only the loaded texture decoder's block
  arithmetic. It does not recover precision already lost when loading into fp16
  parameters or change checkpoint files. **May exceed 8 GB**; it is not a new
  production memory profile or a validated fix.
- `--texture-flow-autocast checkpoint`: disables extra autocast only inside the
  texture sampler so the existing flow uses its configured dtype (bf16 in the
  inspected configs). Shape/conditioning stages and model parameters are unchanged.

Run separate processes with the same input/seed/baker. These diagnostic switches
are off by default. Trace overhead makes these quality comparisons unsuitable as
performance measurements. Example from the engine repository root:

```powershell
$image = Read-Host 'Absolute path to local warlord fixture'
$model = Read-Host 'Local weights directory containing pipeline.json'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:ZYREX_LOW_VRAM_DTYPE = 'fp16'
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py --image $image --model-id $model --seed 42 --baker zyrex --texture-trace --vram-profile 8gb --output-dir outputs\texture-ablation\baseline_512
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py --image $image --model-id $model --seed 42 --baker zyrex --texture-trace --vram-profile 8gb_1024 --output-dir outputs\texture-ablation\baseline_1024
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py --image $image --model-id $model --seed 42 --baker zyrex --texture-trace --vram-profile 8gb_1024 --texture-decoder-dtype fp32 --output-dir outputs\texture-ablation\decoder_fp32
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\profile_1024_vram.py --image $image --model-id $model --seed 42 --baker zyrex --texture-trace --vram-profile 8gb_1024 --texture-flow-autocast checkpoint --output-dir outputs\texture-ablation\checkpoint_flow
& C:\dev\TRELLIS.2\.venv\Scripts\python.exe scripts\texture_diagnostics.py `
  --report baseline_512=outputs\texture-ablation\baseline_512\metrics.json `
  --report baseline_1024=outputs\texture-ablation\baseline_1024\metrics.json `
  --report decoder_fp32=outputs\texture-ablation\decoder_fp32\metrics.json `
  --report checkpoint_flow=outputs\texture-ablation\checkpoint_flow\metrics.json
```

The comparison prints raw metallic min/mean/std, baked ratios, flow/decoder
precision and whether decoder input bytes **and coordinates** match the 1024
baseline. Only attribute a decoder-only difference to precision if those inputs
match. A 512 input is not expected to match 1024. OOM/failed runs are not passing
quality results. No GPU ablations were run by the agent; no metallic correction
has been validated. The existing `gpu_regression` test remains the acceptance gate.
