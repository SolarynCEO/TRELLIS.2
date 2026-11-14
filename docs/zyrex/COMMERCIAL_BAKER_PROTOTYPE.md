# Commercial Baker Prototype (ZYREX_BAKER=zyrex)

## Scope

Prototype-only replacement for NVIDIA-licensed UV baking in the **shipping GLB export path** (`o_voxel.postprocess.to_glb`) behind `ZYREX_BAKER`.

- Default remains upstream behavior: `ZYREX_BAKER=nvdiffrast`
- Prototype path: `ZYREX_BAKER=zyrex`

---

## Step 1 — NVIDIA call-site map

| File | Operation(s) | Shipping path? | Zyrex status |
|---|---|---:|---|
| `o-voxel/o_voxel/postprocess.py` | **UV-space rasterize**, barycentric **interpolate**, UV mask, position lookup for voxel sampling during GLB bake | **Yes** (app/example GLB export) | Implemented: torch/triton rasterizer + barycentric interpolation + `flex_gemm` `grid_sample_3d` + OpenCV Telea inpaint |
| `trellis2/pipelines/trellis2_texturing.py::postprocess_mesh` | UV-space rasterize + interpolate + voxel sampling for texturing pipeline output mesh | No (not in default app export path) | Not switched in this PR |
| `trellis2/renderers/mesh_renderer.py` | Camera-space rasterize/interpolate/texture/antialias | No (render preview/training/eval) | Out of scope in this PR |
| `trellis2/renderers/pbr_mesh_renderer.py` | Depth peeling, interpolation, texture sampling, env-map lookup | No (render preview/training/eval) | Out of scope in this PR |
| `trellis2/renderers/pbr_mesh_renderer.py::EnvMap` (`nvdiffrec_render.light`) | nvdiffrec envlight split-sum shading + mip build | No (render preview/training/eval) | Out of scope in this PR |
| `trellis2/utils/render_utils.py` | Indirect usage via renderers | No (preview/video snapshots) | Blocked on renderer migration |
| `trellis2/datasets/*`, `trellis2/trainers/vae/*` | Indirect usage via renderers for visualization/losses | Training/eval only | Blocked on renderer migration |
| `setup.sh`, `README.md`, `THIRD_PARTY_NOTICES.md` | Installation/license references for `nvdiffrast` / `nvdiffrec` | N/A | Commercial blocker documented upstream |

---

## Step 2 — Prototype implementation summary

Implemented modules:

- `o-voxel/o_voxel/zyrex_baker_backend.py`
  - Resolves `ZYREX_BAKER` values with default fallback to `nvdiffrast`
- `o-voxel/o_voxel/zyrex_uv_baker.py`
  - UV triangle rasterization (`ZYREX_RASTER_IMPL=torch|triton`, defaults to torch; Triton is explicit opt-in)
  - Barycentric interpolation (torch)
  - Standard fixed-point rasterization (8 subpixel bits) + top-left rule

Triton dependency note:
- The Triton raster path relies on the runtime Triton install available in the execution environment (e.g. laptop `triton-windows`); this prototype does not add new package dependencies.
- `o-voxel/o_voxel/postprocess.py`
  - Adds feature-flagged branch:
    - `nvdiffrast` path unchanged (default)
    - `zyrex` path uses the custom baker
    - export sampling uses `flex_gemm.ops.grid_sample.grid_sample_3d`
    - hole fill uses OpenCV Telea inpaint

Behavioral constraints:

- No production default replacement.
- No NVIDIA path removal.
- No GPL/AGPL dependency introduction.

Raster parity note:

- CUDA parity tests compare `zyrex` rasterization against `nvdiffrast` on a non-overlapping UV tiling.
- Face-id mismatches are only accepted for true edge-tie texels on shared edges:
  - the two winning faces must be adjacent (share an edge), and
  - the texel center must be on that shared edge within 8-bit fixed-point tolerance (`|edge_fn| <= 1` subpixel unit).
- Non-edge mismatches are treated as regressions.
- Raster implementation can be forced with `ZYREX_RASTER_IMPL=torch|triton`; default is `torch` (Triton remains explicit opt-in).

Runtime knobs:
- `ZYREX_RASTER_IMPL=torch|triton`
- `ZYREX_RASTER_PROFILE=1`
- `ZYREX_RASTER_PROFILE_SYNC=1`
- `ZYREX_EMIT_NORMAL_MAP=1` (experimental/unvalidated normal-map path)

---

## Step 3 — Comparison harness

Script: `trellis2/utils/zyrex_baker_compare.py`

- CPU synthetic validation against analytic expectations
- Optional real-asset UV re-bake sanity check from GLB textures
- GPU side-by-side modes for `nvdiffrast` vs `zyrex` with fixed seeds and JSON metrics output
- On non-CUDA hosts, all GPU metrics are emitted as **`pending GPU run`** (never estimated)

### Windows laptop command (RTX 4070 Laptop 8 GB, CUDA 12.4, torch 2.6)

Run from the repository root in PowerShell:

```powershell
$env:ZYREX_LOW_VRAM = "1"

py -3 trellis2\utils\zyrex_baker_compare.py `
  --output "outputs\benchmarks\zyrex_gpu_compare_2048.json" `
  --real-asset "C:\path\to\pipeline_or_real_asset.glb" `
  --run-gpu `
  --texture-size 2048 `
  --synthetic-texture-size 128 `
  --skip-rendered-views `
  --gpu-artifacts-dir "outputs\benchmarks\zyrex_gpu_compare_2048_artifacts"
```

Note: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is currently unsupported on Windows; do not set it for this command.

Expected outputs:

- JSON metrics including time, peak VRAM (allocated/reserved), seam coverage, and map error vs `nvdiffrast`
- Side-by-side PNGs (`nvdiffrast | zyrex | amplified abs diff`) for:
  - base color
  - metallic/roughness

### Produce real pipeline voxel-volume input (for production-path baker comparison)

From repo root in PowerShell:

```powershell
$env:ZYREX_LOW_VRAM = "1"

py -3 scripts\benchmark_low_vram.py `
  --image "C:\path\to\input.png" `
  --vram-profile 8gb `
  --pipeline-type 512 `
  --max-num-tokens 32768 `
  --texture-size 2048 `
  --output-dir "outputs\benchmarks" `
  --dump-baker-input
```

This writes an `.npz` baker-input dump path in the benchmark JSON under
`settings.baker_input_dump_path`.

### Compare nvdiffrast vs zyrex using production to_glb inputs

```powershell
$env:ZYREX_LOW_VRAM = "1"

py -3 trellis2\utils\zyrex_baker_compare.py `
  --output "outputs\benchmarks\zyrex_gpu_compare_voxel_2048.json" `
  --voxel-input "outputs\benchmarks\low_vram_baker_input_YYYYMMDDTHHMMSSZ.npz" `
  --run-gpu `
  --texture-size 2048 `
  --synthetic-texture-size 128 `
  --skip-rendered-views `
  --gpu-artifacts-dir "outputs\benchmarks\zyrex_gpu_compare_voxel_2048_artifacts"
```

The JSON includes:

- timing + peak VRAM allocated/reserved for both bakers
- seam coverage (nvd, zyrex, overlap, xor)
- map metrics vs nvdiffrast on **shared covered mask** and **full texture**
- per-channel MAE/RMSE
- side-by-side PNG outputs

### Synthetic rasterizer timing proxy (no GPU required)

If you only need a rough throughput check of the rasterizer path, run:

```powershell
py -3 trellis2\utils\zyrex_baker_compare.py `
  --output "outputs\benchmarks\zyrex_rasterizer_proxy.json" `
  --benchmark-rasterizer-faces 500000 `
  --benchmark-rasterizer-resolution 512 `
  --skip-rendered-views
```

This emits `rasterizer_proxy_benchmark` timing and coverage metrics in the output JSON.
