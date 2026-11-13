# Tests, Baseline Validation, and Known Limitations

## 1) Official test coverage in this repo

### What exists

- GitHub workflow: `/workspace/.github/workflows/codeql.yml`
  - Runs CodeQL analysis for `python` and `c-cpp`

### What does **not** exist

- No functional/unit/integration test suite was found for inference/training behavior in this checkout.
- `pytest` discovery result in this repo: **0 tests collected**.

## 2) Commands actually run in this VM

> This VM does not expose NVIDIA GPU tooling; results below are real command outcomes.

### Environment / hardware probes

```bash
uname -a
nvidia-smi
python3 --version
python3 -m pip --version
lscpu
grep -E 'MemTotal|MemAvailable' /proc/meminfo
```

Observed highlights:

- `nvidia-smi`: command not found
- Python: `3.12.3`
- CPU: 4 vCPU Intel Xeon (KVM)
- RAM: ~16 GB total

### Baseline import checks

Before dependency install:

- `import trellis2` -> OK
- `import trellis2.pipelines.trellis2_image_to_3d` -> failed (`ModuleNotFoundError: No module named 'torch'`)

After installing CPU-safe Python packages (`torch`, `torchvision`, `transformers`, etc.):

- `import torch` -> OK
- `import trellis2` -> OK
- `import trellis2.pipelines.trellis2_image_to_3d` -> failed (`ModuleNotFoundError: No module named 'cumesh'`)
- `import trellis2.pipelines.trellis2_texturing` -> failed (`ModuleNotFoundError: No module named 'o_voxel'`)

### Native extension install attempts (expected blocker on this VM)

Commands run:

```bash
python3 -m pip install --user /workspace/o-voxel --no-build-isolation
python3 -m pip install --user git+https://github.com/JeffreyXiang/CuMesh.git --no-build-isolation
```

Both failed during build metadata with:

- `OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

### Test/lint-style checks run

```bash
python3 -m pytest
python3 -m compileall trellis2 data_toolkit app.py app_texturing.py example.py example_texturing.py train.py
```

Results:

- `pytest`: no tests ran (`collected 0 items`)
- `compileall`: completed successfully; two `SyntaxWarning` entries in docstring escape sequences (`trellis2/models/sparse_structure_vae.py`)

## 3) Baseline run requirements (what a real inference run needs)

Minimum practical path for official Python TRELLIS.2 stack in this repo:

1. Linux host with NVIDIA GPU and CUDA toolchain
2. Install dependencies and extensions (including `o-voxel`, `cumesh`, `flexgemm`, optional attention backends)
3. Load HF checkpoints via:
   - `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")`
4. Run:
   - image preprocess -> `pipeline.run(...)`
   - optional GLB postprocess/export via `o_voxel.postprocess.to_glb(...)`

Because this VM lacks GPU/CUDA toolchain, **no local inference mesh/GLB output was produced** in this phase.

## 4) Known limitations observed or documented

1. Upstream support scope is Linux + high-memory NVIDIA GPUs (README states >=24 GB VRAM).
2. Critical native modules (`o_voxel`, `cumesh`, `flex_gemm`) require CUDA build environment.
3. Two pipeline-side model dependencies are gated with `license: other` metadata:
   - `facebook/dinov3-vitl16-pretrain-lvd1689m`
   - `briaai/RMBG-2.0`
4. `nvdiffrast` / `nvdiffrec` license texts include non-commercial restrictions (see `THIRD_PARTY_NOTICES.md`).
5. No official functional test harness in repo to validate output fidelity regressions automatically.

## 5) Proposed fixed benchmark set (~10 images) with seeds

This is a recommended repeatable set for Zyrex fork regression tracking (store exact input images in a separate benchmark asset repo/bucket, not in git history here).

| ID | Category | Suggested asset type | Seed |
|---|---|---|---:|
| B01 | Hard-surface prop | Industrial crate/toolbox | 42 |
| B02 | Hard-surface prop | Vehicle part / drone module | 1337 |
| B03 | Weapon | Sword / blade silhouette | 42 |
| B04 | Weapon | Rifle / sci-fi firearm | 31415 |
| B05 | Armor | Helmet bust | 42 |
| B06 | Armor | Chestplate / pauldrons | 27182 |
| B07 | Character | Stylized humanoid full body | 42 |
| B08 | Character | Realistic bust portrait | 777 |
| B09 | Creature | Quadruped creature | 9001 |
| B10 | Architecture | Building facade / tower | 123456 |

Recommended fixed generation knobs for baseline comparability:

- Resolution: `1024` (`1024_cascade`)
- Face target for export: `~200k-300k` depending on use case
- Texture size: `2048`
- Keep all three stage samplers at default steps/guidance unless testing ablations

## 6) External baseline evidence provided by owner (not a local VM run)

Owner-supplied context (to retain in project docs as external reference):

- A TRELLIS.2 output was reportedly generated via Space mirror  
  `prithivMLmods/TRELLIS.2-Text-to-3D`
- Stated run settings: resolution 1024, seed 42, ~200k face target, 2048 textures
- Stated output metrics: 194,529 triangles, 139,386 vertices, one mesh, normals+UVs, 2048 base-color + 2048 metallic-roughness textures, mostly metallic=1.0, ~1.0 unit tall, not watertight

Attached ZIP at  
`/home/ubuntu/.cursor/projects/workspace/uploads/warlord_reference_inputs_e9a0.zip` contains:

- `reference.jpg`, `cutout_square_rgba.png`, `trellis2_preprocessed.png`, `run_trellis2.py`, `trellis2.log`, `render.py`

Note: the included `trellis2.log` in that ZIP shows an official Space attempt failing with ZeroGPU quota exhaustion, so this phase treats the detailed output metrics above as owner-provided evidence rather than locally reproduced output.
