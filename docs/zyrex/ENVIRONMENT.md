# Environment Requirements and Platform Reality Check

This document captures reproducible requirements from the checked-out TRELLIS.2 codebase and a practical assessment for target hardware.

## 1) Upstream-declared requirements (from this repo)

From `/workspace/README.md` and `/workspace/setup.sh`:

- **OS:** Linux (explicitly "tested only on Linux")
- **GPU:** NVIDIA GPU with **at least 24 GB** memory (verified on A100/H100 by upstream)
- **CUDA toolkit:** recommended **12.4**
- **Python:** `>=3.8` (setup script creates conda env with Python 3.10)
- **Default torch stack in setup script:** `torch==2.6.0`, `torchvision==0.21.0` (CUDA 12.4 index)

## 2) Reproducible build/runtime components (from setup scripts)

### Core install path

`setup.sh` supports these switches:

- `--basic`
- `--flash-attn`
- `--nvdiffrast`
- `--nvdiffrec`
- `--cumesh`
- `--o-voxel`
- `--flexgemm`

### Compiler / native extension implications

- `o-voxel` builds a `torch.utils.cpp_extension.CUDAExtension` from `/workspace/o-voxel/setup.py`
- `cumesh` and `flexgemm` are CUDA extension packages
- `nvdiffrast` and `nvdiffrec` are native builds
- Effective requirement: **working CUDA toolchain + `CUDA_HOME` + NVIDIA driver stack**

### Attention/sparse backend knobs

- Attention backend env: `ATTN_BACKEND` (`flash_attn`, `xformers`, `flash_attn_3`, `sdpa`, `naive`) in `trellis2/modules/attention/config.py`
- Sparse conv backend env: `SPARSE_CONV_BACKEND` (`flex_gemm` default, `spconv`, `torchsparse`) in `trellis2/modules/sparse/config.py`
- Sparse attention backend env: `SPARSE_ATTN_BACKEND` (or falls back to `ATTN_BACKEND`)

## 3) Resolution expectations and VRAM reality

Upstream publishes timing by resolution (README) but **does not publish a per-resolution VRAM table**.

| Resolution mode | Upstream timing claim | Upstream VRAM statement |
|---|---:|---|
| 512^3 | ~3 s on H100 | No separate figure; global requirement says >=24 GB GPU memory |
| 1024^3 | ~17 s on H100 | No separate figure; global requirement says >=24 GB GPU memory |
| 1536^3 | ~60 s on H100 | No separate figure; global requirement says >=24 GB GPU memory |

Interpretation for planning:

- Treat **24 GB VRAM as the minimum supported baseline** for official Python pipeline usage.
- Lower-memory cards are out of upstream support scope unless using alternative runtimes (see trellis.cpp section).

## 4) RTX 4070 Laptop GPU (8 GB), 32 GB RAM, Windows assessment

### What upstream says

- Linux-only tested
- >=24 GB VRAM required

### What will not fit / is unsupported in upstream path

- **Windows is not an upstream-tested target** for this Python stack.
- **8 GB VRAM is below stated requirement**; full 1024/1536 official path should be treated as unsupported.
- CUDA extension builds in this repo are Linux-oriented and toolchain-heavy (`apt`, Linux assumptions in scripts).

### What likely needs Linux/cloud

- Official TRELLIS.2 Python inference path at meaningful resolutions (especially 1024 cascade / 1536).
- Training and data_toolkit latent extraction stages that call `.cuda()` heavily.
- Any path depending on `cumesh`, `o_voxel`, `flex_gemm`, `nvdiffrast`, `nvdiffrec` builds without a complete CUDA dev environment.

## 5) `pwilkin/trellis.cpp` as native Windows backend (requested evaluation)

Repository inspected: https://github.com/pwilkin/trellis.cpp

- **License:** MIT
- **Current activity snapshot:** created 2026-06-28; latest release `v0.8.1` (2026-09-25); recent updates in Nov 2026
- **Claimed supported stages (README):** full TRELLIS.2-like image->3D pipeline including:
  - background removal
  - image conditioning
  - sparse structure flow
  - shape flow + decoder
  - texture flow + decoder
  - mesh extraction + UV-textured GLB export
- **Windows support in README:** explicit PowerShell install/build paths, prebuilt Windows binaries, Vulkan/CUDA notes
- **Quantization options (model artifacts):**
  - GGUF repositories show full-precision set plus `q4/` and `q8/` variants (e.g. `ilintar/trellis2-gguf`)

Important caveats:

- This is **not** microsoft/TRELLIS.2 official Python runtime; it is an independent reimplementation.
- Model-pack licensing around helper/gated models remains a separate compliance check (`license: other` appears in related GGUF and upstream gated model metadata).
- This phase did **not** run trellis.cpp binaries locally; conclusions are repo-metadata based.

## 6) What we actually observed on this VM

- `nvidia-smi` unavailable (no NVIDIA GPU exposed)
- `python3` available (`3.12.3`)
- Attempted extension installs fail without CUDA toolchain root, e.g.:
  - `OSError: CUDA_HOME environment variable is not set` during `o-voxel` and `CuMesh` build attempts

See `docs/zyrex/TESTS_AND_LIMITATIONS.md` for exact command outputs.
