# Windows native extension build (MSVC 2022 + CUDA 12.4 + torch 2.6.0)

This guide targets the Windows toolchain that was validated on user hardware:

- Visual Studio 2022 Build Tools 17.14 (`vcvars64.bat`)
- CUDA Toolkit 12.4
- Python 3.12 (venv)
- `torch==2.6.0+cu124`
- RTX 4070 Laptop GPU (`sm_89`)

The automation script is:

- `scripts/windows/build_extensions.ps1`

It builds:

1. `nvdiffrast` (pinned `v0.4.0`)
2. `CuMesh` (pinned commit `12289e1062f0603f2f0d0771b02e1395d247f26f`)
3. `FlexGEMM` (pinned commit `6dd94a859c26ee8246888502eada3dd8ad85532e`, with local MSVC patch)
4. local `o-voxel`

## Why this script exists

- `o-voxel` uses the Eigen submodule under `o-voxel/third_party/eigen`; missing submodules cause `Eigen/Dense` include failures.
- Windows/MSVC needs different compile flags than Linux/GCC/Clang.
- FlexGEMM at the pinned commit needs a small CUDA template parsing patch under MSVC.
- TRELLIS.2 defaults to `flex_gemm` for sparse conv (`trellis2/modules/sparse/config.py`), so FlexGEMM is in the default runtime path.

## Triton on Windows

FlexGEMM uses Triton kernels. On Windows, install `triton-windows`.

- The script installs/updates `triton-windows` automatically before FlexGEMM build.
- PyPI metadata currently reports `triton-windows` license as `MIT` (permissive).

## Usage

From a PowerShell prompt in the repo root:

```powershell
.\scripts\windows\build_extensions.ps1 -PythonExe C:\path\to\venv\Scripts\python.exe
```

Optional overrides:

```powershell
.\scripts\windows\build_extensions.ps1 `
  -PythonExe C:\path\to\venv\Scripts\python.exe `
  -VcVarsPath "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" `
  -CudaHome "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" `
  -TorchCudaArchList "8.9"
```

## What the script does

1. Imports MSVC environment from `vcvars64.bat`.
2. Sets:
   - `CUDA_HOME` / `CUDA_PATH`
   - `TORCH_CUDA_ARCH_LIST` (defaults to `8.9`, overridable)
   - `DISTUTILS_USE_SDK=1`
3. Runs `git submodule update --init --recursive` in this repo.
4. Verifies `o-voxel/third_party/eigen/Eigen/Dense` exists.
5. Clones external extension repos at pinned commits.
6. Applies `patches/flexgemm-windows-msvc.patch`.
7. Builds each extension with `pip install --no-build-isolation`.
8. Writes per-step logs under `build-logs/windows-extensions-<timestamp>/`.

## Troubleshooting

### `fatal error C1083: cannot open include file 'Eigen/Dense'`

Run:

```powershell
git submodule update --init --recursive
```

If you are using a shallow/partial clone, ensure submodules are allowed and reachable.

### MSVC or CUDA not found

- Verify Build Tools 2022 are installed.
- Verify the `vcvars64.bat` path.
- Verify `CUDA_HOME` points to CUDA 12.4.

### Wrong GPU arch

Set:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "8.9"
```

or pass `-TorchCudaArchList` to the script.
