# Task

Investigate and prototype a commercially compatible replacement for NVIDIA-licensed rendering/baking dependencies in the Zyrex TRELLIS.2 fork, while keeping the production path unchanged by default.

# Implementation

## Feature-flagged prototype path

- Added `ZYREX_BAKER` backend resolver with supported values:
  - `nvdiffrast` (default, current upstream behavior)
  - `zyrex` (prototype UV baker)
  - `pytorch3d`, `kaolin` (reserved; explicit `NotImplementedError`)
- Wired `o_voxel.postprocess.to_glb` to use `ZYREX_BAKER=zyrex` for a custom UV-space baker and keep default behavior on `nvdiffrast`.
- Kept production NVIDIA path intact and offloaded prototype to explicit opt-in.

## Custom UV baker prototype (`zyrex`)

- Implemented in pure PyTorch:
  - UV-space triangle rasterization
  - Barycentric interpolation
  - Sparse voxel trilinear sampling via dense staging + `torch.nn.functional.grid_sample`
  - Seam dilation/padding
  - Tangent-space normal-map bake
- Added optional mapping callback for post-simplification texels to be projected back onto the original mesh before voxel sampling (preserves current fidelity strategy).

## Call-site map (NVIDIA dependencies)

Detailed map: `docs/zyrex/COMMERCIAL_BAKER_PROTOTYPE.md`.

High-level summary:

| Area | NVIDIA dependency | Operation(s) | Shipping/export path vs training/eval |
|---|---|---|---|
| `o-voxel/o_voxel/postprocess.py` | `nvdiffrast` | UV rasterize/interpolate during GLB texture bake | **Shipping/export path** |
| `trellis2/pipelines/trellis2_texturing.py` | `nvdiffrast` | UV bake for texturing pipeline output | non-default pipeline path |
| `trellis2/renderers/mesh_renderer.py` | `nvdiffrast` | rasterize/interpolate/texture/antialias for rendering | training/eval/preview |
| `trellis2/renderers/pbr_mesh_renderer.py` | `nvdiffrast`, `nvdiffrec_render` | depth peel, texture lookup, env-map split-sum shading | training/eval/preview |
| `trellis2/utils/render_utils.py`, datasets/trainers | indirect renderer usage | render-based visualization/losses | training/eval only |

Closest alternatives documented in map:
- Kaolin (Apache-2.0)
- PyTorch3D (BSD-style)
- Custom Zyrex implementation where no drop-in exists

# Files

- `o-voxel/o_voxel/postprocess.py`
- `o-voxel/o_voxel/zyrex_baker_backend.py`
- `o-voxel/o_voxel/zyrex_uv_baker.py`
- `trellis2/utils/zyrex_baker_compare.py`
- `tests/test_zyrex_uv_baker.py`
- `docs/zyrex/COMMERCIAL_BAKER_PROTOTYPE.md`
- `docs/zyrex/COMMERCIAL_BAKER_PROTOTYPE_REPORT.md`

# Tests

## Command

```bash
python3 -m pytest tests/test_zyrex_uv_baker.py -q
```

## Real output

```text
.......                                                                  [100%]
=============================== warnings summary ===============================
tests/test_zyrex_uv_baker.py::test_feature_flag_default_and_override
  /agent/repos/TRELLIS.2/tests/test_zyrex_uv_baker.py:114: UserWarning: Unsupported ZYREX_BAKER='invalid-backend'; falling back to 'nvdiffrast'. Supported values: ('nvdiffrast', 'pytorch3d', 'kaolin', 'zyrex')
    assert resolve_baker_backend() == DEFAULT_BAKER_BACKEND

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 1 warning in 1.66s
```

# Benchmarks

## Command (CPU synthetic + real-asset sanity)

```bash
python3 trellis2/utils/zyrex_baker_compare.py \
  --output /tmp/zyrex-baker/metrics_cpu_followup.json \
  --texture-size 128 \
  --real-asset /tmp/zyrex-baker/warlord_trellis2.glb \
  --render-resolution 96 \
  --render-max-faces 5000
```

## Measured output (CPU)

- Synthetic:
  - coverage ratio: `0.98443603515625`
  - base_color PSNR/SSIM: `99.0 / 1.0`
  - metallic PSNR/SSIM: `99.0 / 1.0`
  - roughness PSNR/SSIM: `99.0 / 1.0`
  - alpha PSNR/SSIM: `99.0 / 1.0`
  - normal PSNR/SSIM: `99.0 / 1.0`
  - normal orientation check: `true`
  - processing time: `0.014618692000112787 s`
  - peak RSS: `555.08984375 MB`
- Real asset (`warlord_trellis2.glb`) re-bake sanity:
  - texture size: `2048x2048`
  - mesh-footprint UV coverage ratio: `0.5422093868255615`
  - source alpha coverage ratio: `0.999993085861206`
  - union coverage ratio: `0.9999980926513672`
  - **legacy/full-texture** base color RGBA PSNR/SSIM: `16.412878788940326 / 0.9277622699737549`
  - **corrected/covered-texels (footprint)** base color RGBA PSNR/SSIM: `18.608159288557747 / 0.9564441442489624`
  - corrected covered-texels metallic-roughness RGB PSNR/SSIM: `26.110288005954278 / 0.9937278032302856`
  - CPU rendered-view comparison (3 fixed views, union render mask):
    - mean PSNR: `17.510401485102008`
    - mean SSIM: `0.10918882737557094`
    - face budget for CPU render: `5000 / 194529`

## GPU-side comparison status

- `pending GPU run` for:
  - nvdiffrast-vs-zyrex baked map metrics on the same asset
  - rendered-view PSNR/SSIM
  - GPU processing time comparison
  - GPU memory comparison

The comparison harness writes explicit pending fields instead of estimating values when CUDA/nvdiffrast are unavailable.

# Hardware

- OS: `Linux-6.12.113+-x86_64-with-glibc2.39`
- Python: `3.12.3`
- CUDA available on this VM: `false`
- GPU device: `null`

# Known limitations

- Prototype shipping integration is limited to `o_voxel.postprocess.to_glb`.
- `pytorch3d` and `kaolin` backend values are currently placeholders (explicitly not implemented).
- No CUDA on this VM, so direct side-by-side nvdiffrast vs zyrex metrics remain pending.
- CPU rendered-view comparison is approximate (face budget cap) to keep runtime feasible on CPU-only hosts.
- Texturing and renderer subsystems that use `nvdiffrast`/`nvdiffrec_render` are mapped but not replaced in this PR.

# Licensing implications

- Keeps default NVIDIA path unchanged; therefore current commercial blocker remains unless deployments opt into alternative paths and complete migration.
- Introduces a prototype custom path designed to avoid `nvdiffrast`/`nvdiffrec` usage for GLB baking.
- No GPL/AGPL dependencies were added.
- Upstream MIT notices are preserved.

# Status

- **Partially complete / prototype state**:
  - CPU synthetic and real-asset sanity validation: complete
  - CPU rendered-view comparison: complete (approximate, face-capped)
  - GPU equivalence against nvdiffrast: **pending GPU run**
  - Full renderer stack migration away from nvdiffrast/nvdiffrec: **not started in this PR**

# Next task

Run `trellis2/utils/zyrex_baker_compare.py --run-gpu` on a CUDA host with both bakers available, collect side-by-side map + rendered-view metrics, and decide whether to expand `ZYREX_BAKER=zyrex` from prototype to broader export workflows.
