# Commercial Baker Prototype Report (Laptop Validation)

## Status summary

- Production default remains `ZYREX_BAKER=nvdiffrast`.
- Opt-in path `ZYREX_BAKER=zyrex` is validated on the target laptop for the current PR state.
- Raster default in zyrex is `torch`; Triton is explicit opt-in via `ZYREX_RASTER_IMPL=triton`.

## Current validated laptop results

Validated externally on RTX 4070 Laptop (Windows, Triton 3.2 / triton-windows):

- Tests:
  - `24/24` pass with both torch and triton raster modes.
- Warlord 2048 bake:
  - `nvdiffrast`: `4.2 s`, `0.84 GB`
  - `zyrex + triton`: `4.6 s`, `0.55 GB`
  - `zyrex + torch`: `5.1 s`, `0.55 GB`
- Raster stage:
  - `triton`: `0.59 s`
  - `torch`: `1.49 s`
- Output agreement:
  - mask XOR: `0.13%`
  - shared-coverage PSNR:
    - base color: `79 dB`
    - metallic/roughness: `86 dB`

## Implementation notes reflected in this report

- Zyrex export path (`o_voxel.postprocess`) uses:
  - `flex_gemm.ops.grid_sample.grid_sample_3d` for sparse volume sampling
  - OpenCV Telea (`cv2.INPAINT_TELEA`) for hole filling
- Debug-only artifacts (`face_indices_raw`, sample positions, sampled attrs, UV hashes) are now gated by `collect_debug` / `return_debug` and are off by default.
- Backend resolver supports only:
  - `nvdiffrast` (default)
  - `zyrex`

## Relevant runtime controls

- `ZYREX_BAKER=nvdiffrast|zyrex`
- `ZYREX_RASTER_IMPL=torch|triton`
- `ZYREX_RASTER_PROFILE=1`
- `ZYREX_RASTER_PROFILE_SYNC=1`
- `ZYREX_EMIT_NORMAL_MAP=1` (experimental/unvalidated)

## Parity criterion used in tests

For raster face-id mismatches against nvdiffrast, only true shared-edge ties are accepted:

- faces share an edge, and
- texel center lies on that edge at 8-bit fixed-point tolerance (`|edge_fn| <= 1`).

No UV-interpolation fallback criterion is used.
