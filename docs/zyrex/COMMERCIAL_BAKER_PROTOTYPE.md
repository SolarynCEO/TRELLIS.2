# Commercial Baker Prototype (ZYREX_BAKER=zyrex)

## Scope

Prototype-only replacement for NVIDIA-licensed UV baking in the **shipping GLB export path** (`o_voxel.postprocess.to_glb`) behind `ZYREX_BAKER`.

- Default remains upstream behavior: `ZYREX_BAKER=nvdiffrast`
- Prototype path: `ZYREX_BAKER=zyrex`
- Reserved but not implemented: `ZYREX_BAKER=pytorch3d|kaolin`

---

## Step 1 — NVIDIA call-site map

| File | Operation(s) | Shipping path? | Closest Kaolin equivalent (Apache-2.0) | Closest PyTorch3D equivalent (BSD-style) | Custom Zyrex requirement |
|---|---|---:|---|---|---|
| `o-voxel/o_voxel/postprocess.py` | **UV-space rasterize**, barycentric **interpolate**, UV mask, position lookup for voxel sampling during GLB bake | **Yes** (app/example GLB export) | `kaolin.render.mesh.rasterize` (+ barycentrics) | `rasterize_meshes` + `interpolate_face_attributes` | Implemented: CPU/GPU torch UV rasterizer + barycentric interpolation + sparse-volume trilinear sampling + seam dilation + tangent-space normal map |
| `trellis2/pipelines/trellis2_texturing.py::postprocess_mesh` | UV-space rasterize + interpolate + voxel sampling for texturing pipeline output mesh | No (not in default app export path) | same as above | same as above | Not switched in this PR (left on existing path) |
| `trellis2/renderers/mesh_renderer.py` | Camera-space **rasterize**, interpolate depth/normal/coord/attr, **antialias**, UV **texture sampling** | No (render preview/training/eval) | Rasterization + texture utilities in Kaolin; no direct drop-in antialias parity | Rasterization/interpolation + texture sampling APIs; antialias needs custom supersampling/postfilter | Needs dedicated renderer migration; out of scope for this PR |
| `trellis2/renderers/pbr_mesh_renderer.py` | Depth peeling rasterizer, interpolation, texture sampling, env-map lookup | No (render preview/training/eval) | Kaolin rendering building blocks; custom PBR shader path still required | PyTorch3D custom shader stack required | Needs full renderer parity work; out of scope |
| `trellis2/renderers/pbr_mesh_renderer.py::EnvMap` (`nvdiffrec_render.light`) | nvdiffrec renderutils/envlight split-sum shading + mip build | No (render preview/training/eval) | Partial building blocks only | Partial building blocks only | Requires custom IBL/split-sum implementation to remove nvdiffrec |
| `trellis2/utils/render_utils.py` | Indirect usage via `MeshRenderer` / `PbrMeshRenderer` | No (preview/video snapshots) | indirect | indirect | blocked on renderer migration |
| `trellis2/datasets/*`, `trellis2/trainers/vae/*` | Indirect usage via renderers for dataset visualization and render losses | Training/eval only | indirect | indirect | blocked on renderer migration |
| `setup.sh`, `README.md`, `THIRD_PARTY_NOTICES.md` | Installation and license declaration of `nvdiffrast` / `nvdiffrec` | N/A | N/A | N/A | Commercial blocker already documented in notices |

---

## Step 2 — Prototype implementation summary

Implemented modules:

- `o-voxel/o_voxel/zyrex_baker_backend.py` (plus root helper mirror for tests/scripts)
  - Resolves `ZYREX_BAKER` values with default fallback to `nvdiffrast`
- `o-voxel/o_voxel/zyrex_uv_baker.py` (loaded via `zyrex_uv_baker.py` helper for scripts/tests)
  - UV triangle rasterization (torch)
  - Barycentric interpolation (torch)
  - Sparse voxel trilinear sampling (torch `grid_sample`)
  - Seam dilation/padding (iterative neighbor fill)
  - Tangent-space normal-map baking
- `o-voxel/o_voxel/postprocess.py`
  - Adds feature-flagged branch:
    - `nvdiffrast` path unchanged (default)
    - `zyrex` path uses the custom baker
    - `kaolin`/`pytorch3d` explicitly reserved (`NotImplementedError`)

Behavioral constraints:

- No production default replacement.
- No NVIDIA path removal.
- No GPL/AGPL dependency introduction.

---

## Step 3 — Comparison harness

Script: `trellis2/utils/zyrex_baker_compare.py`

- CPU synthetic validation against analytic expectations
- Optional real-asset UV re-bake sanity check from GLB textures
- GPU side-by-side mode skeleton for `nvdiffrast` vs `zyrex` with fixed seeds and JSON metrics output
- On non-CUDA hosts, all GPU metrics are emitted as **`pending GPU run`** (never estimated)
