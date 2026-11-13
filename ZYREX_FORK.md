# Zyrex TRELLIS.2 Fork Audit (Phase 0)

## Repository identity and upstream pin

- **Fork repository:** `SolarynCEO/TRELLIS.2`
- **Intended maintained name:** `zyrex-trellis2`
- **Upstream repository:** `microsoft/TRELLIS.2`
- **Upstream branch audited:** `main`
- **Exact upstream commit SHA this fork is based on:**  
  `75fbf0183001ed9876c8dbb35de6b68552ee08bd`

Verification performed in this checkout:

- `git rev-parse HEAD` -> `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- `git rev-parse upstream/main` -> `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- `git merge-base HEAD upstream/main` -> `75fbf0183001ed9876c8dbb35de6b68552ee08bd`
- `git rev-list --left-right --count upstream/main...HEAD` -> `0 0` (no divergence at audit start)

## Submodules (URL, pin, license)

| Path | Remote URL | Pinned commit | License (verified) | Notes |
|---|---|---:|---|---|
| `o-voxel/third_party/eigen` | `https://gitlab.com/libeigen/eigen.git` | `21e4582d1739107337a03460c81412981130373e` | **MPL-2.0** primary (`LICENSE`/`COPYING.MPL2`) plus bundled third-party notices (`COPYING.BSD`, `COPYING.APACHE`, etc.) | Checked via `git submodule status --recursive` and local files after `git submodule update --init --recursive` |

## Upstream vs. Zyrex-owned changes

### Upstream code/content (from `microsoft/TRELLIS.2`)

All core implementation content in this fork is upstream TRELLIS.2 material at the pinned commit, including:

- `trellis2/` (pipelines, models, trainers, renderers, sparse modules)
- `o-voxel/` (O-Voxel package and CUDA extension sources)
- `configs/`, `data_toolkit/`
- Root scripts/apps (`app.py`, `app_texturing.py`, `example*.py`, `train.py`, `setup.sh`)

### Zyrex modifications in this branch

This branch adds **documentation only** for fork governance/audit:

- `ZYREX_FORK.md` (this file)
- `THIRD_PARTY_NOTICES.md`
- `docs/zyrex/ARCHITECTURE.md`
- `docs/zyrex/ENVIRONMENT.md`
- `docs/zyrex/TESTS_AND_LIMITATIONS.md`

No Zyrex product runtime/orchestrator code is added here.

## Repository boundary statement

- This repo is the TRELLIS.2 model fork and technical baseline.
- **Original SolarynTech product modules live in a separate repository** (`SolarynCEO/Zyrex`) and are intentionally out of scope here.

## License and notice preservation

- Upstream `LICENSE` (MIT) is preserved.
- Existing upstream copyright/license notices are left untouched.
- No upstream notice file was removed or rewritten in this phase.
