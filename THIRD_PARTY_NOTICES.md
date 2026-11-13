# THIRD_PARTY_NOTICES

This document inventories third-party components referenced by this fork at audit time.
It is a technical compliance aid, not legal advice.

## Audit scope and evidence sources

- Repository scripts/configs inspected:
  - `/workspace/setup.sh`
  - `/workspace/data_toolkit/setup.sh`
  - `/workspace/o-voxel/pyproject.toml`
  - `/workspace/trellis2/pipelines/*.py`
- Hugging Face model metadata inspected via Hub APIs/model cards.
- GitHub license metadata inspected via `gh api repos/<owner>/<repo>/license`.

Status labels:

- **Verified**: License string/text directly observed from repository metadata or local file.
- **Unverified**: Not directly verifiable from this checkout/tooling; do not assume.

---

## 1) Python dependencies declared by this repo

> These are dependencies explicitly declared in `setup.sh`, `data_toolkit/setup.sh`, or `o-voxel/pyproject.toml`.
> If license metadata was missing in this environment, it is marked **unverified**.

| Dependency | Declared in | Source | License | Commercial use / redistribution |
|---|---|---|---|---|
| `torch==2.6.0` (script pin) | `setup.sh --new-env` | https://pypi.org/project/torch/ | **Unverified** in local metadata | **Unverified** |
| `torchvision==0.21.0` (script pin) | `setup.sh --new-env` | https://pypi.org/project/torchvision/ | BSD (metadata) | Permissive; generally allowed |
| `imageio` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/imageio/ | **Unverified** in local metadata | **Unverified** |
| `imageio-ffmpeg` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/imageio-ffmpeg/ | **Unverified** (not installed in this VM) | **Unverified** |
| `tqdm` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/tqdm/ | MPL-2.0 AND MIT (metadata) | Permissive/copyleft-lite; generally allowed |
| `easydict` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/easydict/ | **Unverified** in local metadata | **Unverified** |
| `opencv-python-headless` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/opencv-python-headless/ | Apache-2.0 (metadata) | Permissive; generally allowed |
| `ninja` | `setup.sh --basic` | https://pypi.org/project/ninja/ | **Unverified** (not installed) | **Unverified** |
| `trimesh` | `setup.sh --basic` | https://pypi.org/project/trimesh/ | MIT (metadata) | Permissive; generally allowed |
| `transformers` | `setup.sh --basic` | https://pypi.org/project/transformers/ | Apache-2.0 (metadata) | Permissive; generally allowed |
| `gradio==6.0.1` | `setup.sh --basic` | https://pypi.org/project/gradio/ | **Unverified** (not installed) | **Unverified** |
| `tensorboard` | `setup.sh --basic` | https://pypi.org/project/tensorboard/ | **Unverified** (not installed) | **Unverified** |
| `pandas` | `setup.sh --basic`, `data_toolkit/setup.sh` | https://pypi.org/project/pandas/ | **Unverified** (not installed) | **Unverified** |
| `lpips` | `setup.sh --basic` | https://pypi.org/project/lpips/ | **Unverified** (not installed) | **Unverified** |
| `zstandard` | `setup.sh --basic`, `o-voxel/pyproject.toml` | https://pypi.org/project/zstandard/ | **Unverified** (not installed) | **Unverified** |
| `pillow-simd` | `setup.sh --basic` | https://pypi.org/project/Pillow-SIMD/ | **Unverified** (not installed) | **Unverified** |
| `kornia` | `setup.sh --basic` | https://pypi.org/project/kornia/ | **Unverified** (not installed) | **Unverified** |
| `timm` | `setup.sh --basic` | https://pypi.org/project/timm/ | **Unverified** (not installed) | **Unverified** |
| `open3d` | `data_toolkit/setup.sh` | https://pypi.org/project/open3d/ | **Unverified** (not installed) | **Unverified** |
| `objaverse` | `data_toolkit/setup.sh` | https://pypi.org/project/objaverse/ | **Unverified** (not installed) | **Unverified** |
| `huggingface_hub[cli]` | `data_toolkit/setup.sh` | https://pypi.org/project/huggingface-hub/ | Apache-2.0 (metadata) | Permissive; generally allowed |
| `open_clip_torch` | `data_toolkit/setup.sh` | https://pypi.org/project/open-clip-torch/ | **Unverified** (not installed) | **Unverified** |
| `numpy` | transitive + `o-voxel/pyproject.toml` | https://pypi.org/project/numpy/ | **Unverified** in local metadata | **Unverified** |
| `plyfile` | `o-voxel/pyproject.toml` | https://pypi.org/project/plyfile/ | **Unverified** (not installed) | **Unverified** |
| `safetensors` | runtime import in `trellis2/models/__init__.py` | https://pypi.org/project/safetensors/ | **Unverified** in local metadata | **Unverified** |
| `utils3d` (git) | `setup.sh --basic` | https://github.com/EasternJournalist/utils3d (pin: `9a4eb15e4021b67b12c460c7057d642626897ec8`) | MIT (repo license metadata) | Permissive; generally allowed |

---

## 2) CUDA/C++ extension and accelerated backend dependencies

| Component | How referenced | Source | License | Commercial/redistribution status |
|---|---|---|---|---|
| `o_voxel` (local extension) | `setup.sh --o-voxel`; imports in pipeline and data_toolkit | `/workspace/o-voxel` | **Unverified package-specific** (repo root is MIT) | **Unverified** |
| `cumesh` | `setup.sh --cumesh`; imported by mesh/postprocess modules | https://github.com/JeffreyXiang/CuMesh | MIT | Permissive; generally allowed |
| `flex_gemm` | `setup.sh --flexgemm`; sparse conv/grid sampling | https://github.com/JeffreyXiang/FlexGEMM | MIT | Permissive; generally allowed |
| `nvdiffrast` | `setup.sh --nvdiffrast` | https://github.com/NVlabs/nvdiffrast | NVIDIA Source Code License (1-Way Commercial) | **⚠️ NON-COMMERCIAL LIMITATION in license text section 3.3** |
| `nvdiffrec` (JeffreyXiang fork, `renderutils` branch) | `setup.sh --nvdiffrec` | https://github.com/JeffreyXiang/nvdiffrec | NVIDIA Source Code License for nvdiffrec | **⚠️ NON-COMMERCIAL LIMITATION in license text section 3.3** |
| `flash-attn==2.7.3` | `setup.sh --flash-attn`; default attention backend | https://github.com/Dao-AILab/flash-attention | BSD-3-Clause | Permissive; generally allowed |
| `xformers` (optional fallback backend) | Mentioned in README and attention config | https://github.com/facebookresearch/xformers | BSD 3-Clause text in repo `LICENSE` | Permissive; generally allowed |
| `spconv` (optional sparse conv backend) | `SPARSE_CONV_BACKEND=spconv` path | https://github.com/traveller59/spconv | Apache-2.0 | Permissive; generally allowed |
| `torchsparse` (optional sparse conv backend) | `SPARSE_CONV_BACKEND=torchsparse` path | https://github.com/mit-han-lab/torchsparse | MIT | Permissive; generally allowed |

**Important:** The verified NVIDIA license texts for `nvdiffrast` and `nvdiffrec` include a non-commercial use limitation in section 3.3 (research/evaluation-only for non-NVIDIA parties). Treat production/commercial redistribution as blocked unless legal review approves an alternative path.

---

## 3) Git submodule notice

| Submodule | URL | Pin | License |
|---|---|---:|---|
| `o-voxel/third_party/eigen` | https://gitlab.com/libeigen/eigen.git | `21e4582d1739107337a03460c81412981130373e` | MPL-2.0 primary; additional bundled notices in `COPYING.*` |

---

## 4) Pretrained weights / encoders downloaded by pipelines

### 4.1 TRELLIS model checkpoints (from `microsoft/TRELLIS.2-4B`)

Source repo: https://huggingface.co/microsoft/TRELLIS.2-4B  
Model card license metadata: **MIT**

Artifacts referenced by `pipeline.json` / `texturing_pipeline.json`:

- `ckpts/ss_flow_img_dit_1_3B_64_bf16(.json/.safetensors)`
- `ckpts/shape_dec_next_dc_f16c32_fp16(.json/.safetensors)`
- `ckpts/shape_enc_next_dc_f16c32_fp16(.json/.safetensors)` (texturing pipeline)
- `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16(.json/.safetensors)`
- `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16(.json/.safetensors)`
- `ckpts/tex_dec_next_dc_f16c32_fp16(.json/.safetensors)`
- `ckpts/tex_enc_next_dc_f16c32_fp16(.json/.safetensors)` (data_toolkit default)
- `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16(.json/.safetensors)`
- `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16(.json/.safetensors)`

Commercial use / redistribution: MIT metadata suggests permissive usage; still verify model-card terms for downstream distribution obligations.

### 4.2 Sparse-structure decoder/encoder from TRELLIS v1 model repo

Source repo: https://huggingface.co/microsoft/TRELLIS-image-large  
Model card license metadata: **MIT**

Artifacts referenced:

- `ckpts/ss_dec_conv3d_16l8_fp16(.json/.safetensors)` (pipeline)
- `ckpts/ss_enc_conv3d_16l8_fp16(.json/.safetensors)` (data_toolkit default)

Commercial use / redistribution: permissive per MIT metadata (verify card text in deployment process).

### 4.3 Image encoder model (gated)

Source repo: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m  
Status: **Gated**  
License metadata: **other** (not SPDX-standard)

Used by: `DinoV3FeatureExtractor` in `trellis2/modules/image_feature_extractor.py`; model name set in HF `pipeline.json`.

Commercial use / redistribution: **Unclear / unverified** from this audit; treat as restricted pending legal review.

### 4.4 Background-removal model (gated default in HF pipeline config)

Source repo: https://huggingface.co/briaai/RMBG-2.0  
Status: **Gated**  
License metadata: **other**

Used by HF pipeline configs as `rembg_model.args.model_name = "briaai/RMBG-2.0"`.

Commercial use / redistribution: **Unclear / unverified** from this audit; terms were inaccessible without gated access.

### 4.5 Background-removal fallback class default (ungated)

Source repo: https://huggingface.co/ZhengPeng7/BiRefNet  
License metadata: **MIT**

Used by: `BiRefNet` class default in `trellis2/pipelines/rembg/BiRefNet.py` (if no override is provided).

Commercial use / redistribution: permissive per MIT metadata.

---

## 5) Immediate licensing risk summary

1. **High risk / blocker for commercial deployment in current form:** `nvdiffrast` and `nvdiffrec` licenses include non-commercial limitations.
2. **Unclear rights due to gated “license: other” models:** `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0`.
3. No GPL/AGPL components were positively identified in verified entries above, but many Python package licenses remain **unverified** in this phase and require a dedicated legal pass before release.
