import sys
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image


def _install_optional_native_stubs() -> None:
    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")

        class _DummyHFModel:
            def __init__(self):
                self.embeddings = types.SimpleNamespace(
                    patch_embeddings=types.SimpleNamespace(weight=torch.zeros(1)),
                )
                self.layer = []

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def eval(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

            def cpu(self):
                return self

            def cuda(self):
                return self

            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def __call__(self, *_args, **_kwargs):
                return [torch.zeros((1, 1, 1, 1))]

        transformers_mod.DINOv3ViTModel = _DummyHFModel
        transformers_mod.AutoModelForImageSegmentation = _DummyHFModel
        sys.modules["transformers"] = transformers_mod

    if "torchvision" not in sys.modules:
        torchvision_mod = types.ModuleType("torchvision")
        transforms_mod = types.ModuleType("torchvision.transforms")

        class _Compose:
            def __init__(self, funcs):
                self.funcs = funcs

            def __call__(self, value):
                for func in self.funcs:
                    value = func(value)
                return value

        class _IdentityTransform:
            def __init__(self, *_args, **_kwargs):
                pass

            def __call__(self, value):
                return value

        transforms_mod.Compose = _Compose
        transforms_mod.Normalize = _IdentityTransform
        transforms_mod.Resize = _IdentityTransform
        transforms_mod.ToTensor = _IdentityTransform
        transforms_mod.ToPILImage = _IdentityTransform
        torchvision_mod.transforms = transforms_mod
        sys.modules["torchvision"] = torchvision_mod
        sys.modules["torchvision.transforms"] = transforms_mod

    if "cumesh" not in sys.modules:
        class _DummyCuMesh:
            def init(self, *_args, **_kwargs):
                return None

            def __getattr__(self, _name):
                def _noop(*_args, **_kwargs):
                    return None
                return _noop

            def read(self):
                return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.int32)

        sys.modules["cumesh"] = types.SimpleNamespace(CuMesh=_DummyCuMesh)

    if "flex_gemm.ops.grid_sample" not in sys.modules:
        flex_gemm_mod = types.ModuleType("flex_gemm")
        flex_gemm_ops_mod = types.ModuleType("flex_gemm.ops")
        flex_gemm_grid_sample_mod = types.ModuleType("flex_gemm.ops.grid_sample")

        def _grid_sample_3d(*_args, **_kwargs):
            return torch.zeros((1, 0, 0))

        flex_gemm_grid_sample_mod.grid_sample_3d = _grid_sample_3d
        sys.modules["flex_gemm"] = flex_gemm_mod
        sys.modules["flex_gemm.ops"] = flex_gemm_ops_mod
        sys.modules["flex_gemm.ops.grid_sample"] = flex_gemm_grid_sample_mod


_install_optional_native_stubs()

from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline  # noqa: E402


class TrackingFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 2
        self.resolution = 4
        self.weight = nn.Parameter(torch.ones(1))
        self.to_calls = []
        self.cpu_calls = 0

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)

    def cpu(self):
        self.cpu_calls += 1
        return super().cpu()


class TrackingDecoder(TrackingFlowModel):
    def __init__(self):
        super().__init__()
        self.low_vram = False

    def forward(self, z):
        return z[:, :1]

    def set_resolution(self, _resolution):
        return None

    def upsample(self, slat, upsample_times=4):
        _ = upsample_times
        return slat.coords


class DummySampler:
    def sample(self, model, noise, **_kwargs):
        _ = model
        return types.SimpleNamespace(samples=noise)


def _build_pipeline(low_vram: bool = False) -> Trellis2ImageTo3DPipeline:
    models = {
        "sparse_structure_flow_model": TrackingFlowModel(),
        "sparse_structure_decoder": TrackingDecoder(),
        "shape_slat_flow_model_512": TrackingFlowModel(),
        "shape_slat_flow_model_1024": TrackingFlowModel(),
        "shape_slat_decoder": TrackingDecoder(),
        "tex_slat_flow_model_512": TrackingFlowModel(),
        "tex_slat_flow_model_1024": TrackingFlowModel(),
        "tex_slat_decoder": TrackingDecoder(),
    }
    return Trellis2ImageTo3DPipeline(
        models=models,
        sparse_structure_sampler=DummySampler(),
        shape_slat_sampler=DummySampler(),
        tex_slat_sampler=DummySampler(),
        sparse_structure_sampler_params={},
        shape_slat_sampler_params={},
        tex_slat_sampler_params={},
        shape_slat_normalization={"std": [1.0, 1.0], "mean": [0.0, 0.0]},
        tex_slat_normalization={"std": [1.0, 1.0], "mean": [0.0, 0.0]},
        image_cond_model=types.SimpleNamespace(to=lambda *_: None, cpu=lambda: None, image_size=512, __call__=lambda *_: torch.zeros((1, 4))),
        rembg_model=types.SimpleNamespace(to=lambda *_: None, cpu=lambda: None, __call__=lambda image: image.convert("RGBA")),
        low_vram=low_vram,
    )


def test_env_flag_enables_default_profile(monkeypatch):
    monkeypatch.setenv("ZYREX_LOW_VRAM", "1")
    pipeline = _build_pipeline()
    assert pipeline.vram_profile == "8gb"


def test_profile_dtype_parsing(monkeypatch):
    pipeline = _build_pipeline()
    monkeypatch.setenv("ZYREX_LOW_VRAM_DTYPE", "bf16")
    assert pipeline._resolve_autocast_dtype(pipeline.vram_profiles["8gb"]) == torch.bfloat16
    monkeypatch.setenv("ZYREX_LOW_VRAM_DTYPE", "none")
    assert pipeline._resolve_autocast_dtype(pipeline.vram_profiles["8gb"]) is None


def test_sample_sparse_structure_offloads_models():
    pipeline = _build_pipeline(low_vram=True)
    coords = pipeline.sample_sparse_structure(cond={}, resolution=4, num_samples=1, sampler_params={})
    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    assert coords.shape[1] == 4
    assert flow_model.cpu_calls >= 1
    assert decoder.cpu_calls >= 1
    assert len(flow_model.to_calls) >= 1


def test_run_returns_profile_metrics_and_defaults():
    pipeline = _build_pipeline(low_vram=False)
    dummy_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    dummy_shape = {"shape": "latent"}
    dummy_tex = {"tex": "latent"}

    pipeline.preprocess_image = lambda image: image
    pipeline.get_cond = lambda _images, _resolution, include_neg_cond=True: {"cond": torch.zeros((1, 2)), "neg_cond": torch.zeros((1, 2))} if include_neg_cond else {"cond": torch.zeros((1, 2))}
    pipeline.sample_sparse_structure = lambda *_args, **_kwargs: dummy_coords
    pipeline.sample_shape_slat = lambda *_args, **_kwargs: dummy_shape
    pipeline.sample_tex_slat = lambda *_args, **_kwargs: dummy_tex
    pipeline.decode_latent = lambda *_args, **_kwargs: ["mesh"]

    meshes, run_metrics = pipeline.run(
        image=Image.new("RGB", (8, 8)),
        return_metrics=True,
        vram_profile="8gb",
    )
    assert meshes == ["mesh"]
    assert run_metrics["pipeline_type"] == "512"
    assert run_metrics["max_num_tokens"] == 32768
    assert run_metrics["low_vram"] is True
    assert any(stage["stage"] == "condition_512" for stage in run_metrics["stage_metrics"])
    assert pipeline.last_run_stats == run_metrics


def test_run_with_return_latent_and_metrics():
    pipeline = _build_pipeline(low_vram=False)
    dummy_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
    dummy_shape = {"shape": "latent"}
    dummy_tex = {"tex": "latent"}

    pipeline.preprocess_image = lambda image: image
    pipeline.get_cond = lambda *_args, **_kwargs: {"cond": torch.zeros((1, 2)), "neg_cond": torch.zeros((1, 2))}
    pipeline.sample_sparse_structure = lambda *_args, **_kwargs: dummy_coords
    pipeline.sample_shape_slat = lambda *_args, **_kwargs: dummy_shape
    pipeline.sample_tex_slat = lambda *_args, **_kwargs: dummy_tex
    pipeline.decode_latent = lambda *_args, **_kwargs: ["mesh"]

    meshes, latent, metrics = pipeline.run(
        image=Image.new("RGB", (8, 8)),
        return_latent=True,
        return_metrics=True,
        vram_profile="8gb",
    )
    assert meshes == ["mesh"]
    assert latent == (dummy_shape, dummy_tex, 512)
    assert metrics["vram_profile"] == "8gb"


def test_invalid_profile_raises():
    pipeline = _build_pipeline()
    with pytest.raises(ValueError):
        pipeline._resolve_vram_profile("nope")
