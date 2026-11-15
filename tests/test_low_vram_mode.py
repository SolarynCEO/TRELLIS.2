import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image


def _install_optional_native_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    if "tqdm" not in sys.modules:
        tqdm_mod = types.ModuleType("tqdm")

        def _tqdm(iterable=None, *args, **kwargs):
            _ = args, kwargs
            return iterable

        tqdm_mod.tqdm = _tqdm
        monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    if "easydict" not in sys.modules:
        easydict_mod = types.ModuleType("easydict")

        class _EasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

        easydict_mod.EasyDict = _EasyDict
        monkeypatch.setitem(sys.modules, "easydict", easydict_mod)

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
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

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
        monkeypatch.setitem(sys.modules, "torchvision", torchvision_mod)
        monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms_mod)

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

        monkeypatch.setitem(sys.modules, "cumesh", types.SimpleNamespace(CuMesh=_DummyCuMesh))

    if "flex_gemm.ops.grid_sample" not in sys.modules:
        flex_gemm_mod = types.ModuleType("flex_gemm")
        flex_gemm_ops_mod = types.ModuleType("flex_gemm.ops")
        flex_gemm_grid_sample_mod = types.ModuleType("flex_gemm.ops.grid_sample")

        def _grid_sample_3d(*_args, **_kwargs):
            return torch.zeros((1, 0, 0))

        flex_gemm_grid_sample_mod.grid_sample_3d = _grid_sample_3d
        monkeypatch.setitem(sys.modules, "flex_gemm", flex_gemm_mod)
        monkeypatch.setitem(sys.modules, "flex_gemm.ops", flex_gemm_ops_mod)
        monkeypatch.setitem(sys.modules, "flex_gemm.ops.grid_sample", flex_gemm_grid_sample_mod)

@pytest.fixture
def pipeline_cls(monkeypatch: pytest.MonkeyPatch):
    _install_optional_native_stubs(monkeypatch)
    module = importlib.import_module("trellis2.pipelines.trellis2_image_to_3d")
    return module.Trellis2ImageTo3DPipeline


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


def _build_pipeline(pipeline_cls, low_vram: bool = False):
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
    return pipeline_cls(
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


def test_env_flag_enables_default_profile(monkeypatch, pipeline_cls):
    monkeypatch.setenv("ZYREX_LOW_VRAM", "1")
    pipeline = _build_pipeline(pipeline_cls)
    assert pipeline.vram_profile == "8gb"


def test_profile_dtype_parsing(monkeypatch, pipeline_cls):
    pipeline = _build_pipeline(pipeline_cls)
    monkeypatch.setenv("ZYREX_LOW_VRAM_DTYPE", "bf16")
    assert pipeline._resolve_autocast_dtype(pipeline.vram_profiles["8gb"]) == torch.bfloat16
    monkeypatch.setenv("ZYREX_LOW_VRAM_DTYPE", "none")
    assert pipeline._resolve_autocast_dtype(pipeline.vram_profiles["8gb"]) is None


def test_sample_sparse_structure_offloads_models(pipeline_cls):
    pipeline = _build_pipeline(pipeline_cls, low_vram=True)
    coords = pipeline.sample_sparse_structure(cond={}, resolution=4, num_samples=1, sampler_params={})
    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    assert coords.shape[1] == 4
    assert flow_model.cpu_calls >= 1
    assert decoder.cpu_calls >= 1
    assert len(flow_model.to_calls) >= 1


def test_run_returns_profile_metrics_and_defaults(pipeline_cls):
    pipeline = _build_pipeline(pipeline_cls, low_vram=False)
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


def test_run_with_return_latent_and_metrics(pipeline_cls):
    pipeline = _build_pipeline(pipeline_cls, low_vram=False)
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


def test_invalid_profile_raises(pipeline_cls):
    pipeline = _build_pipeline(pipeline_cls)
    with pytest.raises(ValueError):
        pipeline._resolve_vram_profile("nope")


def test_explicit_1024_profile_env(monkeypatch, pipeline_cls):
    monkeypatch.setenv('ZYREX_VRAM_PROFILE', '8gb_1024')
    monkeypatch.setenv('ZYREX_LOW_VRAM', '1')
    pipeline = _build_pipeline(pipeline_cls)
    assert pipeline.vram_profile == '8gb_1024'
    assert pipeline._resolve_vram_profile('off') is None
    assert pipeline.vram_profiles['8gb_1024']['default_pipeline_type'] == '1024_cascade'


@pytest.mark.parametrize('kwargs', [
    {'pipeline_type': '512'}, {'num_samples': 2}, {'max_num_tokens': 0},
    {'decoder_chunk_size': 0},
])
def test_1024_rejects_invalid_config_before_moving_models(pipeline_cls, kwargs):
    pipeline = _build_pipeline(pipeline_cls)
    with pytest.raises(ValueError):
        pipeline.run(Image.new('RGB', (8, 8)), vram_profile='8gb_1024', **kwargs)
    assert pipeline.low_vram is False
    assert all(not m.to_calls and not m.cpu_calls for m in pipeline.models.values())


def test_sampler_history_opt_out_preserves_real_cpu_euler_result(pipeline_cls):
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
    sampler = FlowEulerSampler(sigma_min=0.001)
    noise = torch.randn(2, 4)
    def model(x, t, cond):
        return x * 0.2 + t[:, None] * 0.0001
    full = sampler.sample(model, noise, steps=4, verbose=False)
    small = sampler.sample(model, noise, steps=4, verbose=False, retain_history=False)
    torch.testing.assert_close(full.samples, small.samples, rtol=0, atol=0)
    assert len(full.pred_x_t) == len(full.pred_x_0) == 4
    assert small.pred_x_t == small.pred_x_0 == []


def _synthetic_1024(monkeypatch, pipeline_cls):
    """Mock generation and native extraction; exercise real orchestration on CPU."""
    pipeline = _build_pipeline(pipeline_cls)
    coords = torch.tensor([[0, 0, 0, 0], [0, 16, 16, 16]], dtype=torch.int32)
    shape = types.SimpleNamespace(coords=coords)
    sampled = []
    pipeline.preprocess_image = lambda image: image
    pipeline.get_cond = lambda *args: {'cond': torch.zeros(1, 2), 'neg_cond': torch.zeros(1, 2)}
    def sparse(cond, resolution, num_samples, params):
        sampled.append(params)
        return coords
    def sample_shape(cond, model, coords, params):
        sampled.append(params)
        return shape
    def sample_tex(cond, model, shape, params):
        sampled.append(params)
        return 'texture latent'
    pipeline.sample_sparse_structure = sparse
    pipeline.sample_shape_slat = sample_shape
    pipeline.sample_tex_slat = sample_tex
    mesh = types.SimpleNamespace(vertices=torch.zeros(3, 3), faces=torch.zeros(1, 3), fill_holes=lambda: None)
    voxels = types.SimpleNamespace(coords=coords, feats=torch.zeros(2, 6), shape=[1, 6], spatial_shape=[1024]*3)
    pipeline.decode_shape_slat = lambda slat, res: ([mesh], [])
    pipeline.decode_tex_slat = lambda slat, subs: [voxels]
    from trellis2.pipelines import low_vram_1024
    monkeypatch.setattr(low_vram_1024, 'MeshWithVoxel', lambda *args, **kwargs: kwargs)
    return pipeline, sampled


@pytest.mark.parametrize('return_latent', [False, True])
def test_1024_mock_generation_restores_flags_and_records_cpu_stats(monkeypatch, pipeline_cls, return_latent):
    pipeline, sampled = _synthetic_1024(monkeypatch, pipeline_cls)
    result = pipeline.run(Image.new('RGB', (8, 8)), vram_profile='8gb_1024',
                          return_metrics=True, return_latent=return_latent)
    metrics = result[-1]
    assert metrics['status'] == 'ok'
    assert metrics['output_resolution'] == 1024
    assert metrics['num_tokens_1024'] == 2
    assert metrics['peak_vram_reserved_bytes'] is None
    assert metrics['target_met'] is None
    assert all(params['retain_history'] is False for params in sampled)
    assert pipeline.low_vram is False
    assert all(model.cpu_calls >= 2 for model in pipeline.models.values())
    decoder = pipeline.models['shape_slat_decoder']
    assert not hasattr(decoder, 'stream_device')
    assert not hasattr(decoder, 'stage_profiler')
    assert decoder.low_vram is False
    if return_latent:
        assert result[1][-1] == 1024


def test_1024_token_limit_fails_without_downgrade_and_cleans_up(monkeypatch, pipeline_cls):
    pipeline, sampled = _synthetic_1024(monkeypatch, pipeline_cls)
    with pytest.raises(ValueError, match='No resolution downgrade'):
        pipeline.run(Image.new('RGB', (8, 8)), vram_profile='8gb_1024', max_num_tokens=1)
    assert len(sampled) == 2  # No HR or texture sampling after rejection.
    assert pipeline.low_vram is False
    assert all(model.cpu_calls >= 2 for model in pipeline.models.values())
    assert pipeline.last_run_stats['status'] == 'failed'
    assert pipeline.last_run_stats['stage_metrics'][-1]['status'] == 'failed'
    assert not hasattr(pipeline.models['shape_slat_decoder'], 'stream_device')


def test_1024_sampling_failure_offloads_all_models(monkeypatch, pipeline_cls):
    pipeline, _ = _synthetic_1024(monkeypatch, pipeline_cls)
    def fail(*args):
        raise RuntimeError('synthetic sampler failure')
    pipeline.sample_shape_slat = fail
    with pytest.raises(RuntimeError, match='synthetic sampler failure'):
        pipeline.run(Image.new('RGB', (8, 8)), vram_profile='8gb_1024')
    assert pipeline.low_vram is False
    assert all(model.cpu_calls >= 2 for model in pipeline.models.values())
    assert pipeline.last_run_stats['stage_metrics'][-1]['stage'] == 'shape_slat_512'
    assert pipeline.last_run_stats['stage_metrics'][-1]['status'] == 'failed'


def test_profiled_baseline_uses_original_cascade_and_keeps_history(monkeypatch, pipeline_cls):
    pipeline, sampled = _synthetic_1024(monkeypatch, pipeline_cls)
    calls = []
    def cascade(*args):
        calls.append(args)
        return types.SimpleNamespace(coords=torch.zeros(2, 4)), 1024
    pipeline.sample_shape_slat_cascade = cascade
    _, stats = pipeline.run(Image.new('RGB', (8, 8)), vram_profile='none',
                            profile_stages=True, return_metrics=True)
    assert len(calls) == 1
    assert not any('retain_history' in params for params in sampled)
    assert stats['vram_profile'] is None
    assert stats['low_vram'] is False
    assert not hasattr(pipeline, '_stage_profiler')


def test_profiled_baseline_restores_residency_after_optimized_run(monkeypatch, pipeline_cls):
    pipeline, _ = _synthetic_1024(monkeypatch, pipeline_cls)
    pipeline.run(Image.new('RGB', (8, 8)), vram_profile='8gb_1024')
    assert pipeline._models_offloaded_for_1024 is True
    pipeline.sample_shape_slat_cascade = lambda *args: (types.SimpleNamespace(coords=torch.zeros(2, 4)), 1024)
    pipeline.run(Image.new('RGB', (8, 8)), vram_profile='none', profile_stages=True)
    assert pipeline._models_offloaded_for_1024 is False
    assert all(model.to_calls for model in pipeline.models.values())


def test_decoder_streaming_real_cpu_pointwise_forward(monkeypatch, pipeline_cls):
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from trellis2.modules.sparse import SparseTensor
    # No sparse convolution is instantiated: this isolates the real projection,
    # streaming dispatcher, tensor replacement and precision on CPU.
    decoder = SparseUnetVaeDecoder(3, [4], 2, [0], ['unused'], [], [{}]).eval()
    x = SparseTensor(feats=torch.randn(9, 2), coords=torch.tensor([[0, i, 0, 0] for i in range(9)], dtype=torch.int32))
    events = []
    class PointwiseBlock(nn.Module):
        def forward(self, value):
            return value.replace(value.feats * 0.75)
        def cpu(self):
            events.append('block')
            return super().cpu()
    decoder.blocks[0].append(PointwiseBlock().eval())
    for name in ('from_latent', 'output_layer'):
        module = getattr(decoder, name)
        original_cpu = module.cpu
        def cpu(name=name, original=original_cpu):
            events.append(name)
            return original()
        monkeypatch.setattr(module, 'cpu', cpu)
    with torch.no_grad():
        expected = decoder(x)
        decoder.stream_device = 'cpu'
        decoder.output_chunk_size = 4
        actual = decoder(x)
    torch.testing.assert_close(actual.feats, expected.feats)
    assert torch.equal(actual.coords, expected.coords)
    assert events == ['from_latent', 'block', 'output_layer']


def test_streaming_decoder_block_failure_is_offloaded(pipeline_cls):
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    decoder = SparseUnetVaeDecoder(3, [4], 2, [0], ['unused'], [], [{}]).eval()
    decoder.stream_device = 'cpu'
    calls = []
    class FailingBlock(nn.Module):
        def forward(self, value):
            raise RuntimeError('block failed')
        def cpu(self):
            calls.append('cpu')
            return super().cpu()
    with pytest.raises(RuntimeError, match='block failed'):
        decoder._inference_call(FailingBlock(), torch.ones(1))
    assert calls == ['cpu']
