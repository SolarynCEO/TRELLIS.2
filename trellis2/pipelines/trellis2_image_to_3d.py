from typing import *
import os
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]
    vram_profiles: Dict[str, Dict[str, Any]] = {
        '8gb': {
            'force_low_vram': True,
            'default_pipeline_type': '512',
            'max_num_tokens': 32768,
            'empty_cache_between_stages': True,
            'autocast_dtype': 'fp16',
            'prefer_attention_backend': 'sdpa',
        },
    }

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
        vram_profile: Optional[str] = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.vram_profile = self._resolve_vram_profile(vram_profile)
        self.last_run_stats: Dict[str, Any] = {}
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        rembg_config = rembg.resolve_rembg_config(args['rembg_model'])
        pipeline.rembg_model = rembg.build_rembg_model(rembg_config)
        args['rembg_model'] = rembg_config
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.vram_profile = pipeline._resolve_vram_profile(args.get('vram_profile'))
        pipeline.last_run_stats = {}
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    @staticmethod
    def _is_truthy(value: Optional[str]) -> bool:
        if value is None:
            return False
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}

    def _resolve_vram_profile(self, requested: Optional[str]) -> Optional[str]:
        if requested is None:
            if self._is_truthy(os.environ.get("ZYREX_LOW_VRAM")):
                return '8gb'
            return None
        requested = requested.strip().lower()
        if requested in {'', '0', 'false', 'off', 'none', 'default'}:
            return None
        if requested not in self.vram_profiles:
            raise ValueError(f"Unsupported VRAM profile: {requested}. Available: {', '.join(sorted(self.vram_profiles))}")
        return requested

    def _resolve_autocast_dtype(self, profile_cfg: Dict[str, Any]) -> Optional[torch.dtype]:
        requested = os.environ.get("ZYREX_LOW_VRAM_DTYPE", profile_cfg.get('autocast_dtype'))
        if requested is None:
            return None
        requested = str(requested).strip().lower()
        if requested in {'', '0', 'false', 'off', 'none'}:
            return None
        if requested == 'fp16':
            return torch.float16
        if requested == 'bf16':
            return torch.bfloat16
        raise ValueError("ZYREX_LOW_VRAM_DTYPE must be one of: fp16, bf16, none")

    def _dtype_name(self, dtype: Optional[torch.dtype]) -> Optional[str]:
        if dtype is None:
            return None
        return {
            torch.float16: 'fp16',
            torch.bfloat16: 'bf16',
            torch.float32: 'fp32',
        }.get(dtype, str(dtype))

    def _is_cuda_runtime_active(self) -> bool:
        if not torch.cuda.is_available():
            return False
        device = self.device
        if isinstance(device, torch.device):
            return device.type == "cuda"
        return str(device).startswith("cuda")

    def _configure_attention_backend_for_profile(self, profile_cfg: Dict[str, Any]) -> None:
        preferred_backend = profile_cfg.get('prefer_attention_backend')
        if preferred_backend is None:
            return
        if os.environ.get('ATTN_BACKEND') or os.environ.get('SPARSE_ATTN_BACKEND'):
            return
        from ..modules.attention import config as dense_attn_config
        from ..modules.sparse import config as sparse_config
        dense_attn_config.BACKEND = preferred_backend
        sparse_config.ATTN = preferred_backend
        print(f"[LOW_VRAM] forcing ATTN_BACKEND={preferred_backend} for profile compatibility")

    def _run_stage(
        self,
        stage_name: str,
        stage_metrics: List[Dict[str, Any]],
        fn: Callable[[], Any],
        *,
        use_autocast: bool = False,
        autocast_dtype: Optional[torch.dtype] = None,
        clear_cuda_cache: bool = False,
    ) -> Any:
        use_cuda_metrics = self._is_cuda_runtime_active()
        if use_cuda_metrics:
            torch.cuda.reset_peak_memory_stats(self.device)
        start_time = time.perf_counter()
        context = (
            torch.autocast(device_type='cuda', dtype=autocast_dtype)
            if use_autocast and use_cuda_metrics and autocast_dtype is not None
            else nullcontext()
        )
        with context:
            out = fn()
        elapsed = time.perf_counter() - start_time
        metric = {
            'stage': stage_name,
            'time_s': elapsed,
            'peak_vram_allocated_bytes': int(torch.cuda.max_memory_allocated(self.device)) if use_cuda_metrics else None,
            'peak_vram_reserved_bytes': int(torch.cuda.max_memory_reserved(self.device)) if use_cuda_metrics else None,
        }
        stage_metrics.append(metric)
        if clear_cuda_cache and use_cuda_metrics:
            torch.cuda.empty_cache()
        return out

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        return rembg.preprocess_with_rembg(
            input,
            self.rembg_model,
            device=str(self.device),
            low_vram=self.low_vram,
            warning_context=f"{self.__class__.__name__}.preprocess_image",
        )
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device)
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        return_metrics: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: Optional[int] = None,
        vram_profile: Optional[str] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            return_metrics (bool): Whether to return per-stage timing/VRAM metrics.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            vram_profile (str): Optional VRAM profile preset. Supported: '8gb'.
        """
        run_start_time = time.perf_counter()
        profile_name = self._resolve_vram_profile(vram_profile if vram_profile is not None else self.vram_profile)
        profile_cfg = self.vram_profiles.get(profile_name, {})
        if profile_cfg:
            self._configure_attention_backend_for_profile(profile_cfg)
        effective_pipeline_type = pipeline_type or profile_cfg.get('default_pipeline_type') or self.default_pipeline_type
        effective_max_num_tokens = max_num_tokens if max_num_tokens is not None else profile_cfg.get('max_num_tokens', 49152)
        effective_low_vram = bool(self.low_vram or profile_cfg.get('force_low_vram', False))
        clear_cuda_cache = bool(profile_cfg.get('empty_cache_between_stages', False))
        autocast_dtype = self._resolve_autocast_dtype(profile_cfg)
        stage_metrics: List[Dict[str, Any]] = []

        prev_low_vram = self.low_vram
        self.low_vram = effective_low_vram
        # Check pipeline type
        if effective_pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif effective_pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif effective_pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif effective_pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {effective_pipeline_type}")
        try:
            if preprocess_image:
                image = self._run_stage(
                    "preprocess_image",
                    stage_metrics,
                    lambda: self.preprocess_image(image),
                    clear_cuda_cache=clear_cuda_cache,
                )
            torch.manual_seed(seed)
            cond_512 = self._run_stage(
                "condition_512",
                stage_metrics,
                lambda: self.get_cond([image], 512),
                use_autocast=True,
                autocast_dtype=autocast_dtype,
                clear_cuda_cache=clear_cuda_cache,
            )
            cond_1024 = (
                self._run_stage(
                    "condition_1024",
                    stage_metrics,
                    lambda: self.get_cond([image], 1024),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                if effective_pipeline_type != '512'
                else None
            )
            ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[effective_pipeline_type]
            coords = self._run_stage(
                "sparse_structure",
                stage_metrics,
                lambda: self.sample_sparse_structure(
                    cond_512,
                    ss_res,
                    num_samples,
                    sparse_structure_sampler_params,
                ),
                use_autocast=True,
                autocast_dtype=autocast_dtype,
                clear_cuda_cache=clear_cuda_cache,
            )
            if effective_pipeline_type == '512':
                shape_slat = self._run_stage(
                    "shape_slat_512",
                    stage_metrics,
                    lambda: self.sample_shape_slat(
                        cond_512,
                        self.models['shape_slat_flow_model_512'],
                        coords,
                        shape_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                tex_slat = self._run_stage(
                    "tex_slat_512",
                    stage_metrics,
                    lambda: self.sample_tex_slat(
                        cond_512,
                        self.models['tex_slat_flow_model_512'],
                        shape_slat,
                        tex_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                res = 512
            elif effective_pipeline_type == '1024':
                shape_slat = self._run_stage(
                    "shape_slat_1024",
                    stage_metrics,
                    lambda: self.sample_shape_slat(
                        cond_1024,
                        self.models['shape_slat_flow_model_1024'],
                        coords,
                        shape_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                tex_slat = self._run_stage(
                    "tex_slat_1024",
                    stage_metrics,
                    lambda: self.sample_tex_slat(
                        cond_1024,
                        self.models['tex_slat_flow_model_1024'],
                        shape_slat,
                        tex_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                res = 1024
            elif effective_pipeline_type == '1024_cascade':
                shape_slat, res = self._run_stage(
                    "shape_slat_1024_cascade",
                    stage_metrics,
                    lambda: self.sample_shape_slat_cascade(
                        cond_512,
                        cond_1024,
                        self.models['shape_slat_flow_model_512'],
                        self.models['shape_slat_flow_model_1024'],
                        512,
                        1024,
                        coords,
                        shape_slat_sampler_params,
                        effective_max_num_tokens,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                tex_slat = self._run_stage(
                    "tex_slat_1024",
                    stage_metrics,
                    lambda: self.sample_tex_slat(
                        cond_1024,
                        self.models['tex_slat_flow_model_1024'],
                        shape_slat,
                        tex_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
            elif effective_pipeline_type == '1536_cascade':
                shape_slat, res = self._run_stage(
                    "shape_slat_1536_cascade",
                    stage_metrics,
                    lambda: self.sample_shape_slat_cascade(
                        cond_512,
                        cond_1024,
                        self.models['shape_slat_flow_model_512'],
                        self.models['shape_slat_flow_model_1024'],
                        512,
                        1536,
                        coords,
                        shape_slat_sampler_params,
                        effective_max_num_tokens,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )
                tex_slat = self._run_stage(
                    "tex_slat_1024",
                    stage_metrics,
                    lambda: self.sample_tex_slat(
                        cond_1024,
                        self.models['tex_slat_flow_model_1024'],
                        shape_slat,
                        tex_slat_sampler_params,
                    ),
                    use_autocast=True,
                    autocast_dtype=autocast_dtype,
                    clear_cuda_cache=clear_cuda_cache,
                )

            out_mesh = self._run_stage(
                "decode_latent",
                stage_metrics,
                lambda: self.decode_latent(shape_slat, tex_slat, res),
                clear_cuda_cache=clear_cuda_cache,
            )
        finally:
            self.low_vram = prev_low_vram
        run_stats = {
            'vram_profile': profile_name,
            'device': str(self.device),
            'low_vram': effective_low_vram,
            'pipeline_type': effective_pipeline_type,
            'max_num_tokens': effective_max_num_tokens,
            'autocast_dtype': self._dtype_name(autocast_dtype),
            'stage_metrics': stage_metrics,
            'total_time_s': time.perf_counter() - run_start_time,
            'peak_vram_allocated_bytes': max(
                [m['peak_vram_allocated_bytes'] for m in stage_metrics if m['peak_vram_allocated_bytes'] is not None],
                default=None,
            ),
            'peak_vram_reserved_bytes': max(
                [m['peak_vram_reserved_bytes'] for m in stage_metrics if m['peak_vram_reserved_bytes'] is not None],
                default=None,
            ),
        }
        self.last_run_stats = run_stats

        if return_latent and return_metrics:
            return out_mesh, (shape_slat, tex_slat, res), run_stats
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        if return_metrics:
            return out_mesh, run_stats
        return out_mesh
