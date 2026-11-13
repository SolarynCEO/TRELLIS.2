from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parent / "o-voxel" / "o_voxel" / "zyrex_uv_baker.py"
_SPEC = importlib.util.spec_from_file_location("zyrex_uv_baker_impl", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load zyrex uv baker module from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

UVRasterization = _MODULE.UVRasterization
rasterize_uv_triangles = _MODULE.rasterize_uv_triangles
interpolate_vertex_attributes = _MODULE.interpolate_vertex_attributes
sample_sparse_volume_trilinear = _MODULE.sample_sparse_volume_trilinear
dilate_texture = _MODULE.dilate_texture
bake_tangent_space_normal_map = _MODULE.bake_tangent_space_normal_map
bake_voxel_pbr_maps = _MODULE.bake_voxel_pbr_maps

__all__ = [
    "UVRasterization",
    "rasterize_uv_triangles",
    "interpolate_vertex_attributes",
    "sample_sparse_volume_trilinear",
    "dilate_texture",
    "bake_tangent_space_normal_map",
    "bake_voxel_pbr_maps",
]
