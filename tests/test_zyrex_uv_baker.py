import importlib.util
import importlib
import io
import json
import math
import os
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_BACKEND_PATH = ROOT / "o-voxel" / "o_voxel" / "zyrex_baker_backend.py"
_BACKEND_SPEC = importlib.util.spec_from_file_location("zyrex_baker_backend_test_impl", _BACKEND_PATH)
if _BACKEND_SPEC is None or _BACKEND_SPEC.loader is None:
    raise RuntimeError(f"Failed to load backend module from {_BACKEND_PATH}")
_BACKEND_MODULE = importlib.util.module_from_spec(_BACKEND_SPEC)
sys.modules[_BACKEND_SPEC.name] = _BACKEND_MODULE
_BACKEND_SPEC.loader.exec_module(_BACKEND_MODULE)

_UV_PATH = ROOT / "o-voxel" / "o_voxel" / "zyrex_uv_baker.py"
_UV_SPEC = importlib.util.spec_from_file_location("zyrex_uv_baker_test_impl", _UV_PATH)
if _UV_SPEC is None or _UV_SPEC.loader is None:
    raise RuntimeError(f"Failed to load uv baker module from {_UV_PATH}")
_UV_MODULE = importlib.util.module_from_spec(_UV_SPEC)
sys.modules[_UV_SPEC.name] = _UV_MODULE
_UV_SPEC.loader.exec_module(_UV_MODULE)

DEFAULT_BAKER_BACKEND = _BACKEND_MODULE.DEFAULT_BAKER_BACKEND
resolve_baker_backend = _BACKEND_MODULE.resolve_baker_backend
bake_tangent_space_normal_map = _UV_MODULE.bake_tangent_space_normal_map
bake_voxel_pbr_maps = _UV_MODULE.bake_voxel_pbr_maps
dilate_texture = _UV_MODULE.dilate_texture
interpolate_vertex_attributes = _UV_MODULE.interpolate_vertex_attributes
rasterize_uv_triangles = _UV_MODULE.rasterize_uv_triangles

_COMPARE_PATH = ROOT / "trellis2" / "utils" / "zyrex_baker_compare.py"
_COMPARE_SPEC = importlib.util.spec_from_file_location("zyrex_baker_compare_test", _COMPARE_PATH)
if _COMPARE_SPEC is None or _COMPARE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load compare module from {_COMPARE_PATH}")
_COMPARE_MODULE = importlib.util.module_from_spec(_COMPARE_SPEC)
_COMPARE_SPEC.loader.exec_module(_COMPARE_MODULE)
run_real_asset_rebake = _COMPARE_MODULE.run_real_asset_rebake


def _plane_mesh(device: str = "cpu"):
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    uvs = vertices[:, :2].clone()
    return vertices, faces, uvs


def test_uv_rasterization_coverage_square():
    _, faces, uvs = _plane_mesh()
    raster = rasterize_uv_triangles(uvs, faces, texture_size=32)
    coverage = raster.mask.float().mean().item()
    assert coverage > 0.9


def test_barycentric_interpolation_matches_weights():
    vertices = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    values = torch.tensor([[1.0], [3.0], [5.0]], dtype=torch.float32)

    raster = rasterize_uv_triangles(vertices, faces, texture_size=8)
    interp = interpolate_vertex_attributes(values, faces, raster)
    assert interp.shape == (8, 8, 1)
    bary = raster.barycentric[raster.mask]
    expected = (bary * values[faces[0], 0]).sum(dim=-1)
    assert torch.allclose(interp[..., 0][raster.mask], expected, atol=1e-5)


def test_seam_dilation_fills_neighbors():
    texture = torch.zeros((5, 5, 1), dtype=torch.float32)
    texture[2, 2, 0] = 1.0
    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[2, 2] = True

    dilated = dilate_texture(texture, mask, iterations=1)
    assert dilated[2, 2, 0].item() == 1.0
    # 3x3 neighborhood around the center should be filled after one iteration.
    assert torch.all(dilated[1:4, 1:4, 0] > 0.0)


def test_tangent_space_normal_orientation_plane():
    vertices, faces, uvs = _plane_mesh()
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32)
    raster = rasterize_uv_triangles(uvs, faces, texture_size=32)

    normal_map = bake_tangent_space_normal_map(vertices, faces, uvs, normals, raster)
    sampled = normal_map[raster.mask]
    assert sampled.shape[0] > 0
    assert torch.all(sampled[:, 2] > 0.95)
    assert torch.all(torch.abs(sampled[:, 0] - 0.5) < 0.1)
    assert torch.all(torch.abs(sampled[:, 1] - 0.5) < 0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for nvdiffrast raster parity test")
def test_uv_rasterization_matches_nvdiffrast_mask_and_face_ids():
    dr = pytest.importorskip("nvdiffrast.torch")
    device = torch.device("cuda")
    # Build a true non-overlapping tiling over [0.05, 0.95]^2 with thin sliver rows/columns.
    xs = [0.05, 0.30, 0.60, 0.90, 0.95]  # thin right strip [0.90, 0.95]
    ys = [0.05, 0.12, 0.50, 0.88, 0.95]  # thin bottom/top strips
    verts = []
    for y in ys:
        for x in xs:
            verts.append([x, y])
    uvs = torch.tensor(verts, dtype=torch.float32, device=device)

    def vid(ix: int, iy: int) -> int:
        return iy * len(xs) + ix

    faces_list: list[list[int]] = []
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            v00 = vid(ix, iy)
            v10 = vid(ix + 1, iy)
            v01 = vid(ix, iy + 1)
            v11 = vid(ix + 1, iy + 1)
            if (ix + iy) % 2 == 0:
                faces_list.append([v00, v10, v11])
                faces_list.append([v00, v11, v01])
            else:
                faces_list.append([v00, v10, v01])
                faces_list.append([v10, v11, v01])
    faces = torch.tensor(faces_list, dtype=torch.long, device=device)
    texture_size = 256

    # Verify fixture property: no texel center is strictly inside more than one triangle.
    pts = (torch.arange(texture_size, device=device, dtype=torch.float32) + 0.5) / texture_size
    yy, xx = torch.meshgrid(pts, pts, indexing="ij")
    p = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [P, 2]
    tri = uvs[faces]  # [F, 3, 2]
    a = tri[:, 0][:, None, :]
    b = tri[:, 1][:, None, :]
    c = tri[:, 2][:, None, :]
    v0 = b - a
    v1 = c - a
    v2 = p[None, :, :] - a
    det = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    inv_det = torch.where(det.abs() > 1e-12, 1.0 / det, torch.zeros_like(det))
    w1 = (v2[..., 0] * v1[..., 1] - v2[..., 1] * v1[..., 0]) * inv_det
    w2 = (v0[..., 0] * v2[..., 1] - v0[..., 1] * v2[..., 0]) * inv_det
    w0 = 1.0 - w1 - w2
    strictly_inside = (w0 > 1e-7) & (w1 > 1e-7) & (w2 > 1e-7)
    cover_count = strictly_inside.sum(dim=0)
    assert int(cover_count.max().item()) <= 1

    raster = rasterize_uv_triangles(uvs, faces, texture_size)
    ctx = dr.RasterizeCudaContext(device=device)
    uvs_clip = torch.cat(
        [uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])],
        dim=-1,
    )[None]
    rast, _ = dr.rasterize(ctx, uvs_clip, faces.int(), resolution=[texture_size, texture_size])
    nvd_face = (rast[0, ..., 3].long() - 1).cpu()
    nvd_mask = nvd_face >= 0

    zyx_face = raster.face_indices.cpu()
    zyx_mask = raster.mask.cpu()

    def _summarize_diff(mask_diff: torch.Tensor, face_diff: torch.Tensor, limit: int = 20) -> str:
        coords = torch.nonzero(mask_diff | face_diff, as_tuple=False)
        if coords.numel() == 0:
            return "no diffs"
        y0 = int(coords[:, 0].min().item())
        y1 = int(coords[:, 0].max().item())
        x0 = int(coords[:, 1].min().item())
        x1 = int(coords[:, 1].max().item())
        face_coords = torch.nonzero(face_diff, as_tuple=False)
        entries = []
        for rc in face_coords[:limit]:
            rr = int(rc[0].item())
            cc = int(rc[1].item())
            entries.append((rr, cc, int(zyx_face[rr, cc].item()), int(nvd_face[rr, cc].item())))
        return f"bbox=({y0}:{y1},{x0}:{x1}) mismatches={entries}"

    mask_mismatch = torch.logical_xor(zyx_mask, nvd_mask)
    mismatch_count = int(mask_mismatch.sum().item())
    face_diff = (zyx_face != nvd_face) & (zyx_mask & nvd_mask)
    diff_summary = _summarize_diff(mask_mismatch, face_diff)
    # nvdiffrast uses internal fixed-point/subpixel rasterization; our plain-PyTorch
    # rasterizer targets parity but may differ on a handful of boundary texels.
    assert mismatch_count <= 4, (
        f"mask mismatch too large vs nvdiffrast: {mismatch_count} / {texture_size * texture_size} texels; {diff_summary}"
    )
    shared = nvd_mask & zyx_mask
    face_mismatch_count = int((zyx_face[shared] != nvd_face[shared]).sum().item())

    # Accept face-id differences only for genuine edge ties:
    #  - faces must share an edge (adjacent),
    #  - texel center must be within one 4-bit subpixel unit of that edge
    #    (distance <= 1/16 pixel).
    uvs_cpu = uvs.cpu().to(torch.float64)
    faces_cpu = faces.cpu().to(torch.long)
    face_coords = torch.nonzero(face_diff, as_tuple=False)
    unexplained = []
    explained = []
    distance_diag = []
    for rc in face_coords:
        rr = int(rc[0].item())
        cc = int(rc[1].item())
        fa = int(zyx_face[rr, cc].item())
        fb = int(nvd_face[rr, cc].item())
        va = set(int(v) for v in faces_cpu[fa].tolist())
        vb = set(int(v) for v in faces_cpu[fb].tolist())
        shared_verts = sorted(list(va.intersection(vb)))
        tie_ok = False
        distance_px = float("inf")
        if len(shared_verts) == 2:
            e0, e1 = shared_verts
            a = uvs_cpu[e0, :2]
            b = uvs_cpu[e1, :2]
            p = torch.tensor([(cc + 0.5) / texture_size, (rr + 0.5) / texture_size], dtype=torch.float64)
            edge = b - a
            edge_len2 = float(torch.dot(edge, edge).item())
            if edge_len2 > 0.0:
                t = float(torch.dot(p - a, edge).item() / edge_len2)
                t = min(1.0, max(0.0, t))
                closest = a + edge * t
                distance_uv = float(torch.linalg.norm(p - closest).item())
                distance_px = distance_uv * texture_size
            if distance_px <= (1.0 / 16.0 + 1e-9):
                tie_ok = True
        item = (rr, cc, fa, fb, float(distance_px))
        distance_diag.append(item)
        if tie_ok:
            explained.append(item)
        else:
            unexplained.append(item)

    diff_summary = _summarize_diff(mask_mismatch, face_diff)
    assert len(unexplained) <= 4, (
        f"face-id mismatch too large on shared coverage: {face_mismatch_count} total, "
        f"{len(explained)} explained-edge-ties, {len(unexplained)} unexplained; "
        f"distance_px={distance_diag[:40]}; "
        f"unexplained={unexplained[:20]}; {diff_summary}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for overlap permissive parity test")
def test_uv_rasterization_overlap_face_choice_is_valid():
    dr = pytest.importorskip("nvdiffrast.torch")
    device = torch.device("cuda")
    # Two coincident triangles with opposite winding overlap exactly in UV space.
    uvs = torch.tensor(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.50, 0.75],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 1],  # mirrored winding duplicate
        ],
        dtype=torch.long,
        device=device,
    )
    texture_size = 128

    raster = rasterize_uv_triangles(uvs, faces, texture_size)
    ctx = dr.RasterizeCudaContext(device=device)
    uvs_clip = torch.cat(
        [uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])],
        dim=-1,
    )[None]
    rast, _ = dr.rasterize(ctx, uvs_clip, faces.int(), resolution=[texture_size, texture_size])
    nvd_mask = (rast[0, ..., 3].long() - 1) >= 0
    assert torch.equal(raster.mask.cpu(), nvd_mask.cpu())

    chosen = raster.face_indices[raster.mask]
    valid = (chosen == 0) | (chosen == 1)
    assert bool(torch.all(valid).item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for triton raster parity test")
def test_uv_rasterization_triton_matches_torch(monkeypatch):
    pytest.importorskip("triton")
    device = torch.device("cuda")
    uvs = torch.tensor(
        [
            [0.03, 0.04],
            [0.97, 0.06],
            [0.95, 0.94],
            [0.06, 0.96],
            [0.15, 0.20],
            [0.85, 0.18],
            [0.55, 0.78],
            [0.10, 0.86],
            [0.91, 0.83],
        ],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [6, 5, 8],
            [7, 8, 6],
        ],
        dtype=torch.long,
        device=device,
    )
    tex = 256
    monkeypatch.setenv("ZYREX_RASTER_IMPL", "torch")
    raster_torch = rasterize_uv_triangles(uvs, faces, tex)
    monkeypatch.setenv("ZYREX_RASTER_IMPL", "triton")
    assert _UV_MODULE._resolve_raster_impl(device) == "triton"
    raster_triton = rasterize_uv_triangles(uvs, faces, tex)
    assert raster_triton.stats is not None, "Expected Triton rasterizer stats when Triton path is active."
    assert torch.equal(raster_torch.mask, raster_triton.mask)
    assert torch.equal(raster_torch.face_indices[raster_torch.mask], raster_triton.face_indices[raster_triton.mask])


def test_tangent_space_normal_has_detail_for_slanted_surface():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [1.0, 1.0, 0.4],
            [0.0, 1.0, 0.1],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    uvs = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    # Varying normals to simulate non-flat tangent-space detail.
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 0.98],
            [0.0, 0.3, 0.95],
            [-0.2, 0.1, 0.97],
        ],
        dtype=torch.float32,
    )
    normals = torch.nn.functional.normalize(normals, dim=-1)

    raster = rasterize_uv_triangles(uvs, faces, texture_size=64)
    normal_map = bake_tangent_space_normal_map(vertices, faces, uvs, normals, raster)
    sampled = normal_map[raster.mask]
    assert sampled.shape[0] > 0
    # x/y channels should vary and not stay flat at 0.5
    assert float(torch.std(sampled[:, 0])) > 0.01
    assert float(torch.std(sampled[:, 1])) > 0.01
    assert torch.all(sampled[:, 2] > 0.5)


def test_feature_flag_default_and_override(monkeypatch):
    monkeypatch.delenv("ZYREX_BAKER", raising=False)
    assert resolve_baker_backend() == DEFAULT_BAKER_BACKEND

    monkeypatch.setenv("ZYREX_BAKER", "zyrex")
    assert resolve_baker_backend() == "zyrex"

    monkeypatch.setenv("ZYREX_BAKER", "invalid-backend")
    assert resolve_baker_backend() == DEFAULT_BAKER_BACKEND


def test_raster_impl_default_is_torch(monkeypatch):
    monkeypatch.delenv("ZYREX_RASTER_IMPL", raising=False)
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert _UV_MODULE._resolve_raster_impl(torch.device("cpu")) == "torch"
    assert _UV_MODULE._resolve_raster_impl(torch.device("cuda")) == "torch"


def test_uv_rasterization_triton_interpret_matches_torch_cpu():
    pytest.importorskip("triton")
    script = f"""
import importlib.util
import json
import sys
import torch

uv_path = {str(_UV_PATH)!r}
spec = importlib.util.spec_from_file_location("zyrex_uv_baker_interpret_test_impl", uv_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load uv baker module from {{uv_path}}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

uvs = torch.tensor(
    [
        [0.03, 0.04],
        [0.97, 0.06],
        [0.95, 0.94],
        [0.06, 0.96],
        [0.15, 0.20],
        [0.85, 0.18],
        [0.55, 0.78],
        [0.10, 0.86],
        [0.91, 0.83],
    ],
    dtype=torch.float32,
    device="cpu",
)
faces = torch.tensor(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [6, 5, 8],
        [7, 8, 6],
    ],
    dtype=torch.long,
    device="cpu",
)
tex = 256

import os
os.environ["ZYREX_RASTER_IMPL"] = "torch"
raster_torch = mod.rasterize_uv_triangles(uvs, faces, tex)
os.environ["ZYREX_RASTER_IMPL"] = "triton"
raster_triton = mod.rasterize_uv_triangles(uvs, faces, tex)

payload = {{
    "resolved_impl": mod._resolve_raster_impl(torch.device("cpu")),
    "triton_stats_present": raster_triton.stats is not None,
    "mask_equal": bool(torch.equal(raster_torch.mask, raster_triton.mask)),
    "face_equal": bool(torch.equal(
        raster_torch.face_indices[raster_torch.mask],
        raster_triton.face_indices[raster_triton.mask]
    )),
}}
print(json.dumps(payload))
"""
    env = os.environ.copy()
    env["TRITON_INTERPRET"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["resolved_impl"] == "triton"
    assert payload["triton_stats_present"]
    assert payload["mask_equal"]
    assert payload["face_equal"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real export-path parity test")
def test_bake_glb_maps_from_prepared_zyrex_matches_nvdiffrast_tiny(monkeypatch):
    try:
        import nvdiffrast.torch as _  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing module: {exc.name}")
    try:
        flex_grid_sample = importlib.import_module("flex_gemm.ops.grid_sample")
    except ImportError as exc:
        missing = exc.name or "flex_gemm.ops.grid_sample"
        pytest.skip(f"Missing module: {missing}")
    if not hasattr(flex_grid_sample, "grid_sample_3d"):
        pytest.skip("Missing symbol: flex_gemm.ops.grid_sample.grid_sample_3d")

    # postprocess imports cumesh at module import time; this test path does not
    # execute any cumesh calls, so provide a lightweight stub when unavailable.
    try:
        importlib.import_module("cumesh")
    except ImportError:
        import types

        monkeypatch.setitem(sys.modules, "cumesh", types.ModuleType("cumesh"))

    post_module = importlib.import_module("o_voxel.postprocess")

    device = torch.device("cuda")
    out_vertices = torch.tensor(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    out_faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    out_uvs = torch.tensor(
        [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95]],
        dtype=torch.float32,
        device=device,
    )
    out_normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32, device=device)
    source_vertices = out_vertices.clone()
    source_faces = out_faces.clone()

    class _DummyBVH:
        @staticmethod
        def unsigned_distance(query_positions: torch.Tensor, return_uvw: bool = True):
            x = query_positions[:, 0]
            y = query_positions[:, 1]
            uvw = torch.stack([1.0 - x - y, x, y], dim=-1)
            face_id = torch.zeros((query_positions.shape[0],), dtype=torch.int64, device=query_positions.device)
            dist = torch.zeros((query_positions.shape[0],), dtype=query_positions.dtype, device=query_positions.device)
            return dist, face_id, uvw

    prepared = {
        "out_vertices": out_vertices,
        "out_faces": out_faces,
        "out_uvs": out_uvs,
        "out_normals": out_normals,
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "bvh": _DummyBVH(),
    }

    grid_res = 8
    gx, gy, gz = torch.meshgrid(
        torch.arange(grid_res, dtype=torch.int32, device=device),
        torch.arange(grid_res, dtype=torch.int32, device=device),
        torch.arange(grid_res, dtype=torch.int32, device=device),
        indexing="ij",
    )
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    xyz = coords.float() / float(grid_res - 1)
    attr_volume = torch.cat(
        [
            xyz,
            (0.2 + 0.5 * xyz[:, 0:1]),
            (0.3 + 0.4 * xyz[:, 1:2]),
            torch.ones_like(xyz[:, 0:1]),
        ],
        dim=-1,
    ).to(torch.float32)
    attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    grid_size = torch.tensor([grid_res, grid_res, grid_res], dtype=torch.int32, device=device)
    voxel_size = torch.tensor([1.0 / (grid_res - 1)] * 3, dtype=torch.float32, device=device)
    tex = 64

    baked_nvd = post_module.bake_glb_maps_from_prepared(
        prepared=prepared,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        texture_size=tex,
        backend="nvdiffrast",
        collect_debug=True,
    )
    baked_zyx = post_module.bake_glb_maps_from_prepared(
        prepared=prepared,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        texture_size=tex,
        backend="zyrex",
        collect_debug=True,
    )

    mask_nvd = baked_nvd["mask_raw"].astype(bool)
    mask_zyx = baked_zyx["mask_raw"].astype(bool)
    xor_ratio = float(np.logical_xor(mask_nvd, mask_zyx).mean())
    assert xor_ratio <= 0.01
    overlap = mask_nvd & mask_zyx
    assert overlap.any()

    for key, min_psnr in (
        ("base_color", 35.0),
        ("metallic", 35.0),
        ("roughness", 35.0),
        ("alpha", 60.0),
    ):
        ref = baked_nvd["maps_filled"][key].astype(np.float32) / 255.0
        pred = baked_zyx["maps_filled"][key].astype(np.float32) / 255.0
        psnr_val = _psnr(pred[overlap].reshape(-1), ref[overlap].reshape(-1))
        assert psnr_val >= min_psnr, f"{key} PSNR too low: {psnr_val}"


def _analytic_attr(pos: torch.Tensor) -> dict[str, torch.Tensor]:
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    base_color = torch.stack(
        [
            0.5 + 0.5 * torch.sin(20.0 * x + 5.0 * z),
            0.5 + 0.5 * torch.cos(18.0 * y - 3.0 * z),
            0.5 + 0.5 * torch.sin(33.0 * (x + y) + 2.0 * z),
        ],
        dim=-1,
    )
    metallic = torch.clamp(0.15 + 0.75 * x + 0.05 * torch.sin(11.0 * y), 0.0, 1.0)[..., None]
    roughness = torch.clamp(0.1 + 0.8 * y + 0.05 * torch.cos(7.0 * x), 0.0, 1.0)[..., None]
    alpha = torch.ones_like(metallic)
    return {
        "base_color": torch.clamp(base_color, 0.0, 1.0),
        "metallic": metallic,
        "roughness": roughness,
        "alpha": alpha,
    }


def _make_complex_mesh_case(texture_size: int = 128):
    vertices = []
    uvs = []
    faces = []

    def add_face(v0, v1, v2, uv0, uv1, uv2):
        start = len(vertices)
        vertices.extend([v0, v1, v2])
        uvs.extend([uv0, uv1, uv2])
        faces.append([start, start + 1, start + 2])
        return len(faces) - 1

    # Island A (dense UV)
    add_face((0.05, 0.05, 0.20), (0.45, 0.05, 0.20), (0.45, 0.45, 0.20), (0.05, 0.05), (0.45, 0.05), (0.45, 0.45))
    add_face((0.05, 0.05, 0.20), (0.45, 0.45, 0.20), (0.05, 0.45, 0.20), (0.05, 0.05), (0.45, 0.45), (0.05, 0.45))

    # Island B with UV seam gap (same geometric region split in UV)
    add_face((0.45, 0.05, 0.25), (0.85, 0.05, 0.25), (0.85, 0.45, 0.25), (0.55, 0.05), (0.95, 0.05), (0.95, 0.45))
    add_face((0.45, 0.05, 0.25), (0.85, 0.45, 0.25), (0.45, 0.45, 0.25), (0.55, 0.05), (0.95, 0.45), (0.55, 0.45))

    # Overlapping + mirrored UV island.
    overlap_face_id = add_face(
        (0.15, 0.55, 0.10), (0.40, 0.55, 0.10), (0.275, 0.82, 0.10),
        (0.47, 0.18), (0.53, 0.18), (0.50, 0.34),
    )
    add_face(
        (0.15, 0.55, 0.85), (0.275, 0.82, 0.85), (0.40, 0.55, 0.85),
        (0.47, 0.18), (0.50, 0.34), (0.53, 0.18),
    )

    # Thin geometry in UV and 3D.
    add_face((0.10, 0.90, 0.40), (0.90, 0.92, 0.40), (0.10, 0.905, 0.40), (0.08, 0.88), (0.92, 0.92), (0.08, 0.90))

    # Non-uniform texel density: large 3D area collapsed to tiny UV island.
    add_face((0.05, 0.55, 0.60), (0.95, 0.60, 0.60), (0.50, 0.98, 0.60), (0.72, 0.72), (0.82, 0.72), (0.77, 0.80))

    vertices_t = torch.tensor(vertices, dtype=torch.float32)
    faces_t = torch.tensor(faces, dtype=torch.long)
    uvs_t = torch.tensor(uvs, dtype=torch.float32)
    normals_t = torch.tensor([[0.0, 0.0, 1.0]] * len(vertices), dtype=torch.float32)

    grid_res = 72
    gx, gy, gz = torch.meshgrid(
        torch.arange(grid_res, dtype=torch.long),
        torch.arange(grid_res, dtype=torch.long),
        torch.arange(grid_res, dtype=torch.long),
        indexing="ij",
    )
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    xyz = coords.float() / float(grid_res - 1)
    attrs = _analytic_attr(xyz)
    attr_volume = torch.cat([attrs["base_color"], attrs["metallic"], attrs["roughness"], attrs["alpha"]], dim=-1)
    attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }

    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    grid_size = torch.tensor([grid_res, grid_res, grid_res], dtype=torch.long)
    voxel_size = (aabb[1] - aabb[0]) / (grid_size.float() - 1.0)

    return {
        "vertices": vertices_t,
        "faces": faces_t,
        "uvs": uvs_t,
        "normals": normals_t,
        "coords": coords,
        "attr_volume": attr_volume,
        "attr_layout": attr_layout,
        "aabb": aabb,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "texture_size": texture_size,
        "overlap_face_id": overlap_face_id,
    }


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def test_complex_uv_cases_cover_seams_overlap_thin_density_and_frequency():
    case = _make_complex_mesh_case(texture_size=128)
    baked_raw = bake_voxel_pbr_maps(
        vertices=case["vertices"],
        faces=case["faces"],
        uvs=case["uvs"],
        vertex_normals=case["normals"],
        attr_volume=case["attr_volume"],
        coords=case["coords"],
        attr_layout=case["attr_layout"],
        aabb=case["aabb"],
        voxel_size=case["voxel_size"],
        grid_size=case["grid_size"],
        texture_size=case["texture_size"],
        seam_padding_iterations=0,
    )
    baked = bake_voxel_pbr_maps(
        vertices=case["vertices"],
        faces=case["faces"],
        uvs=case["uvs"],
        vertex_normals=case["normals"],
        attr_volume=case["attr_volume"],
        coords=case["coords"],
        attr_layout=case["attr_layout"],
        aabb=case["aabb"],
        voxel_size=case["voxel_size"],
        grid_size=case["grid_size"],
        texture_size=case["texture_size"],
        seam_padding_iterations=6,
    )

    mask = baked["mask"]
    coverage = float(mask.float().mean())
    assert 0.05 < coverage < 0.95

    for key in ("base_color", "metallic", "roughness", "alpha", "normal"):
        assert torch.isfinite(baked[key]).all(), f"{key} has NaN/Inf values"

    # Ensure padding affects unknown texels but preserves known texels.
    mask_f = mask.float()[None, None]
    neighbors = F.conv2d(mask_f, torch.ones((1, 1, 3, 3)), padding=1)[0, 0] > 0
    boundary_unknown = neighbors & (~mask)
    assert boundary_unknown.any()
    delta_boundary = torch.mean(torch.abs(baked["base_color"][boundary_unknown] - baked_raw["base_color"][boundary_unknown]))
    assert float(delta_boundary) > 1e-4
    assert torch.allclose(baked["base_color"][mask], baked_raw["base_color"][mask], atol=1e-6)

    # Behavior for overlapping mirrored UVs: first overlapping face should win.
    raster = rasterize_uv_triangles(case["uvs"], case["faces"], case["texture_size"])
    overlap_px_x = int(0.50 * (case["texture_size"] - 1))
    overlap_px_y = int(0.24 * (case["texture_size"] - 1))
    assert int(raster.face_indices[overlap_px_y, overlap_px_x]) == case["overlap_face_id"]

    # High-frequency behavior over covered texels should remain numerically stable.
    pos_map = interpolate_vertex_attributes(case["vertices"], case["faces"], raster)
    expected = _analytic_attr(pos_map)
    pred = baked["base_color"][mask].cpu().numpy()
    tgt = expected["base_color"][mask].cpu().numpy()
    assert _psnr(pred, tgt) > 30.0

    # No NaNs in thin geometry region and normals remain forward-facing in tangent space.
    normal_samples = baked["normal"][mask]
    assert torch.all(normal_samples[:, 2] > 0.70)


def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    pad = (-len(data)) % 4
    if pad == 0:
        return data
    return data + pad_byte * pad


def _write_interleaved_accessor_glb(path: Path) -> None:
    positions = [
        (-0.8, -0.8, 0.0),
        (0.8, -0.8, 0.0),
        (0.8, 0.8, 0.0),
        (-0.8, 0.8, 0.0),
    ]
    uvs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    indices = [0, 1, 2, 0, 2, 3]

    interleaved = bytearray()
    for p, uv in zip(positions, uvs):
        interleaved += struct.pack("<3f2f", *p, *uv)
    interleaved_bytes = bytes(interleaved)

    idx_bytes = struct.pack("<6H", *indices)

    img = Image.new("RGBA", (16, 16), color=(0, 0, 0, 255))
    px = img.load()
    for y in range(16):
        for x in range(16):
            if (x // 2 + y // 2) % 2 == 0:
                px[x, y] = (240, 40, 40, 255)
            else:
                px[x, y] = (40, 220, 180, 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b0 = interleaved_bytes
    b0_p = _pad4(b0)
    o1 = len(b0_p)
    b1 = idx_bytes
    b1_p = _pad4(b1)
    o2 = o1 + len(b1_p)
    b2 = png_bytes
    b2_p = _pad4(b2)
    bin_blob = b0_p + b1_p + b2_p

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(b0), "byteStride": 20, "target": 34962},
            {"buffer": 0, "byteOffset": o1, "byteLength": len(b1), "target": 34963},
            {"buffer": 0, "byteOffset": o2, "byteLength": len(b2)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-0.8, -0.8, 0.0],
                "max": [0.8, 0.8, 0.0],
            },
            {
                "bufferView": 0,
                "byteOffset": 12,
                "componentType": 5126,
                "count": 4,
                "type": "VEC2",
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5123,
                "count": 6,
                "type": "SCALAR",
            },
        ],
        "images": [{"bufferView": 2, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad_byte=b" ")
    bin_chunk = _pad4(bin_blob)
    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    glb = bytearray()
    glb += struct.pack("<4sII", b"glTF", 2, total_len)
    glb += struct.pack("<I4s", len(json_chunk), b"JSON")
    glb += json_chunk
    glb += struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
    glb += bin_chunk
    path.write_bytes(glb)


def test_interleaved_accessor_glb_real_asset_path(tmp_path: Path):
    glb_path = tmp_path / "interleaved_checker.glb"
    _write_interleaved_accessor_glb(glb_path)

    result = run_real_asset_rebake(glb_path, render_views=True, render_resolution=96)
    assert result["coverage"]["footprint_ratio"] > 0.10

    corrected = result["map_metrics"]["corrected_covered_texels"]["base_color_rgba"]
    legacy = result["map_metrics"]["legacy_full_texture"]["base_color_rgba"]
    assert math.isfinite(corrected["psnr"]) and math.isfinite(corrected["ssim"])
    assert math.isfinite(legacy["psnr"]) and math.isfinite(legacy["ssim"])
    assert corrected["psnr"] > 5.0

    rv = result["rendered_view_metrics"]
    assert rv["status"] == "measured"
    assert rv["mean_psnr"] > 6.0
    assert math.isfinite(rv["mean_ssim"])
