import torch

from zyrex_baker_backend import DEFAULT_BAKER_BACKEND, resolve_baker_backend
from zyrex_uv_baker import (
    bake_tangent_space_normal_map,
    dilate_texture,
    interpolate_vertex_attributes,
    rasterize_uv_triangles,
)


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
    bary = raster.barycentric[raster.mask]
    expected = (bary * values[faces[0], 0]).sum(dim=-1)
    assert torch.allclose(interp[raster.mask, 0], expected, atol=1e-5)


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


def test_feature_flag_default_and_override(monkeypatch):
    monkeypatch.delenv("ZYREX_BAKER", raising=False)
    assert resolve_baker_backend() == DEFAULT_BAKER_BACKEND

    monkeypatch.setenv("ZYREX_BAKER", "zyrex")
    assert resolve_baker_backend() == "zyrex"

    monkeypatch.setenv("ZYREX_BAKER", "invalid-backend")
    assert resolve_baker_backend() == DEFAULT_BAKER_BACKEND
