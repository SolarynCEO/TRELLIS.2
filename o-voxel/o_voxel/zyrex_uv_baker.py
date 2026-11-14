from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class UVRasterization:
    mask: torch.Tensor
    face_indices: torch.Tensor
    barycentric: torch.Tensor


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _to_texture_hw(texture_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(texture_size, int):
        return texture_size, texture_size
    if len(texture_size) != 2:
        raise ValueError(f"texture_size must be an int or length-2 sequence, got {texture_size}")
    return int(texture_size[0]), int(texture_size[1])


def rasterize_uv_triangles(
    uvs: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int | Sequence[int],
    eps: float = 1e-6,
) -> UVRasterization:
    height, width = _to_texture_hw(texture_size)
    device = uvs.device
    dtype = uvs.dtype

    face_indices = torch.full((height, width), -1, dtype=torch.long, device=device)
    barycentric = torch.zeros((height, width, 3), dtype=dtype, device=device)

    uvs_px = uvs[:, :2].clone()
    uvs_px[:, 0] = uvs_px[:, 0] * (width - 1)
    uvs_px[:, 1] = uvs_px[:, 1] * (height - 1)

    for face_id in range(faces.shape[0]):
        face = faces[face_id]
        tri = uvs_px[face]  # [3, 2]

        min_x = max(int(torch.floor(tri[:, 0].min()).item()), 0)
        max_x = min(int(torch.ceil(tri[:, 0].max()).item()), width - 1)
        min_y = max(int(torch.floor(tri[:, 1].min()).item()), 0)
        max_y = min(int(torch.ceil(tri[:, 1].max()).item()), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        x = torch.arange(min_x, max_x + 1, device=device, dtype=dtype) + 0.5
        y = torch.arange(min_y, max_y + 1, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        ax, ay = tri[0, 0], tri[0, 1]
        bx, by = tri[1, 0], tri[1, 1]
        cx, cy = tri[2, 0], tri[2, 1]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if torch.abs(denom) < eps:
            continue

        w0 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        w1 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue

        region_faces = face_indices[min_y : max_y + 1, min_x : max_x + 1]
        region_bary = barycentric[min_y : max_y + 1, min_x : max_x + 1]
        write_mask = inside & (region_faces < 0)
        if not write_mask.any():
            continue

        region_faces[write_mask] = face_id
        region_bary[..., 0] = torch.where(write_mask, w0, region_bary[..., 0])
        region_bary[..., 1] = torch.where(write_mask, w1, region_bary[..., 1])
        region_bary[..., 2] = torch.where(write_mask, w2, region_bary[..., 2])

    mask = face_indices >= 0
    return UVRasterization(mask=mask, face_indices=face_indices, barycentric=barycentric)


def interpolate_vertex_attributes(
    values: torch.Tensor,
    faces: torch.Tensor,
    raster: UVRasterization,
) -> torch.Tensor:
    if values.dim() == 1:
        values = values[:, None]

    out = torch.zeros(
        (*raster.mask.shape, values.shape[-1]),
        dtype=values.dtype,
        device=values.device,
    )
    if not raster.mask.any():
        return out

    face_ids = raster.face_indices[raster.mask].long()
    tri_values = values[faces[face_ids]]  # [N, 3, C]
    bary = raster.barycentric[raster.mask].to(values.dtype)  # [N, 3]
    out[raster.mask] = torch.sum(tri_values * bary.unsqueeze(-1), dim=1)
    return out


def _normalize_index(values: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(values)
    return values / (size - 1) * 2.0 - 1.0


def sample_sparse_volume_trilinear(
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    grid_size: torch.Tensor | Sequence[int],
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    query_positions: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    if query_positions.shape[0] == 0:
        return torch.zeros((0, attr_volume.shape[-1]), dtype=attr_volume.dtype, device=attr_volume.device)

    device = attr_volume.device
    dtype = attr_volume.dtype
    grid_size_t = torch.as_tensor(grid_size, dtype=torch.long, device=device)
    coords_t = coords.to(device=device, dtype=torch.long)
    attrs_t = attr_volume.to(device=device, dtype=dtype)

    if chunk_size is None:
        chunk_size = 131072 if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")) else 524288
    chunk_size = max(int(chunk_size), 1)

    gx = int(grid_size_t[0].item())
    gy = int(grid_size_t[1].item())
    gz = int(grid_size_t[2].item())
    stride_y = gz
    stride_x = gy * gz

    sparse_keys = coords_t[:, 0] * stride_x + coords_t[:, 1] * stride_y + coords_t[:, 2]
    sort_idx = torch.argsort(sparse_keys)
    sparse_keys_sorted = sparse_keys[sort_idx]
    sparse_attrs_sorted = attrs_t[sort_idx]

    query = ((query_positions.to(device=device, dtype=dtype) - aabb[0].to(device=device, dtype=dtype)) / voxel_size.to(device=device, dtype=dtype))
    query = torch.clamp(query, min=0.0)
    query[:, 0] = torch.clamp(query[:, 0], max=float(gx - 1))
    query[:, 1] = torch.clamp(query[:, 1], max=float(gy - 1))
    query[:, 2] = torch.clamp(query[:, 2], max=float(gz - 1))

    out = torch.zeros((query.shape[0], attrs_t.shape[-1]), dtype=dtype, device=device)

    def corner_values(keys_1d: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(sparse_keys_sorted, keys_1d)
        in_bounds = idx < sparse_keys_sorted.shape[0]
        valid = torch.zeros_like(in_bounds)
        if in_bounds.any():
            valid[in_bounds] = sparse_keys_sorted[idx[in_bounds]] == keys_1d[in_bounds]
        values = torch.zeros((keys_1d.shape[0], attrs_t.shape[-1]), dtype=dtype, device=device)
        if valid.any():
            values[valid] = sparse_attrs_sorted[idx[valid]]
        return values

    for start in range(0, query.shape[0], chunk_size):
        q = query[start : start + chunk_size]

        x0 = torch.floor(q[:, 0]).long()
        y0 = torch.floor(q[:, 1]).long()
        z0 = torch.floor(q[:, 2]).long()
        x1 = torch.clamp(x0 + 1, max=gx - 1)
        y1 = torch.clamp(y0 + 1, max=gy - 1)
        z1 = torch.clamp(z0 + 1, max=gz - 1)

        wx = (q[:, 0] - x0.to(dtype)).unsqueeze(-1)
        wy = (q[:, 1] - y0.to(dtype)).unsqueeze(-1)
        wz = (q[:, 2] - z0.to(dtype)).unsqueeze(-1)
        ix0 = 1.0 - wx
        iy0 = 1.0 - wy
        iz0 = 1.0 - wz

        corners = (
            (x0, y0, z0, ix0 * iy0 * iz0),
            (x0, y0, z1, ix0 * iy0 * wz),
            (x0, y1, z0, ix0 * wy * iz0),
            (x0, y1, z1, ix0 * wy * wz),
            (x1, y0, z0, wx * iy0 * iz0),
            (x1, y0, z1, wx * iy0 * wz),
            (x1, y1, z0, wx * wy * iz0),
            (x1, y1, z1, wx * wy * wz),
        )

        chunk_out = torch.zeros((q.shape[0], attrs_t.shape[-1]), dtype=dtype, device=device)
        for cx, cy, cz, weight in corners:
            keys = cx * stride_x + cy * stride_y + cz
            chunk_out = chunk_out + corner_values(keys) * weight

        out[start : start + q.shape[0]] = chunk_out

    return out.contiguous()


def dilate_texture(
    texture: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int = 8,
) -> torch.Tensor:
    if texture.dim() != 3:
        raise ValueError(f"texture must be [H, W, C], got {texture.shape}")
    if valid_mask.dim() != 2:
        raise ValueError(f"valid_mask must be [H, W], got {valid_mask.shape}")

    if not valid_mask.any():
        return texture

    work_dtype = torch.float32 if texture.dtype in (torch.float16, torch.bfloat16) else texture.dtype
    filled = texture.to(work_dtype).clone()
    known = valid_mask.bool().clone()
    kernel = torch.ones((1, 1, 3, 3), dtype=filled.dtype, device=filled.device)

    for _ in range(iterations):
        if known.all():
            break

        known_f = known.float().unsqueeze(0).unsqueeze(0)
        known_neighbors = F.conv2d(known_f, kernel, padding=1).squeeze(0).squeeze(0)
        expandable = (known_neighbors > 0) & (~known)
        if not expandable.any():
            break

        tex_chw = filled.permute(2, 0, 1).unsqueeze(0)
        sum_neighbors = F.conv2d(
            tex_chw * known_f,
            kernel.expand(filled.shape[-1], 1, 3, 3),
            padding=1,
            groups=filled.shape[-1],
        )
        mean_neighbors = (sum_neighbors / known_neighbors.clamp_min(1.0).unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1, 2, 0)

        filled[expandable] = mean_neighbors[expandable]
        known[expandable] = True

    return filled.to(texture.dtype)


def bake_tangent_space_normal_map(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    vertex_normals: torch.Tensor,
    raster: UVRasterization,
    eps: float = 1e-8,
) -> torch.Tensor:
    normal_map = torch.zeros((*raster.mask.shape, 3), dtype=vertices.dtype, device=vertices.device)
    normal_map[..., 2] = 1.0
    if not raster.mask.any():
        return normal_map

    tri_pos = vertices[faces]  # [F, 3, 3]
    tri_uv = uvs[faces][:, :, :2]  # [F, 3, 2]
    e1 = tri_pos[:, 1] - tri_pos[:, 0]
    e2 = tri_pos[:, 2] - tri_pos[:, 0]
    duv1 = tri_uv[:, 1] - tri_uv[:, 0]
    duv2 = tri_uv[:, 2] - tri_uv[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv1[:, 1] * duv2[:, 0]
    safe_det = torch.where(det.abs() < eps, torch.ones_like(det), det)

    tangent = (e1 * duv2[:, 1:2] - e2 * duv1[:, 1:2]) / safe_det[:, None]
    bitangent = (e2 * duv1[:, 0:1] - e1 * duv2[:, 0:1]) / safe_det[:, None]

    fallback_tangent = torch.tensor([1.0, 0.0, 0.0], dtype=vertices.dtype, device=vertices.device).expand_as(tangent)
    fallback_bitangent = torch.tensor([0.0, 1.0, 0.0], dtype=vertices.dtype, device=vertices.device).expand_as(bitangent)
    valid_basis = det.abs() >= eps
    tangent = torch.where(valid_basis[:, None], tangent, fallback_tangent)
    bitangent = torch.where(valid_basis[:, None], bitangent, fallback_bitangent)

    interp_normals = interpolate_vertex_attributes(vertex_normals, faces, raster)
    n = F.normalize(interp_normals[raster.mask], dim=-1)
    face_ids = raster.face_indices[raster.mask]
    t = F.normalize(tangent[face_ids], dim=-1)
    t = F.normalize(t - (t * n).sum(dim=-1, keepdim=True) * n, dim=-1)
    b = F.normalize(torch.cross(n, t, dim=-1), dim=-1)

    normal_ts = torch.stack(
        [
            (n * t).sum(dim=-1),
            (n * b).sum(dim=-1),
            (n * n).sum(dim=-1),
        ],
        dim=-1,
    )
    normal_map[raster.mask] = normal_ts
    return torch.clamp(normal_map * 0.5 + 0.5, 0.0, 1.0)


def bake_voxel_pbr_maps(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    vertex_normals: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor | Sequence[int],
    texture_size: int,
    seam_padding_iterations: int = 8,
    position_mapper: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    sample_chunk_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    raster = rasterize_uv_triangles(uvs, faces, texture_size)
    positions = interpolate_vertex_attributes(vertices, faces, raster)
    query_positions = positions[raster.mask]
    if position_mapper is not None:
        query_positions = position_mapper(query_positions)

    sampled_attrs = sample_sparse_volume_trilinear(
        attr_volume=attr_volume,
        coords=coords,
        grid_size=grid_size,
        aabb=aabb,
        voxel_size=voxel_size,
        query_positions=query_positions,
        chunk_size=sample_chunk_size,
    )

    attrs_tex = torch.zeros((texture_size, texture_size, attr_volume.shape[-1]), dtype=attr_volume.dtype, device=attr_volume.device)
    attrs_tex[raster.mask] = sampled_attrs

    base_color = attrs_tex[..., attr_layout["base_color"]]
    metallic = attrs_tex[..., attr_layout["metallic"]]
    roughness = attrs_tex[..., attr_layout["roughness"]]
    alpha = attrs_tex[..., attr_layout["alpha"]]
    normal = bake_tangent_space_normal_map(vertices, faces, uvs, vertex_normals, raster)

    base_color = dilate_texture(base_color, raster.mask, seam_padding_iterations)
    metallic = dilate_texture(metallic, raster.mask, seam_padding_iterations)
    roughness = dilate_texture(roughness, raster.mask, seam_padding_iterations)
    alpha = dilate_texture(alpha, raster.mask, seam_padding_iterations)
    normal = dilate_texture(normal, raster.mask, seam_padding_iterations)

    return {
        "mask": raster.mask,
        "base_color": torch.clamp(base_color, 0.0, 1.0),
        "metallic": torch.clamp(metallic, 0.0, 1.0),
        "roughness": torch.clamp(roughness, 0.0, 1.0),
        "alpha": torch.clamp(alpha, 0.0, 1.0),
        "normal": torch.clamp(normal, 0.0, 1.0),
    }
