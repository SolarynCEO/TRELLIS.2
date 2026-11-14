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
    face_chunk_size: Optional[int] = None,
    max_candidates_per_chunk: Optional[int] = None,
) -> UVRasterization:
    height, width = _to_texture_hw(texture_size)
    device = uvs.device
    dtype = uvs.dtype

    if face_chunk_size is None:
        face_chunk_size = 2048 if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")) else 4096
    if max_candidates_per_chunk is None:
        max_candidates_per_chunk = 1_500_000 if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")) else 4_000_000
    face_chunk_size = max(int(face_chunk_size), 1)
    max_candidates_per_chunk = max(int(max_candidates_per_chunk), 1)
    subpixel_bits = 8
    subpixel_scale = 1 << subpixel_bits
    area_eps = max(1, int(round(eps * (subpixel_scale**2))))

    faces_long = faces.long()
    uvs_px = uvs[:, :2].to(dtype=torch.float32).clone()
    # Match rasterization pixel-center convention used by clip-space backends:
    # texel center i maps to UV (i + 0.5) / size.
    uvs_px[:, 0] = uvs_px[:, 0] * width - 0.5
    uvs_px[:, 1] = uvs_px[:, 1] * height - 0.5
    tris = uvs_px[faces_long]  # [F, 3, 2]
    tris_fp = torch.round(tris * subpixel_scale).to(torch.int64)

    num_pixels = height * width
    best_face = torch.full((num_pixels,), -1, dtype=torch.long, device=device)
    best_bary = torch.zeros((num_pixels, 3), dtype=dtype, device=device)

    total_faces = tris.shape[0]
    start = 0
    while start < total_faces:
        tentative_end = min(start + face_chunk_size, total_faces)
        while True:
            tri_chunk_fp = tris_fp[start:tentative_end]
            min_x = torch.ceil(torch.min(tri_chunk_fp[..., 0], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, width - 1)
            max_x = torch.floor(torch.max(tri_chunk_fp[..., 0], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, width - 1)
            min_y = torch.ceil(torch.min(tri_chunk_fp[..., 1], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, height - 1)
            max_y = torch.floor(torch.max(tri_chunk_fp[..., 1], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, height - 1)
            widths = max_x - min_x + 1
            heights = max_y - min_y + 1
            valid_bbox = (widths > 0) & (heights > 0)
            if not valid_bbox.any():
                break
            areas = widths[valid_bbox] * heights[valid_bbox]
            num_candidates = int(areas.sum().item())
            if num_candidates <= max_candidates_per_chunk or tentative_end - start <= 1:
                break
            tentative_end = start + max(1, (tentative_end - start) // 2)

        tri_chunk_fp = tris_fp[start:tentative_end]
        min_x = torch.ceil(torch.min(tri_chunk_fp[..., 0], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, width - 1)
        max_x = torch.floor(torch.max(tri_chunk_fp[..., 0], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, width - 1)
        min_y = torch.ceil(torch.min(tri_chunk_fp[..., 1], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, height - 1)
        max_y = torch.floor(torch.max(tri_chunk_fp[..., 1], dim=1).values.to(torch.float32) / subpixel_scale).long().clamp(0, height - 1)
        widths = max_x - min_x + 1
        heights = max_y - min_y + 1
        valid_bbox = (widths > 0) & (heights > 0)
        if not valid_bbox.any():
            start = tentative_end
            continue

        local_face_ids = torch.nonzero(valid_bbox, as_tuple=False).squeeze(1)
        tri_valid_fp = tri_chunk_fp[local_face_ids]
        min_x_v = min_x[local_face_ids]
        min_y_v = min_y[local_face_ids]
        widths_v = widths[local_face_ids]
        heights_v = heights[local_face_ids]
        areas_v = widths_v * heights_v
        total_candidates = int(areas_v.sum().item())
        if total_candidates == 0:
            start = tentative_end
            continue

        compact_ids = torch.arange(local_face_ids.shape[0], device=device, dtype=torch.long)
        compact_rep = torch.repeat_interleave(compact_ids, areas_v)
        local_offsets = torch.cumsum(areas_v, dim=0) - areas_v
        offsets_rep = torch.repeat_interleave(local_offsets, areas_v)
        linear = torch.arange(total_candidates, device=device, dtype=torch.long) - offsets_rep
        widths_rep = torch.repeat_interleave(widths_v, areas_v)

        x_local = linear % widths_rep
        y_local = torch.div(linear, widths_rep, rounding_mode="floor")
        min_x_rep = torch.repeat_interleave(min_x_v, areas_v)
        min_y_rep = torch.repeat_interleave(min_y_v, areas_v)
        x_int = min_x_rep + x_local
        y_int = min_y_rep + y_local
        x = x_int.to(torch.int64) * subpixel_scale
        y = y_int.to(torch.int64) * subpixel_scale

        tri = tri_valid_fp[compact_rep]
        v0 = tri[:, 0]
        v1 = tri[:, 1]
        v2 = tri[:, 2]
        area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        swap = area < 0
        if swap.any():
            v1_swapped = torch.where(swap[:, None], v2, v1)
            v2_swapped = torch.where(swap[:, None], v1, v2)
            v1 = v1_swapped
            v2 = v2_swapped
            area = torch.abs(area)
        nondegenerate = area >= area_eps
        safe_area = torch.where(nondegenerate, area, torch.ones_like(area))

        # Edge functions for top-left rule.
        e0 = (x - v1[:, 0]) * (v2[:, 1] - v1[:, 1]) - (y - v1[:, 1]) * (v2[:, 0] - v1[:, 0])
        e1 = (x - v2[:, 0]) * (v0[:, 1] - v2[:, 1]) - (y - v2[:, 1]) * (v0[:, 0] - v2[:, 0])
        e2 = (x - v0[:, 0]) * (v1[:, 1] - v0[:, 1]) - (y - v0[:, 1]) * (v1[:, 0] - v0[:, 0])

        # Screen-space top-left convention in y-down image coordinates.
        def edge_top_left(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            dx = b[:, 0] - a[:, 0]
            dy = b[:, 1] - a[:, 1]
            return (dy < 0) | ((dy == 0) & (dx > 0))

        tl0 = edge_top_left(v1, v2)
        tl1 = edge_top_left(v2, v0)
        tl2 = edge_top_left(v0, v1)

        inside = (
            nondegenerate
            & ((e0 < 0) | ((e0 == 0) & tl0))
            & ((e1 < 0) | ((e1 == 0) & tl1))
            & ((e2 < 0) | ((e2 == 0) & tl2))
        )
        if not inside.any():
            start = tentative_end
            continue

        compact_rep = compact_rep[inside]
        pix_idx = y_int[inside] * width + x_int[inside]
        face_idx = (start + local_face_ids[compact_rep]).long()
        safe_area_f = safe_area.to(torch.float32)
        w0 = -e0.to(torch.float32) / safe_area_f
        w1 = -e1.to(torch.float32) / safe_area_f
        w2 = -e2.to(torch.float32) / safe_area_f
        bary = torch.stack([w0[inside], w1[inside], w2[inside]], dim=-1).to(dtype=dtype)
        # If triangle winding was swapped for coverage tests, remap bary weights
        # back to original face vertex order (v0, v1_orig, v2_orig).
        swapped_inside = swap[inside]
        if swapped_inside.any():
            b1 = bary[:, 1].clone()
            b2 = bary[:, 2].clone()
            bary[:, 1] = torch.where(swapped_inside, b2, b1)
            bary[:, 2] = torch.where(swapped_inside, b1, b2)

        unique_pix, inv = torch.unique(pix_idx, sorted=False, return_inverse=True)
        if unique_pix.numel() == 0:
            start = tentative_end
            continue

        max_face = torch.iinfo(torch.long).max
        min_face_per_u = torch.full((unique_pix.shape[0],), max_face, dtype=torch.long, device=device)
        min_face_per_u.scatter_reduce_(0, inv, face_idx, reduce="amin", include_self=True)

        cand_idx = torch.arange(face_idx.shape[0], device=device, dtype=torch.long)
        large_idx = torch.full_like(cand_idx, cand_idx.shape[0])
        is_best_face = face_idx == min_face_per_u[inv]
        idx_for_u = torch.where(is_best_face, cand_idx, large_idx)
        chosen_idx_per_u = torch.full((unique_pix.shape[0],), cand_idx.shape[0], dtype=torch.long, device=device)
        chosen_idx_per_u.scatter_reduce_(0, inv, idx_for_u, reduce="amin", include_self=True)
        valid_u = chosen_idx_per_u < cand_idx.shape[0]
        if not valid_u.any():
            start = tentative_end
            continue

        chosen_idx = chosen_idx_per_u[valid_u]
        chosen_pix = unique_pix[valid_u]
        chosen_face = face_idx[chosen_idx]
        chosen_bary = bary[chosen_idx]

        prev_face = best_face[chosen_pix]
        better = (prev_face < 0) | (chosen_face < prev_face)
        if better.any():
            update_pix = chosen_pix[better]
            best_face[update_pix] = chosen_face[better]
            best_bary[update_pix] = chosen_bary[better]

        start = tentative_end

    face_indices = best_face.view(height, width)
    barycentric = best_bary.view(height, width, 3)
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
    detail_vertex_normals: Optional[torch.Tensor] = None,
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
    if detail_vertex_normals is not None:
        detail_interp = interpolate_vertex_attributes(detail_vertex_normals, faces, raster)
        detail_n = F.normalize(detail_interp[raster.mask], dim=-1)
    else:
        face_geo_normals = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
        detail_n = face_geo_normals[face_ids]
    t = F.normalize(tangent[face_ids], dim=-1)
    t = F.normalize(t - (t * n).sum(dim=-1, keepdim=True) * n, dim=-1)
    b = F.normalize(torch.cross(n, t, dim=-1), dim=-1)

    normal_ts = torch.stack(
        [
            (detail_n * t).sum(dim=-1),
            (detail_n * b).sum(dim=-1),
            (detail_n * n).sum(dim=-1),
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
