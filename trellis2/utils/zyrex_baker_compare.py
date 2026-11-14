from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_ZYREX_UV_BAKER_PATH = ROOT / "o-voxel" / "o_voxel" / "zyrex_uv_baker.py"
_ZYREX_SPEC = importlib.util.spec_from_file_location("zyrex_uv_baker_compare_impl", _ZYREX_UV_BAKER_PATH)
if _ZYREX_SPEC is None or _ZYREX_SPEC.loader is None:
    raise ImportError(f"Unable to load zyrex uv baker module from {_ZYREX_UV_BAKER_PATH}")
_ZYREX_MODULE = importlib.util.module_from_spec(_ZYREX_SPEC)
sys.modules[_ZYREX_SPEC.name] = _ZYREX_MODULE
_ZYREX_SPEC.loader.exec_module(_ZYREX_MODULE)

bake_tangent_space_normal_map = _ZYREX_MODULE.bake_tangent_space_normal_map
bake_voxel_pbr_maps = _ZYREX_MODULE.bake_voxel_pbr_maps
dilate_texture = _ZYREX_MODULE.dilate_texture
interpolate_vertex_attributes = _ZYREX_MODULE.interpolate_vertex_attributes
rasterize_uv_triangles = _ZYREX_MODULE.rasterize_uv_triangles
sample_sparse_volume_trilinear = _ZYREX_MODULE.sample_sparse_volume_trilinear


def psnr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse <= eps:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_global(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    denom = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denom == 0:
        return 1.0
    return float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / denom)


def masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if pred.ndim == 2:
        pred = pred[..., None]
    if target.ndim == 2:
        target = target[..., None]
    pred_v = pred[mask].reshape(-1)
    target_v = target[mask].reshape(-1)
    if pred_v.size == 0:
        return {"psnr": 0.0, "ssim": 0.0}
    pred_v = np.clip(pred_v.astype(np.float32), 0.0, 1.0)
    target_v = np.clip(target_v.astype(np.float32), 0.0, 1.0)
    return {"psnr": psnr(pred_v, target_v), "ssim": ssim_global(pred_v, target_v)}


def synthetic_case(
    texture_size: int = 128,
    grid_res: int = 17,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    verts = torch.tensor(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long, device=device)
    uvs = verts[:, :2].clone()
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=torch.float32, device=device)

    gx, gy, gz = torch.meshgrid(
        torch.arange(grid_res, dtype=torch.long, device=device),
        torch.arange(grid_res, dtype=torch.long, device=device),
        torch.arange(grid_res, dtype=torch.long, device=device),
        indexing="ij",
    )
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    xyz = coords.float() / float(grid_res - 1)
    base = torch.stack([xyz[:, 0], xyz[:, 1], 0.25 + 0.5 * xyz[:, 0]], dim=-1)
    metallic = 0.1 + 0.7 * xyz[:, 0:1]
    roughness = 0.2 + 0.6 * xyz[:, 1:2]
    alpha = torch.ones_like(metallic)
    attr_volume = torch.cat([base, metallic, roughness, alpha], dim=-1)
    attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }

    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    grid_size = torch.tensor([grid_res, grid_res, grid_res], dtype=torch.long, device=device)
    voxel_size = (aabb[1] - aabb[0]) / (grid_size.float() - 1.0)

    raster = rasterize_uv_triangles(uvs, faces, texture_size)
    pos_map = interpolate_vertex_attributes(verts, faces, raster)
    expected = {
        "base_color": torch.stack([pos_map[..., 0], pos_map[..., 1], 0.25 + 0.5 * pos_map[..., 0]], dim=-1),
        "metallic": (0.1 + 0.7 * pos_map[..., 0:1]),
        "roughness": (0.2 + 0.6 * pos_map[..., 1:2]),
        "alpha": torch.ones(texture_size, texture_size, 1, dtype=torch.float32, device=device),
    }
    expected["normal"] = torch.zeros(texture_size, texture_size, 3, dtype=torch.float32, device=device)
    expected["normal"][..., 0] = 0.5
    expected["normal"][..., 1] = 0.5
    expected["normal"][..., 2] = 1.0

    return {
        "vertices": verts,
        "faces": faces,
        "uvs": uvs,
        "normals": normals,
        "attr_volume": attr_volume,
        "coords": coords,
        "attr_layout": attr_layout,
        "aabb": aabb,
        "grid_size": grid_size,
        "voxel_size": voxel_size,
        "expected": expected,
    }


def run_synthetic_cpu(texture_size: int = 128) -> Dict[str, object]:
    torch.manual_seed(0)
    np.random.seed(0)
    case = synthetic_case(texture_size=texture_size, device="cpu")
    t0 = time.perf_counter()
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
        texture_size=texture_size,
        seam_padding_iterations=6,
    )
    elapsed = time.perf_counter() - t0
    mask = baked["mask"].cpu().numpy().astype(bool)

    metrics: Dict[str, Dict[str, float]] = {}
    for key in ("base_color", "metallic", "roughness", "alpha", "normal"):
        pred = baked[key].cpu().numpy()
        target = case["expected"][key].cpu().numpy()
        metrics[key] = masked_metrics(pred, target, mask)

    normal = baked["normal"][baked["mask"]]
    normal_ok = bool(torch.all(normal[:, 2] > 0.95).item()) if normal.numel() > 0 else False
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "texture_size": texture_size,
        "coverage_ratio": float(mask.mean()),
        "metrics": metrics,
        "normal_orientation_ok": normal_ok,
        "processing_time_seconds": elapsed,
        "peak_rss_mb": rss_mb,
    }


def _sample_texture(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    img = image.astype(np.float32) / 255.0
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(1.0 - uv[:, 1], 0.0, 1.0)
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    return c0 * (1.0 - wy) + c1 * wy


def _sample_texture_torch(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return _sample_texture((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), uv)


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    return rx @ rz


def _render_textured_view_cpu(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    resolution: int = 192,
    focal_px: float = 180.0,
    camera_distance: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    verts = vertices.astype(np.float32).copy()
    center = verts.mean(axis=0, keepdims=True)
    verts -= center
    scale = max(np.linalg.norm(verts, axis=1).max(), 1e-6)
    verts /= scale

    rot = _rotation_matrix(yaw_deg, pitch_deg)
    v = verts @ rot.T
    z = v[:, 2] + camera_distance
    z = np.clip(z, 1e-4, None)
    px = v[:, 0] / z * focal_px + (resolution - 1) * 0.5
    py = -v[:, 1] / z * focal_px + (resolution - 1) * 0.5

    depth = np.full((resolution, resolution), np.inf, dtype=np.float32)
    image = np.zeros((resolution, resolution, 3), dtype=np.float32)
    mask = np.zeros((resolution, resolution), dtype=bool)

    for face in faces:
        idx0, idx1, idx2 = int(face[0]), int(face[1]), int(face[2])
        tri = np.array(
            [[px[idx0], py[idx0]], [px[idx1], py[idx1]], [px[idx2], py[idx2]]],
            dtype=np.float32,
        )
        z_tri = np.array([z[idx0], z[idx1], z[idx2]], dtype=np.float32)
        uv_tri = np.array([uvs[idx0], uvs[idx1], uvs[idx2]], dtype=np.float32)

        min_x = max(int(np.floor(tri[:, 0].min())), 0)
        max_x = min(int(np.ceil(tri[:, 0].max())), resolution - 1)
        min_y = max(int(np.floor(tri[:, 1].min())), 0)
        max_y = min(int(np.ceil(tri[:, 1].max())), resolution - 1)
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        yy, xx = np.meshgrid(ys, xs, indexing="ij")

        ax, ay = tri[0]
        bx, by = tri[1]
        cx, cy = tri[2]
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denom) < 1e-8:
            continue

        w0 = ((by - cy) * (xx - cx) + (cx - bx) * (yy - cy)) / denom
        w1 = ((cy - ay) * (xx - cx) + (ax - cx) * (yy - cy)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        z_interp = w0 * z_tri[0] + w1 * z_tri[1] + w2 * z_tri[2]
        region_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        z_test = inside & (z_interp < region_depth)
        if not z_test.any():
            continue

        uv_interp = (
            uv_tri[0][None, None] * w0[..., None]
            + uv_tri[1][None, None] * w1[..., None]
            + uv_tri[2][None, None] * w2[..., None]
        )
        uv_valid = uv_interp[z_test]
        tex = _sample_texture_torch(texture, uv_valid)

        region_img = image[min_y : max_y + 1, min_x : max_x + 1]
        region_depth[z_test] = z_interp[z_test]
        region_img[z_test] = tex
        mask[min_y : max_y + 1, min_x : max_x + 1][z_test] = True

    return image, mask


def _rendered_view_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    reference_tex: np.ndarray,
    rebaked_tex: np.ndarray,
    views: Iterable[Tuple[float, float]] = ((20.0, 10.0), (140.0, 10.0), (260.0, -12.0)),
    resolution: int = 192,
    max_faces: int = 30000,
) -> Dict[str, object]:
    total_faces = int(faces.shape[0])
    if total_faces > max_faces:
        keep = np.linspace(0, total_faces - 1, max_faces, dtype=np.int64)
        faces = faces[keep]

    per_view = []
    psnrs = []
    ssims = []
    for yaw, pitch in views:
        ref_img, ref_mask = _render_textured_view_cpu(
            vertices, faces, uvs, reference_tex, yaw, pitch, resolution=resolution
        )
        reb_img, reb_mask = _render_textured_view_cpu(
            vertices, faces, uvs, rebaked_tex, yaw, pitch, resolution=resolution
        )
        union = ref_mask | reb_mask
        metrics = masked_metrics(reb_img, ref_img, union)
        per_view.append(
            {
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "coverage_ratio": float(union.mean()),
                **metrics,
            }
        )
        psnrs.append(metrics["psnr"])
        ssims.append(metrics["ssim"])
    return {
        "status": "measured",
        "mask": "rendered union mask",
        "resolution": resolution,
        "total_faces": total_faces,
        "faces_rendered": int(faces.shape[0]),
        "mean_psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "mean_ssim": float(np.mean(ssims)) if ssims else 0.0,
        "per_view": per_view,
    }


def _extract_first_textured_mesh(glb_path: Path):
    import trimesh

    loaded = trimesh.load(glb_path, force="scene")
    geometries = loaded.geometry.values() if hasattr(loaded, "geometry") else [loaded]
    for geom in geometries:
        visual = getattr(geom, "visual", None)
        if visual is None or getattr(visual, "uv", None) is None:
            continue
        material = getattr(visual, "material", None)
        if material is None:
            continue
        if getattr(material, "baseColorTexture", None) is not None:
            return geom
    raise RuntimeError(f"No textured mesh with UV/baseColorTexture found in {glb_path}")


def run_real_asset_rebake(
    glb_path: Path,
    render_views: bool = True,
    render_resolution: int = 192,
    render_max_faces: int = 30000,
) -> Dict[str, object]:
    mesh = _extract_first_textured_mesh(glb_path)
    material = mesh.visual.material
    base = np.array(material.baseColorTexture.convert("RGBA"))
    mr = material.metallicRoughnessTexture
    mr_tex = np.array(mr.convert("RGB")) if mr is not None else np.zeros((*base.shape[:2], 3), dtype=np.uint8)
    normal_tex = getattr(material, "normalTexture", None)
    normal = np.array(normal_tex.convert("RGB")) if normal_tex is not None else None

    h, w = base.shape[:2]
    uvs = torch.from_numpy(mesh.visual.uv.astype(np.float32))
    faces = torch.from_numpy(mesh.faces.astype(np.int64))
    raster = rasterize_uv_triangles(uvs, faces, (h, w))
    uv_map = interpolate_vertex_attributes(uvs, faces, raster).cpu().numpy()
    footprint_mask = raster.mask.cpu().numpy().astype(bool)
    source_coverage_mask = base[..., 3] > 0
    union_mask = footprint_mask | source_coverage_mask

    rebaked_base = np.zeros((h, w, 4), dtype=np.float32)
    rebaked_mr = np.zeros((h, w, 3), dtype=np.float32)
    if footprint_mask.any():
        uv_valid = uv_map[footprint_mask]
        rebaked_base[footprint_mask] = _sample_texture(base, uv_valid)
        rebaked_mr[footprint_mask] = _sample_texture(mr_tex, uv_valid)

    rebaked_base_t = dilate_texture(torch.from_numpy(rebaked_base), torch.from_numpy(footprint_mask), iterations=6).numpy()
    rebaked_mr_t = dilate_texture(torch.from_numpy(rebaked_mr), torch.from_numpy(footprint_mask), iterations=6).numpy()
    base_ref = base.astype(np.float32) / 255.0
    mr_ref = mr_tex.astype(np.float32) / 255.0

    results: Dict[str, object] = {
        "glb": str(glb_path),
        "texture_size": [h, w],
        "coverage": {
            "footprint_ratio": float(footprint_mask.mean()),
            "source_alpha_ratio": float(source_coverage_mask.mean()),
            "union_ratio": float(union_mask.mean()),
        },
        "map_metrics": {
            "legacy_full_texture": {
                "base_color_rgba": masked_metrics(rebaked_base_t, base_ref, np.ones((h, w), dtype=bool)),
                "metallic_roughness_rgb": masked_metrics(rebaked_mr_t, mr_ref, np.ones((h, w), dtype=bool)),
            },
            "corrected_covered_texels": {
                "mask": "mesh rasterized UV footprint",
                "base_color_rgba": masked_metrics(rebaked_base_t, base_ref, footprint_mask),
                "metallic_roughness_rgb": masked_metrics(rebaked_mr_t, mr_ref, footprint_mask),
            },
            "corrected_union_mask": {
                "mask": "footprint union source-alpha-coverage",
                "base_color_rgba": masked_metrics(rebaked_base_t, base_ref, union_mask),
                "metallic_roughness_rgb": masked_metrics(rebaked_mr_t, mr_ref, union_mask),
            },
        },
    }
    if normal is not None:
        rebaked_normal = np.zeros((h, w, 3), dtype=np.float32)
        if footprint_mask.any():
            rebaked_normal[footprint_mask] = _sample_texture(normal, uv_map[footprint_mask])
        rebaked_normal_t = dilate_texture(torch.from_numpy(rebaked_normal), torch.from_numpy(footprint_mask), iterations=6).numpy()
        normal_ref = normal.astype(np.float32) / 255.0
        results["map_metrics"]["legacy_full_texture"]["normal_rgb"] = masked_metrics(rebaked_normal_t, normal_ref, np.ones((h, w), dtype=bool))
        results["map_metrics"]["corrected_covered_texels"]["normal_rgb"] = masked_metrics(rebaked_normal_t, normal_ref, footprint_mask)
        results["map_metrics"]["corrected_union_mask"]["normal_rgb"] = masked_metrics(rebaked_normal_t, normal_ref, union_mask)

    if render_views:
        try:
            results["rendered_view_metrics"] = _rendered_view_metrics(
                vertices=mesh.vertices.astype(np.float32),
                faces=mesh.faces.astype(np.int64),
                uvs=mesh.visual.uv.astype(np.float32),
                reference_tex=base_ref[..., :3],
                rebaked_tex=rebaked_base_t[..., :3],
                resolution=render_resolution,
                max_faces=render_max_faces,
            )
        except Exception as exc:
            results["rendered_view_metrics"] = {
                "status": "pending CPU render comparison",
                "reason": f"CPU render comparison failed: {exc}",
            }
    else:
        results["rendered_view_metrics"] = {"status": "not_run"}
    return results


def nvdiffrast_reference_bake(
    case: Dict[str, torch.Tensor],
    texture_size: int,
) -> Dict[str, torch.Tensor]:
    import nvdiffrast.torch as dr

    vertices = case["vertices"].cuda()
    faces = case["faces"].int().cuda()
    uvs = case["uvs"].cuda()
    normals = case["normals"].cuda()
    attr_volume = case["attr_volume"].cuda()
    coords = case["coords"].cuda()
    aabb = case["aabb"].cuda()
    voxel_size = case["voxel_size"].cuda()
    grid_size = case["grid_size"].cuda()

    ctx = dr.RasterizeCudaContext(device=vertices.device)
    uvs_rast = torch.cat(
        [uvs * 2 - 1, torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])],
        dim=-1,
    )[None]
    rast, _ = dr.rasterize(ctx, uvs_rast, faces, resolution=[texture_size, texture_size])
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(vertices[None], rast, faces)[0][0]
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[-1], device="cuda")
    attrs[mask] = sample_sparse_volume_trilinear(attr_volume, coords, grid_size, aabb, voxel_size, pos[mask])
    normal = bake_tangent_space_normal_map(vertices, faces.long(), uvs, normals, rasterize_uv_triangles(uvs, faces.long(), texture_size))

    out = {
        "mask": mask.cpu(),
        "base_color": dilate_texture(attrs[..., case["attr_layout"]["base_color"]], mask, iterations=6).cpu(),
        "metallic": dilate_texture(attrs[..., case["attr_layout"]["metallic"]], mask, iterations=6).cpu(),
        "roughness": dilate_texture(attrs[..., case["attr_layout"]["roughness"]], mask, iterations=6).cpu(),
        "alpha": dilate_texture(attrs[..., case["attr_layout"]["alpha"]], mask, iterations=6).cpu(),
        "normal": dilate_texture(normal, mask, iterations=6).cpu(),
    }
    return out


def run_gpu_comparison(texture_size: int) -> Dict[str, object]:
    pending = {
        "status": "pending GPU run",
        "reason": "CUDA and nvdiffrast are required to run side-by-side comparisons.",
        "map_metrics": "pending GPU run",
        "rendered_view_metrics": "pending GPU run",
        "processing_time_seconds": "pending GPU run",
        "gpu_memory_mb": "pending GPU run",
    }
    if not torch.cuda.is_available():
        return pending
    try:
        import nvdiffrast.torch as _  # noqa: F401
    except Exception:
        return pending

    torch.manual_seed(0)
    np.random.seed(0)
    case = synthetic_case(texture_size=texture_size, device="cuda")
    t0 = time.perf_counter()
    ref = nvdiffrast_reference_bake(case, texture_size=texture_size)
    ref_t = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    zyrex = bake_voxel_pbr_maps(
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
        texture_size=texture_size,
        seam_padding_iterations=6,
    )
    zyrex_t = time.perf_counter() - t1
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    mask = ref["mask"].numpy().astype(bool) & zyrex["mask"].cpu().numpy().astype(bool)
    map_metrics = {}
    for key in ("base_color", "metallic", "roughness", "alpha", "normal"):
        map_metrics[key] = masked_metrics(
            zyrex[key].detach().cpu().numpy(),
            ref[key].numpy(),
            mask,
        )
    return {
        "status": "measured",
        "map_metrics": map_metrics,
        "rendered_view_metrics": "pending GPU run",
        "processing_time_seconds": {
            "nvdiffrast_reference": ref_t,
            "zyrex": zyrex_t,
        },
        "gpu_memory_mb": peak_mem_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU/GPU comparison harness for the ZYREX UV baker prototype.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON metrics.")
    parser.add_argument("--texture-size", type=int, default=128)
    parser.add_argument("--real-asset", type=Path, default=None, help="Optional GLB path for real-asset re-bake sanity checks.")
    parser.add_argument("--run-gpu", action="store_true", help="Run nvdiffrast vs zyrex GPU comparison when available.")
    parser.add_argument("--render-resolution", type=int, default=192, help="CPU rendered-view resolution for real-asset comparison.")
    parser.add_argument("--render-max-faces", type=int, default=30000, help="Maximum faces to use for CPU rendered-view comparisons.")
    parser.add_argument("--skip-rendered-views", action="store_true", help="Skip CPU rendered-view comparison for real assets.")
    args = parser.parse_args()

    report: Dict[str, object] = {
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "synthetic_cpu": run_synthetic_cpu(texture_size=args.texture_size),
        "gpu_comparison": run_gpu_comparison(args.texture_size) if args.run_gpu else {
            "status": "pending GPU run",
            "reason": "Run again with --run-gpu on a CUDA machine with nvdiffrast installed.",
            "map_metrics": "pending GPU run",
            "rendered_view_metrics": "pending GPU run",
            "processing_time_seconds": "pending GPU run",
            "gpu_memory_mb": "pending GPU run",
        },
    }
    if args.real_asset is not None:
        report["real_asset"] = run_real_asset_rebake(
            args.real_asset,
            render_views=not args.skip_rendered_views,
            render_resolution=args.render_resolution,
            render_max_faces=args.render_max_faces,
        )
    else:
        report["real_asset"] = {"status": "not_run"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
