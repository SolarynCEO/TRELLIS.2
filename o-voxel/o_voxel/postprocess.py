from typing import *
import hashlib
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import cumesh
from .zyrex_baker_backend import resolve_baker_backend

try:
    import nvdiffrast.torch as dr
except Exception:
    dr = None


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_zyrex_sample_chunk_size() -> int:
    env_value = os.environ.get("ZYREX_BAKER_CHUNK_SIZE")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")):
        return 65536
    return 262144


def _resolve_emit_normal_map() -> bool:
    # Keep production-compatible default behavior: no normal texture emitted.
    return _is_truthy(os.environ.get("ZYREX_EMIT_NORMAL_MAP"))


def _normalize_volume_meta(
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor, None],
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor, None],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be list/tuple/ndarray/tensor, got {type(aabb)}"
    assert aabb.dim() == 2 and aabb.size(0) == 2 and aabb.size(1) == 3

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    assert isinstance(voxel_size, torch.Tensor) and voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor) and grid_size.dim() == 1 and grid_size.size(0) == 3
    return aabb, voxel_size, grid_size


def _hash_face_uv(out_faces: torch.Tensor, out_uvs: torch.Tensor) -> Dict[str, str]:
    faces_np = out_faces.detach().cpu().numpy().astype(np.int32, copy=False)
    uvs_np = out_uvs.detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "faces_sha1": hashlib.sha1(faces_np.tobytes()).hexdigest(),
        "uvs_sha1": hashlib.sha1(uvs_np.tobytes()).hexdigest(),
    }


def prepare_glb_bake_inputs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    aabb: torch.Tensor,
    grid_size: torch.Tensor,
    decimation_target: int = 1000000,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    progress_cb: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    vertices = vertices.cuda()
    faces = faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    src_vertices, src_faces = mesh.read()
    if progress_cb is not None:
        progress_cb()

    bvh = cumesh.cuBVH(src_vertices, src_faces)
    if progress_cb is not None:
        progress_cb()

    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(
            *cumesh.remeshing.remesh_narrow_band_dc(
                src_vertices,
                src_faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=remesh_project,
                verbose=verbose,
                bvh=bvh,
            )
        )
        mesh.simplify(decimation_target, verbose=verbose)
    if progress_cb is not None:
        progress_cb()

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    if progress_cb is not None:
        progress_cb()

    return {
        "mesh": mesh,
        "bvh": bvh,
        "source_vertices": src_vertices,
        "source_faces": src_faces,
        "out_vertices": out_vertices,
        "out_faces": out_faces,
        "out_uvs": out_uvs,
        "out_normals": out_normals,
        "hashes": _hash_face_uv(out_faces, out_uvs),
    }


def _bake_maps_nvdiffrast(
    prepared: Dict[str, Any],
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    texture_size: int,
) -> Dict[str, Any]:
    if dr is None:
        raise ImportError("nvdiffrast is required for ZYREX_BAKER=nvdiffrast but is not importable.")

    out_vertices = prepared["out_vertices"]
    out_faces = prepared["out_faces"]
    bvh = prepared["bvh"]
    src_vertices = prepared["source_vertices"]
    src_faces = prepared["source_faces"]

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat(
        [
            prepared["out_uvs"] * 2 - 1,
            torch.zeros_like(prepared["out_uvs"][:, :1]),
            torch.ones_like(prepared["out_uvs"][:, :1]),
        ],
        dim=-1,
    ).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device="cuda", dtype=torch.float32)

    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, out_faces[i : i + 100000], resolution=[texture_size, texture_size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)

    timings["rasterize_uv"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    face_indices = rast[0, ..., 3].long() - 1
    mask = face_indices >= 0
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    timings["interpolate_uv_to_pos"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = src_vertices[src_faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    timings["bvh_reproject"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device="cuda")
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )
    timings["sample_attr_volume"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pos_mapped = torch.zeros(texture_size, texture_size, 3, dtype=valid_pos.dtype, device=valid_pos.device)
    pos_mapped[mask] = valid_pos
    mask_np = mask.cpu().numpy()
    base_color = np.clip(attrs[..., attr_layout["base_color"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout["metallic"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout["roughness"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout["alpha"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    timings["quantize_maps"] = time.perf_counter() - t0
    return {
        "mask_raw": mask_np,
        "face_indices_raw": face_indices.cpu().numpy().astype(np.int32),
        "sample_positions_raw": pos_mapped.cpu().numpy().astype(np.float32),
        "sampled_attrs_raw": attrs.cpu().numpy().astype(np.float32),
        "timings": timings,
        "maps_raw": {
            "base_color": base_color,
            "metallic": metallic,
            "roughness": roughness,
            "alpha": alpha,
            "normal": None,
        },
    }


def _bake_maps_zyrex(
    prepared: Dict[str, Any],
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    texture_size: int,
    emit_normal_map: bool,
) -> Dict[str, Any]:
    from .zyrex_uv_baker import (
        bake_tangent_space_normal_map,
        interpolate_vertex_attributes,
        rasterize_uv_triangles,
    )

    out_vertices = prepared["out_vertices"]
    out_faces = prepared["out_faces"]
    out_uvs = prepared["out_uvs"]
    out_normals = prepared["out_normals"]
    bvh = prepared["bvh"]
    src_vertices = prepared["source_vertices"]
    src_faces = prepared["source_faces"]

    if not isinstance(out_normals, torch.Tensor):
        out_normals_t = torch.as_tensor(out_normals, dtype=torch.float32, device=out_vertices.device)
    else:
        out_normals_t = out_normals.to(out_vertices.device, dtype=torch.float32)

    def map_to_original_surface(query_positions: torch.Tensor) -> torch.Tensor:
        if query_positions.shape[0] == 0:
            return query_positions
        _, face_id, uvw = bvh.unsigned_distance(query_positions, return_uvw=True)
        orig_tri_verts = src_vertices[src_faces[face_id.long()]]
        return (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    raster = rasterize_uv_triangles(out_uvs, out_faces.long(), texture_size)
    pos = interpolate_vertex_attributes(out_vertices, out_faces.long(), raster)
    timings["rasterize_uv_and_interpolate_pos"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    query_positions = pos[raster.mask]
    query_positions = map_to_original_surface(query_positions)
    timings["bvh_reproject"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=out_vertices.device, dtype=attr_volume.dtype)
    attrs[raster.mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((query_positions - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )
    timings["sample_attr_volume"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    normal = None
    if emit_normal_map:
        normal = bake_tangent_space_normal_map(
            vertices=out_vertices,
            faces=out_faces.long(),
            uvs=out_uvs,
            vertex_normals=out_normals_t,
            raster=raster,
        )
        timings["bake_normal_map"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pos_mapped = torch.zeros(texture_size, texture_size, 3, dtype=query_positions.dtype, device=query_positions.device)
    pos_mapped[raster.mask] = query_positions

    mask_np = raster.mask.cpu().numpy()
    base_color = torch.clamp(attrs[..., attr_layout["base_color"]] * 255.0, 0.0, 255.0).byte().cpu().numpy()
    metallic = torch.clamp(attrs[..., attr_layout["metallic"]] * 255.0, 0.0, 255.0).byte().cpu().numpy()
    roughness = torch.clamp(attrs[..., attr_layout["roughness"]] * 255.0, 0.0, 255.0).byte().cpu().numpy()
    alpha = torch.clamp(attrs[..., attr_layout["alpha"]] * 255.0, 0.0, 255.0).byte().cpu().numpy()
    normal_np = None
    if normal is not None:
        normal_np = torch.clamp(normal * 255.0, 0.0, 255.0).byte().cpu().numpy()
    timings["quantize_maps"] = time.perf_counter() - t0
    return {
        "mask_raw": mask_np,
        "face_indices_raw": raster.face_indices.cpu().numpy().astype(np.int32),
        "sample_positions_raw": pos_mapped.cpu().numpy().astype(np.float32),
        "sampled_attrs_raw": attrs.cpu().numpy().astype(np.float32),
        "timings": timings,
        "maps_raw": {
            "base_color": base_color,
            "metallic": metallic,
            "roughness": roughness,
            "alpha": alpha,
            "normal": normal_np,
        },
    }


def bake_glb_maps_from_prepared(
    prepared: Dict[str, Any],
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    texture_size: int,
    backend: str,
    emit_normal_map: bool = False,
) -> Dict[str, Any]:
    if backend == "nvdiffrast":
        baked = _bake_maps_nvdiffrast(prepared, attr_volume, coords, attr_layout, aabb, voxel_size, grid_size, texture_size)
    elif backend == "zyrex":
        baked = _bake_maps_zyrex(
            prepared,
            attr_volume,
            coords,
            attr_layout,
            aabb,
            voxel_size,
            grid_size,
            texture_size,
            emit_normal_map=emit_normal_map,
        )
    else:
        raise NotImplementedError(
            f"ZYREX_BAKER={backend!r} is reserved but not implemented. "
            "Use ZYREX_BAKER=nvdiffrast (default) or ZYREX_BAKER=zyrex."
        )

    mask_raw = baked["mask_raw"]
    timings = dict(baked.get("timings", {}))
    t0 = time.perf_counter()
    maps_raw = baked["maps_raw"]
    mask_inv = (~mask_raw).astype(np.uint8)
    base_color = cv2.inpaint(maps_raw["base_color"], mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(maps_raw["metallic"], mask_inv, 1, cv2.INPAINT_TELEA)
    roughness = cv2.inpaint(maps_raw["roughness"], mask_inv, 1, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(maps_raw["alpha"], mask_inv, 1, cv2.INPAINT_TELEA)
    normal = maps_raw["normal"]
    if normal is not None:
        normal = cv2.inpaint(normal, mask_inv, 3, cv2.INPAINT_TELEA)
    timings["inpaint_maps"] = time.perf_counter() - t0

    maps_filled: Dict[str, Any] = {
        "base_color": base_color,
        "metallic": metallic,
        "roughness": roughness,
        "alpha": alpha,
        "normal": normal,
    }
    return {
        "backend": backend,
        "hashes": prepared["hashes"],
        "mask_raw": mask_raw,
        "face_indices_raw": baked.get("face_indices_raw"),
        "sample_positions_raw": baked.get("sample_positions_raw"),
        "sampled_attrs_raw": baked.get("sampled_attrs_raw"),
        "timings": timings,
        "maps_raw": maps_raw,
        "maps_filled": maps_filled,
    }


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
    return_debug: bool = False,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sprase tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
    """
    aabb, voxel_size, grid_size = _normalize_volume_meta(
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        device=coords.device,
    )
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    if use_tqdm:
        pbar.set_description("Preparing mesh/UVs")
    prepared = prepare_glb_bake_inputs(
        vertices=vertices,
        faces=faces,
        aabb=aabb,
        grid_size=grid_size,
        decimation_target=decimation_target,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        mesh_cluster_threshold_cone_half_angle_rad=mesh_cluster_threshold_cone_half_angle_rad,
        mesh_cluster_refine_iterations=mesh_cluster_refine_iterations,
        mesh_cluster_global_iterations=mesh_cluster_global_iterations,
        mesh_cluster_smooth_strength=mesh_cluster_smooth_strength,
        verbose=verbose,
        progress_cb=(lambda: pbar.update(1)) if use_tqdm else None,
    )
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
    backend = resolve_baker_backend()
    if backend == "nvdiffrast" and dr is None:
        raise ImportError(
            "ZYREX_BAKER is set to 'nvdiffrast' (default), but nvdiffrast could not be imported. "
            "Install/build nvdiffrast or set ZYREX_BAKER=zyrex."
        )
    if backend in ("pytorch3d", "kaolin"):
        raise NotImplementedError(
            f"ZYREX_BAKER={backend!r} is reserved but not implemented. "
            "Use ZYREX_BAKER=nvdiffrast (default) or ZYREX_BAKER=zyrex."
        )

    baked = bake_glb_maps_from_prepared(
        prepared=prepared,
        attr_volume=attr_volume,
        coords=coords,
        attr_layout=attr_layout,
        aabb=aabb,
        voxel_size=voxel_size,
        grid_size=grid_size,
        texture_size=texture_size,
        backend=backend,
        emit_normal_map=_resolve_emit_normal_map(),
    )
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    alpha_mode = 'OPAQUE'
    mask_raw = baked["mask_raw"]
    maps_raw = baked["maps_raw"]
    maps_filled = baked["maps_filled"]
    base_color = maps_filled["base_color"]
    metallic = maps_filled["metallic"]
    roughness = maps_filled["roughness"]
    alpha = maps_filled["alpha"]
    normal_map = maps_filled["normal"]

    if metallic.ndim == 2:
        metallic = metallic[..., None]
    if roughness.ndim == 2:
        roughness = roughness[..., None]
    if alpha.ndim == 2:
        alpha = alpha[..., None]

    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    if normal_map is not None:
        setattr(material, "normalTexture", Image.fromarray(normal_map))
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = prepared["out_vertices"].cpu().numpy()
    faces_np = prepared["out_faces"].cpu().numpy()
    uvs_np = prepared["out_uvs"].cpu().numpy()
    normals_np = prepared["out_normals"].cpu().numpy()
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    if not return_debug:
        return textured_mesh
    debug = {
        "backend": backend,
        "hashes": prepared["hashes"],
        "mask_raw": mask_raw,
        "maps_raw": maps_raw,
        "maps_filled": maps_filled,
    }
    return textured_mesh, debug