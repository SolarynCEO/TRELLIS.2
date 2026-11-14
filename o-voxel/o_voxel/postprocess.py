from typing import *
import os
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
    # Lower default chunk for 8GB profile.
    if _is_truthy(os.environ.get("ZYREX_LOW_VRAM")):
        return 65536
    return 262144


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
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()
    
    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        ))
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Simplify and clean the remeshed result (similar logic to above)
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    
    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
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
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    
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

    normal_map = None
    mask = None
    base_color = None
    metallic = None
    roughness = None
    alpha = None
    if backend == "zyrex":
        from .zyrex_uv_baker import bake_voxel_pbr_maps

        out_normals_t = out_normals
        if not isinstance(out_normals_t, torch.Tensor):
            out_normals_t = torch.as_tensor(out_normals_t, dtype=torch.float32, device=out_vertices.device)
        else:
            out_normals_t = out_normals_t.to(out_vertices.device, dtype=torch.float32)

        def map_to_original_surface(query_positions: torch.Tensor) -> torch.Tensor:
            if query_positions.shape[0] == 0:
                return query_positions
            _, face_id, uvw = bvh.unsigned_distance(query_positions, return_uvw=True)
            orig_tri_verts = vertices[faces[face_id.long()]]
            return (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        baked = bake_voxel_pbr_maps(
            vertices=out_vertices,
            faces=out_faces.long(),
            uvs=out_uvs,
            vertex_normals=out_normals_t,
            attr_volume=attr_volume,
            coords=coords,
            attr_layout=attr_layout,
            aabb=aabb,
            voxel_size=voxel_size,
            grid_size=grid_size,
            texture_size=texture_size,
            seam_padding_iterations=0,
            position_mapper=map_to_original_surface,
            sample_chunk_size=_resolve_zyrex_sample_chunk_size(),
        )
        mask = baked["mask"].cpu().numpy()
        base_color = torch.clamp(baked["base_color"] * 255.0, 0.0, 255.0).byte().cpu().numpy()
        metallic = torch.clamp(baked["metallic"] * 255.0, 0.0, 255.0).byte().cpu().numpy()
        roughness = torch.clamp(baked["roughness"] * 255.0, 0.0, 255.0).byte().cpu().numpy()
        alpha = torch.clamp(baked["alpha"] * 255.0, 0.0, 255.0).byte().cpu().numpy()
        normal_map = torch.clamp(baked["normal"] * 255.0, 0.0, 255.0).byte().cpu().numpy()
    else:
        # Setup differentiable rasterizer context
        ctx = dr.RasterizeCudaContext()
        # Prepare UV coordinates for rasterization (rendering in UV space)
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i  # Store face ID in alpha channel
            rast = torch.where(mask_chunk, rast_chunk, rast)

        # Mask of valid pixels in texture
        mask = rast[0, ..., 3] > 0

        # Interpolate 3D positions in UV space (finding 3D coord for every texel)
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]

        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]  # (N_new, 3, 3)
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        # Trilinear sampling from the attribute volume (Color, Material props)
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
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
    mask_raw = mask.copy()
    base_color_raw = base_color.copy()
    metallic_raw = metallic.copy()
    roughness_raw = roughness.copy()
    alpha_raw = alpha.copy()
    normal_raw = normal_map.copy() if normal_map is not None else None

    # Keep seam fill behavior equivalent to production nvdiffrast path.
    mask_inv = (~mask).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)
    if normal_map is not None:
        normal_map = cv2.inpaint(normal_map, mask_inv, 3, cv2.INPAINT_TELEA)

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
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
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
        "mask_raw": mask_raw,
        "maps_raw": {
            "base_color": base_color_raw,
            "metallic": metallic_raw,
            "roughness": roughness_raw,
            "alpha": alpha_raw,
            "normal": normal_raw,
        },
        "maps_filled": {
            "base_color": base_color,
            "metallic": metallic,
            "roughness": roughness,
            "alpha": alpha,
            "normal": normal_map,
        },
    }
    return textured_mesh, debug