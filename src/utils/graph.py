from typing import Union

import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed, cpu_count
from typing import List, Tuple


import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed, cpu_count
from typing import List, Tuple
from tqdm import tqdm

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def _process_chunk_tree_aware_gpu(args):
    """GPU-accelerated worker function for tree-aware edge building"""
    if not HAS_CUPY:
        # Fallback to CPU version
        return _process_chunk_tree_aware(args)
    
    chunk_indices, centroids, pca_dirs, r_axial, r_lateral, cos_axial, cos_pca = args
    
    # Move ALL data to GPU once at the start
    centroids_gpu = cp.asarray(centroids)
    pca_dirs_gpu = cp.asarray(pca_dirs)
    
    # Build KDTree on CPU (still needed for spatial queries)
    tree = cKDTree(centroids)
    
    # Collect all neighbor pairs first (CPU operation)
    all_pairs = []
    for i in chunk_indices:
        ci = centroids[i]
        neighbors = tree.query_ball_point(ci, r_axial)
        for j in neighbors:
            if j > i:
                all_pairs.append((i, j))
    
    if not all_pairs:
        return []
    
    # Convert to GPU arrays for BATCH processing
    pairs_gpu = cp.array(all_pairs, dtype=cp.int32)
    i_indices = pairs_gpu[:, 0]
    j_indices = pairs_gpu[:, 1]
    
    # Batch compute all distances (single GPU operation)
    ci_gpu = centroids_gpu[i_indices]
    cj_gpu = centroids_gpu[j_indices]
    v_gpu = cj_gpu - ci_gpu
    dists_gpu = cp.linalg.norm(v_gpu, axis=1)
    
    # Filter valid distances
    valid_mask = dists_gpu > 1e-6
    if not cp.any(valid_mask):
        return []
    
    # Apply mask to all arrays at once
    v_gpu = v_gpu[valid_mask]
    dists_gpu = dists_gpu[valid_mask]
    i_indices = i_indices[valid_mask]
    j_indices = j_indices[valid_mask]
    
    # Normalize all vectors at once
    v_gpu = v_gpu / dists_gpu[:, cp.newaxis]
    
    # Batch PCA alignment
    pca_i = pca_dirs_gpu[i_indices]
    pca_j = pca_dirs_gpu[j_indices]
    pca_align = cp.abs(cp.sum(pca_i * pca_j, axis=1))
    
    pca_mask = pca_align >= cos_pca
    if not cp.any(pca_mask):
        return []
    
    # Apply PCA mask
    v_gpu = v_gpu[pca_mask]
    dists_gpu = dists_gpu[pca_mask]
    i_indices = i_indices[pca_mask]
    j_indices = j_indices[pca_mask]
    pca_i = pca_i[pca_mask]
    pca_j = pca_j[pca_mask]
    
    # Batch axial alignment
    axial_i = cp.abs(cp.sum(v_gpu * pca_i, axis=1))
    axial_j = cp.abs(cp.sum(v_gpu * pca_j, axis=1))
    axial = cp.maximum(axial_i, axial_j)
    
    # Final filter
    final_mask = (axial > cos_axial) | (dists_gpu < r_lateral)
    
    # Single GPU->CPU transfer at the end
    i_final = cp.asnumpy(i_indices[final_mask])
    j_final = cp.asnumpy(j_indices[final_mask])
    
    edges = list(zip(i_final.tolist(), j_final.tolist()))
    
    return edges


def _process_chunk_tree_aware(args):
    """CPU fallback worker function for tree-aware edge building"""
    chunk_indices, centroids, pca_dirs, r_axial, r_lateral, cos_axial, cos_pca = args
    
    # Rebuild tree in worker
    tree = cKDTree(centroids)
    edges = []
    
    for i in chunk_indices:
        ci = centroids[i]
        neighbors = tree.query_ball_point(ci, r_axial)
        
        for j in neighbors:
            if j <= i:
                continue
            
            cj = centroids[j]
            v = cj - ci
            dist = np.linalg.norm(v)
            
            if dist < 1e-6:
                continue
            
            v /= dist
            
            # --- PCA alignment ---
            pca_align = abs(np.dot(pca_dirs[i], pca_dirs[j]))
            if pca_align < cos_pca:
                continue
            
            # --- axial vs lateral ---
            axial_i = abs(np.dot(v, pca_dirs[i]))
            axial_j = abs(np.dot(v, pca_dirs[j]))
            
            axial = max(axial_i, axial_j)
            
            if axial > cos_axial or dist < r_lateral:
                edges.append((i, j))
    
    return edges


def build_edges_tree_aware(
    superpoints,
    r_axial=1.5,
    r_lateral=0.3,
    theta_axial_deg=25.0,
    theta_pca_deg=20.0,
    n_jobs=-1,
    use_gpu=True
):
    """
    Build edges between superpoints with PCA and geometric constraints.
    
    Parameters:
    -----------
    superpoints : list
        List of superpoint objects with .centroid and .pca_dir attributes
    r_axial : float
        Maximum axial search radius
    r_lateral : float
        Lateral connection threshold
    theta_axial_deg : float
        Axial alignment angle threshold in degrees
    theta_pca_deg : float
        PCA alignment angle threshold in degrees
    n_jobs : int
        Number of parallel jobs. -1 uses all CPUs, 1 disables multiprocessing
    use_gpu : bool
        Whether to use GPU acceleration (requires CuPy)
    
    Returns:
    --------
    list of tuples
        List of edges (i, j) where i < j
    """
    centroids = np.array([sp.centroid for sp in superpoints])
    pca_dirs = np.array([sp.pca_dir for sp in superpoints])
    
    cos_axial = np.cos(np.deg2rad(theta_axial_deg))
    cos_pca = np.cos(np.deg2rad(theta_pca_deg))
    
    # Check GPU availability
    if use_gpu and not HAS_CUPY:
        print("Warning: CuPy not available, falling back to CPU")
        use_gpu = False
    
    # Single-threaded fallback
    if n_jobs == 1:
        if use_gpu:
            # Single GPU processing
            return _build_edges_single_gpu(
                centroids, pca_dirs, r_axial, r_lateral, cos_axial, cos_pca
            )
        else:
            # Single CPU processing
            tree = cKDTree(centroids)
            edges = []
            
            for i, ci in enumerate(centroids):
                neighbors = tree.query_ball_point(ci, r_axial)
                
                for j in neighbors:
                    if j <= i:
                        continue
                    
                    cj = centroids[j]
                    v = cj - ci
                    dist = np.linalg.norm(v)
                    
                    if dist < 1e-6:
                        continue
                    
                    v /= dist
                    
                    # --- PCA alignment ---
                    pca_align = abs(np.dot(pca_dirs[i], pca_dirs[j]))
                    if pca_align < cos_pca:
                        continue
                    
                    # --- axial vs lateral ---
                    axial_i = abs(np.dot(v, pca_dirs[i]))
                    axial_j = abs(np.dot(v, pca_dirs[j]))
                    
                    axial = max(axial_i, axial_j)
                    
                    if axial > cos_axial or dist < r_lateral:
                        edges.append((i, j))
            
            return edges
    
    # Multiprocessing with joblib
    n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
    n_points = len(centroids)
    
    # Split work into chunks
    chunk_size = max(1, n_points // n_workers)
    chunks = []
    
    for i in range(0, n_points, chunk_size):
        chunk_indices = list(range(i, min(i + chunk_size, n_points)))
        chunks.append((
            chunk_indices,
            centroids,
            pca_dirs,
            r_axial,
            r_lateral,
            cos_axial,
            cos_pca
        ))
    
    # Choose worker function based on GPU availability
    worker_func = _process_chunk_tree_aware_gpu if use_gpu else _process_chunk_tree_aware
    
    # Process chunks in parallel with joblib
    results = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(worker_func)(chunk) for chunk in chunks
    )
    
    # Merge results
    edges = []
    for chunk_edges in results:
        edges.extend(chunk_edges)
    
    # Remove duplicates (shouldn't happen with i < j constraint, but just in case)
    edges = list(set(edges))
    edges.sort()
    
    return edges


def _build_edges_single_gpu(centroids, pca_dirs, r_axial, r_lateral, cos_axial, cos_pca):
    """
    Fully GPU-accelerated single-threaded version.
    Most efficient for datasets that fit in GPU memory.
    """
    # Move all data to GPU ONCE
    centroids_gpu = cp.asarray(centroids)
    pca_dirs_gpu = cp.asarray(pca_dirs)
    
    # KDTree still on CPU
    tree = cKDTree(centroids)
    n_points = len(centroids)
    
    # Collect all neighbor pairs first (CPU)
    all_pairs = []
    for i in range(n_points):
        ci = centroids[i]
        neighbors = tree.query_ball_point(ci, r_axial)
        for j in neighbors:
            if j > i:
                all_pairs.append((i, j))
    
    if not all_pairs:
        return []
    
    # SINGLE batch GPU processing
    pairs_gpu = cp.array(all_pairs, dtype=cp.int32)
    i_indices = pairs_gpu[:, 0]
    j_indices = pairs_gpu[:, 1]
    
    # Batch operations
    ci_gpu = centroids_gpu[i_indices]
    cj_gpu = centroids_gpu[j_indices]
    v_gpu = cj_gpu - ci_gpu
    dists_gpu = cp.linalg.norm(v_gpu, axis=1)
    
    valid_mask = dists_gpu > 1e-6
    if not cp.any(valid_mask):
        return []
    
    v_gpu = v_gpu[valid_mask]
    dists_gpu = dists_gpu[valid_mask]
    i_indices = i_indices[valid_mask]
    j_indices = j_indices[valid_mask]
    v_gpu = v_gpu / dists_gpu[:, cp.newaxis]
    
    # PCA alignment
    pca_i = pca_dirs_gpu[i_indices]
    pca_j = pca_dirs_gpu[j_indices]
    pca_align = cp.abs(cp.sum(pca_i * pca_j, axis=1))
    pca_mask = pca_align >= cos_pca
    
    if not cp.any(pca_mask):
        return []
    
    v_gpu = v_gpu[pca_mask]
    dists_gpu = dists_gpu[pca_mask]
    i_indices = i_indices[pca_mask]
    j_indices = j_indices[pca_mask]
    pca_i = pca_i[pca_mask]
    pca_j = pca_j[pca_mask]
    
    # Axial alignment
    axial_i = cp.abs(cp.sum(v_gpu * pca_i, axis=1))
    axial_j = cp.abs(cp.sum(v_gpu * pca_j, axis=1))
    axial = cp.maximum(axial_i, axial_j)
    
    final_mask = (axial > cos_axial) | (dists_gpu < r_lateral)
    
    # Single GPU->CPU transfer
    i_final = cp.asnumpy(i_indices[final_mask])
    j_final = cp.asnumpy(j_indices[final_mask])
    
    edges = list(zip(i_final.tolist(), j_final.tolist()))
    
    return edges


# Hybrid GPU-CPU version for very large datasets
def build_edges_tree_aware_hybrid(
    superpoints,
    r_axial=1.5,
    r_lateral=0.3,
    theta_axial_deg=25.0,
    theta_pca_deg=20.0,
    batch_size=10000
):
    """
    Hybrid version that processes large datasets in batches on GPU.
    Good for datasets larger than GPU memory.
    
    Parameters:
    -----------
    batch_size : int
        Number of points to process at once on GPU
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for GPU acceleration")
    
    centroids = np.array([sp.centroid for sp in superpoints])
    pca_dirs = np.array([sp.pca_dir for sp in superpoints])
    
    cos_axial = np.cos(np.deg2rad(theta_axial_deg))
    cos_pca = np.cos(np.deg2rad(theta_pca_deg))
    
    # Move static data to GPU
    centroids_gpu = cp.asarray(centroids)
    pca_dirs_gpu = cp.asarray(pca_dirs)
    
    # Build KDTree on CPU
    tree = cKDTree(centroids)
    
    edges = []
    n_points = len(centroids)
    
    # Process in batches
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_indices = list(range(batch_start, batch_end))
        
        batch_edges = []
        
        for i in batch_indices:
            ci = centroids[i]
            neighbors = tree.query_ball_point(ci, r_axial)
            neighbors = [j for j in neighbors if j > i]
            
            if not neighbors:
                continue
            
            # GPU computation for this point's neighbors
            neighbor_indices = cp.array(neighbors)
            ci_gpu = centroids_gpu[i]
            cj_gpu = centroids_gpu[neighbor_indices]
            
            v_gpu = cj_gpu - ci_gpu
            dists_gpu = cp.linalg.norm(v_gpu, axis=1)
            
            valid_mask = dists_gpu > 1e-6
            if not cp.any(valid_mask):
                continue
            
            v_gpu = v_gpu[valid_mask]
            dists_gpu = dists_gpu[valid_mask]
            neighbor_indices = neighbor_indices[valid_mask]
            v_gpu = v_gpu / dists_gpu[:, cp.newaxis]
            
            pca_i = pca_dirs_gpu[i]
            pca_j = pca_dirs_gpu[neighbor_indices]
            pca_align = cp.abs(cp.sum(pca_j * pca_i, axis=1))
            pca_mask = pca_align >= cos_pca
            
            if not cp.any(pca_mask):
                continue
            
            v_gpu = v_gpu[pca_mask]
            dists_gpu = dists_gpu[pca_mask]
            neighbor_indices = neighbor_indices[pca_mask]
            pca_j = pca_j[pca_mask]
            
            axial_i = cp.abs(cp.sum(v_gpu * pca_i, axis=1))
            axial_j = cp.abs(cp.sum(v_gpu * pca_j, axis=1))
            axial = cp.maximum(axial_i, axial_j)
            
            final_mask = (axial > cos_axial) | (dists_gpu < r_lateral)
            valid_neighbors = cp.asnumpy(neighbor_indices[final_mask])
            
            for j in valid_neighbors:
                batch_edges.append((i, int(j)))
        
        edges.extend(batch_edges)
        
        # Clear GPU cache periodically
        if batch_end % (batch_size * 5) == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    edges = list(set(edges))
    edges.sort()
    
    return edges

def build_edges(centroids, chunk = 500, radius: float = 1.5, verbose = False):
    if len(centroids) == 0:
        return np.empty((0, 3), dtype=np.int64)
    
    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids to be 2D array with 3 columns, got shape {centroids.shape}")
    
    tree = cKDTree(centroids)
    edges = []

    neighbors = []
    if verbose:
        pbar = tqdm(range(0, centroids.shape[0], chunk), total=centroids.shape[0]//chunk + 1, desc="Building spatial index", position = 1, leave=False)
    else:
        pbar = range(0, centroids.shape[0], chunk)

    for i in pbar:
        end_idx = min(i + chunk, centroids.shape[0])
        batch_neighbors = tree.query_ball_point(centroids[i:end_idx], radius)
        neighbors.extend(batch_neighbors)
    neighbors = np.asarray(neighbors, dtype=object)

    del tree

    edges = []

    if verbose:
        pbar = tqdm(range(len(neighbors)), total=len(neighbors), desc="Building edges", position = 1, leave= False)
    else:
        pbar = range(len(neighbors))

    for i, nbrs in pbar:
        for j in nbrs:
            if i < j:
                edges.append((i, j))

    # Remove duplicates and sort
    edges = list(set(edges))
    edges.sort()
    return np.asarray(edges, np.int64)

def build_edges_sp(centroids: np.ndarray, radius:float = 1.5):


    tree = cKDTree(centroids)

    edges = tree.query_pairs(radius, output_type='ndarray')

    return edges.astype(np.int64)



def _process_chunk_edges(centroids, start_idx, end_idx, radius):
    tree = cKDTree(centroids)
    chunk_edges = []
    
    for i in range(start_idx, end_idx):
        neighbors = tree.query_ball_point(centroids[i], radius)
        for j in neighbors:
            if i < j:
                chunk_edges.append((i, j))
    
    return chunk_edges


def build_edges_mp(centroids, radius: Union[float, list] = 1.5, n_jobs=-1):
    if len(centroids) == 0:
        return np.empty((0, 3), dtype=np.int64)

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids to be 2D array with 3 columns, got shape {centroids.shape}")
    
    n_centroids = len(centroids)
    
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    
    chunk_size = max(1, n_centroids // n_jobs)
    chunks = [(i, min(i + chunk_size, n_centroids)) for i in range(0, n_centroids, chunk_size)]

    all_edges = []
    
    if isinstance(radius, list):
        for rad in radius:
            results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes=None)(
                delayed(_process_chunk_edges)(centroids, start, end, rad)
                for start, end in chunks
            )
            for chunk_edges in results:
                all_edges.extend(chunk_edges)
    else:
        results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes=None)(
            delayed(_process_chunk_edges)(centroids, start, end, radius)
            for start, end in chunks
        )
        for chunk_edges in results:
            all_edges.extend(chunk_edges)
    
    # Remove duplicates and sort
    all_edges = list(set(all_edges))
    all_edges.sort()
    
    return np.asarray(all_edges, dtype=np.int64)

def edge_labels_binary(edges, sp_tree_ids):
    i = edges[:, 0]
    j = edges[:, 1]
    return (sp_tree_ids[i] == sp_tree_ids[j]).astype(np.float32)


def build_edges_voxelized(centroids: np.ndarray, radius: float = 1.5, voxel_factor: float = 0.7, verbose = False):
    """
    Build edges by voxelizing space and using voxel centroids as query points.
    Similar to superpoint building - voxels help determine search centers.
    
    Args:
        centroids: (N, 3) array of centroid positions
        radius: Distance threshold for edge connections
        voxel_factor: Voxel size as fraction of radius (default: 0.7)
    
    Returns:
        edges: (M, 2) array of edge indices
        voxel_assignments: List of edge indices for each voxel
    """
    voxel_size = radius * voxel_factor
    
    # Build KDTree for all centroids
    tree = cKDTree(centroids)
    
    # Assign each centroid to a voxel
    voxel_coords = (centroids / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    
    # Compute voxel centroids (average position of points in each voxel)
    n_voxels = len(unique_voxels)
    voxel_sums = np.zeros((n_voxels, 3), dtype=np.float32)
    np.add.at(voxel_sums, inverse_indices, centroids)
    counts = np.bincount(inverse_indices, minlength=n_voxels)
    voxel_centroids = voxel_sums / counts[:, None]
    
    # Query ball around each voxel centroid
    neighbors = tree.query_ball_point(voxel_centroids, radius)
    
    # Build edges from neighborhood results
    all_edges = []
    voxel_assignments = []
    

    pbar = enumerate(neighbors)
    if verbose:
        pbar = tqdm(pbar, total=len(neighbors), desc="Voxelized edge building", position=1, leave=False)
    for voxel_idx, neighbor_indices in pbar:
        if len(neighbor_indices) < 2:
            voxel_assignments.append([])
            continue
        
        # Create edges between all pairs in this neighborhood
        neighbor_indices = np.array(neighbor_indices)
        tree_local = cKDTree(centroids[neighbor_indices])
        local_edges = tree_local.query_pairs(radius, output_type='ndarray')
        
        if len(local_edges) > 0:
            global_edges = neighbor_indices[local_edges]
            
            # Track which global edge indices belong to this voxel
            start_idx = len(all_edges)
            all_edges.append(global_edges)
            end_idx = start_idx + len(global_edges)
            
            # Store edge indices range for this voxel (before deduplication)
            voxel_assignments.append(list(range(start_idx, end_idx)))
        else:
            voxel_assignments.append([])
    
    if all_edges:
        edges = np.vstack(all_edges).astype(np.int64)
        
        # Before deduplication, map old indices to new
        edges_sorted = np.sort(edges, axis=1)
        unique_edges, unique_inverse = np.unique(edges_sorted, axis=0, return_inverse=True)
        
        # Update voxel_assignments to point to deduplicated edges
        cumsum = 0
        for i, assignment in enumerate(voxel_assignments):
            if assignment:
                # Map old indices to new unique indices
                old_indices = np.array(assignment)
                new_indices = unique_inverse[old_indices]
                voxel_assignments[i] = np.unique(new_indices).tolist()
        
        edges = unique_edges
    else:
        edges = np.array([], dtype=np.int64).reshape(0, 2)
    
    return edges, voxel_assignments