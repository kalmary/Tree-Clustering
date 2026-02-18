import numpy as np
from scipy.spatial import cKDTree


import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from multiprocessing import shared_memory, Lock
from tqdm import tqdm
from typing import Tuple, Optional
import gc

def build_superpoints(points: np.ndarray,
                      chunk: int = 500,
                      radius: float = 0.2,
                      min_pts: int = 30,
                      max_pts: int = 300,
                      max_visits: int = 5,
                      verbose: bool = False):
    
    """Sequential version with density filtering."""
    n_points = len(points)
    tree = cKDTree(points)
    
    # ============================================
    # STEP 1: Batch query all neighborhoods
    # ============================================
    neighbors = []
    if verbose:
        pbar = tqdm(range(0, n_points, chunk), total=len(list(range(0, n_points, chunk))), desc="Querying neighborhoods", leave=False, position=1)
    else:
        pbar = range(0, n_points, chunk)
    
    for i in pbar:
        end_idx = min(i + chunk, n_points)
        batch_points = points[i:end_idx]
        batch_neighbors = tree.query_ball_point(batch_points, radius)
        neighbors.extend(batch_neighbors)

    neighbors = np.asarray(neighbors, dtype=object)
    
    # ============================================
    # STEP 2: Filter by density requirement
    # ============================================
    dense_indices = [i for i in range(n_points) if len(neighbors[i]) >= min_pts]
    
    if len(dense_indices) == 0:
        return [], []
    
    # ============================================
    # STEP 3: Build superpoints from dense points
    # ============================================
    visited = np.zeros(n_points, dtype=np.int32)
    superpoints = []
    seed_indices = []
    
    if verbose:
        pbar = tqdm(dense_indices, desc="Building superpoints", total=len(dense_indices), leave=False, position=1)
    else:
        pbar = dense_indices

    for i in pbar:
        if visited[i] >= max_visits:
            continue
        
        idx = neighbors[i]
        if len(idx) < min_pts:
            continue
        idx = np.array(idx, dtype=np.int32)
        
        valid_mask = visited[idx] < max_visits
        idx = idx[valid_mask]

        if len(idx) > max_pts:
            idx = np.random.choice(idx, size=max_pts, replace=False)
        else:
            idx = np.array(idx, dtype=np.int32)
        
        if len(idx) < min_pts:
            continue

        superpoints.append(idx)
        seed_indices.append(i)
    
    return superpoints, seed_indices


def _process_point_worker(i, neighbors_data, shm_visited_name, n_points, min_pts, max_pts, max_visits, lock = None):
    """Worker function - optimized."""
    # Early exit for insufficient neighbors
    if len(neighbors_data) < min_pts:
        return None, None
    
    idx = np.array(neighbors_data, dtype=np.int32)
    
    # Handle max_visits logic only if max_visits != -1
    if max_visits != -1:
        shm_visited = shared_memory.SharedMemory(name=shm_visited_name)
        visited = np.ndarray(n_points, dtype=np.int32, buffer=shm_visited.buf)
        
        try:
            # Early exit if seed point has been visited too many times
            if visited[i] >= max_visits:
                return None, None
            
            # Filter by visit count
            idx = idx[visited[idx] < max_visits]
            
            if len(idx) < min_pts:
                return None, None
            
            # Downsample if needed
            if len(idx) > max_pts:
                idx = np.random.choice(idx, size=max_pts, replace=False)
            
            # Update visited count (vectorized)
            if lock is not None:
                with lock:
                    visited[idx] += 1
            
            return idx, i  # Return numpy array, not list
        finally:
            shm_visited.close()
    else:
        # No visit tracking - just downsample if needed
        if len(idx) > max_pts:
            idx = np.random.choice(idx, size=max_pts, replace=False)
        
        return idx, i


def build_superpoints_mp(points: np.ndarray,
                      chunk: int = 500,
                      radius: float = 0.2,
                      voxel_factor: float = 0.75,
                      min_pts: int = 30,
                      max_pts: int = 300,
                      max_visits: int = -1,
                      max_superpoints: Optional[int] = 1000,
                      verbose: bool = False,
                      n_jobs: int = -1):
    """
    Parallel version - optimized.
    
    Args:
        ...
        max_superpoints: If set, randomly subsample voxel centroids to this
                         number before querying neighbors. Reduces computation
                         for very dense clouds.
    """
    if n_jobs == -1:
        from joblib import cpu_count
        n_jobs = cpu_count()

    n_points = len(points)
    tree = cKDTree(points)
    
    # ============================================
    # STEP 1: Build voxel centroids
    # ============================================
    voxel_size = radius * voxel_factor 

    coords = (points / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    n_unique_voxels = len(unique_voxels)
    voxel_sums = np.zeros((n_unique_voxels, 3), dtype=np.float32)
    np.add.at(voxel_sums, inverse_indices, points)
    
    counts = np.bincount(inverse_indices, minlength=n_unique_voxels)
    voxel_centroids = voxel_sums / counts[:, None]

    # ============================================
    # STEP 2: Subsample voxel centroids (optional)
    # ============================================
    if max_superpoints is not None and n_unique_voxels > max_superpoints:
        sampled_indices = np.random.choice(n_unique_voxels, size=max_superpoints, replace=False)
        voxel_centroids = voxel_centroids[sampled_indices]
        n_unique_voxels = max_superpoints

    # ============================================
    # STEP 3: Query neighbors
    # ============================================
    neighbors = []
    if verbose:
        pbar = tqdm(range(0, n_unique_voxels, chunk), desc="Querying neighborhoods", leave=False, position=1)
    else:
        pbar = range(0, n_unique_voxels, chunk)

    for i in pbar:
        end_idx = min(i + chunk, n_unique_voxels)
        batch_neighbors = tree.query_ball_point(voxel_centroids[i:end_idx], radius)
        neighbors.extend(batch_neighbors)
    
    # ============================================
    # STEP 4: Filter by density
    # ============================================
    dense_indices = [i for i in range(n_unique_voxels) if len(neighbors[i]) >= min_pts]
    
    if len(dense_indices) == 0:
        return [], []
    
    # ============================================
    # STEP 5: Create shared memory (only if max_visits != -1)
    # ============================================
    shm_visited = None
    visited = None
    lock = None
    if max_visits != -1:
        shm_visited = shared_memory.SharedMemory(
            create=True, 
            size=n_points * np.dtype(np.int32).itemsize
        )
        visited = np.ndarray(n_points, dtype=np.int32, buffer=shm_visited.buf)
        visited[:] = 0
        lock = Lock()
    
    try:
        # ============================================
        # STEP 6: Process in parallel
        # ============================================
        shm_name = shm_visited.name if max_visits != -1 else None
        
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(_process_point_worker)(
                i, 
                neighbors[i],
                shm_name,
                n_points, 
                min_pts, 
                max_pts, 
                max_visits,
                lock
            )
            for i in (tqdm(dense_indices, desc="Building superpoints", leave=False, position=1) if verbose else dense_indices)
        )
        
        # ============================================
        # STEP 7: Collect results
        # ============================================
        superpoints = []
        seed_indices = []
        
        for sp, seed in results:
            if sp is not None:
                superpoints.append(sp)
                seed_indices.append(seed)
        
        return superpoints, seed_indices
    
    finally:
        visited = None
        if max_visits != -1 and shm_visited is not None:
            shm_visited.close()
            shm_visited.unlink()
        gc.collect()