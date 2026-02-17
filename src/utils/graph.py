import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed, cpu_count
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List, Tuple

from tqdm import tqdm

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def _process_voxel_worker(voxel_idx, nbrs, shm_name, edges_shape, edges_dtype):
    if len(nbrs) < 2:
        return voxel_idx, []
    
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        edges = np.ndarray(edges_shape, dtype=edges_dtype, buffer=shm.buf)
        mask = np.isin(edges[:, 0], nbrs) & np.isin(edges[:, 1], nbrs)
        result = np.where(mask)[0].tolist()
    finally:
        shm.close()  # always release worker handle

    return voxel_idx, result


def build_edges_mp(centroids, radius, voxel_factor, verbose=False):
    voxel_size = radius * voxel_factor

    # --- Assign each superpoint to one voxel ---
    voxel_coords = (centroids / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    n_voxels = len(unique_voxels)

    # --- Voxel centroid = mean of member superpoint coords ---
    counts = np.bincount(inverse_indices, minlength=n_voxels)
    voxel_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(voxel_sums, inverse_indices, centroids)
    voxel_centroids = voxel_sums / counts[:, None]

    # --- Build ALL edges once globally ---
    tree = cKDTree(centroids)
    all_pairs = tree.query_pairs(radius, output_type='ndarray')

    if len(all_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64), [[] for _ in range(n_voxels)]

    edges = all_pairs.astype(np.int64)

    # --- Batched neighbor query ---
    all_neighbors = tree.query_ball_point(voxel_centroids, radius)

    # --- Put edges in shared memory ---
    shm = shared_memory.SharedMemory(create=True, size=edges.nbytes)
    shm_edges = np.ndarray(edges.shape, dtype=edges.dtype, buffer=shm.buf)
    shm_edges[:] = edges

    try:
        pbar = range(n_voxels)
        if verbose:
            pbar = tqdm(pbar, desc="Voxelized edge building", total=n_voxels, position=1, leave=False)

        results = Parallel(n_jobs=mp.cpu_count(), backend='multiprocessing')(
            delayed(_process_voxel_worker)(
                voxel_idx,
                all_neighbors[voxel_idx],
                shm.name,
                edges.shape,
                edges.dtype
            )
            for voxel_idx in pbar
        )
    finally:
        shm.close()   # main process releases its handle
        shm.unlink()  # destroy the block — always, even on exception

    voxel_assignments = [[] for _ in range(n_voxels)]
    for voxel_idx, assignment in results:
        voxel_assignments[voxel_idx] = assignment

    return edges, voxel_assignments


def build_edges(centroids, radius, voxel_factor, verbose=False):
    voxel_size = radius * voxel_factor

    voxel_coords = (centroids / voxel_size).astype(np.int32)
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    n_voxels = len(unique_voxels)

    counts = np.bincount(inverse_indices, minlength=n_voxels)
    voxel_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(voxel_sums, inverse_indices, centroids)
    voxel_centroids = voxel_sums / counts[:, None]

    tree = cKDTree(centroids)
    all_pairs = tree.query_pairs(radius, output_type='ndarray')

    if len(all_pairs) == 0:
        return np.empty((0, 2), dtype=np.int64), [[] for _ in range(n_voxels)]

    edges = all_pairs.astype(np.int64)

    # Hoist out of loop — same arrays every iteration
    edge_i = edges[:, 0]
    edge_j = edges[:, 1]

    all_neighbors = tree.query_ball_point(voxel_centroids, radius)

    # Single allocation, reused every iteration
    in_neighborhood = np.zeros(len(centroids), dtype=bool)

    voxel_assignments = [[] for _ in range(n_voxels)]

    pbar = enumerate(all_neighbors)
    if verbose:
        pbar = tqdm(pbar, desc="Voxelized edge building", total=n_voxels, position=1, leave=False)

    for voxel_idx, nbrs in pbar:
        if len(nbrs) < 2:
            continue

        nbrs = np.asarray(nbrs)
        in_neighborhood[nbrs] = True

        mask = in_neighborhood[edge_i] & in_neighborhood[edge_j]
        voxel_assignments[voxel_idx] = np.where(mask)[0].tolist()

        # Reset only the touched indices, not the whole array
        in_neighborhood[nbrs] = False

    return edges, voxel_assignments

def edge_labels_binary(edges, sp_tree_ids):
    i = edges[:, 0]
    j = edges[:, 1]
    return (sp_tree_ids[i] == sp_tree_ids[j]).astype(np.float32)