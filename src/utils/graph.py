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


def build_edges(centroids: np.ndarray,
                radius: float = 1.5,
                voxel_factor: float = 0.78,
                tight_factor: float = 0.3,
                verbose: bool = False) -> tuple[np.ndarray, list]:
    """
    Build edges between superpoint centroids using a two-radius strategy,
    then split into voxel-local subgraphs.

    Two-radius strategy:
        tight_radius = radius * tight_factor
        1. Build tight edges (within tight_radius) — short-range, same-tree biased.
        2. Any superpoint with zero tight connections is considered isolated.
        3. Isolated superpoints get fallback edges from the full radius pool,
           but only edges where at least one endpoint is isolated.
        4. Final edge set = tight + fallback (deduplicated).

    Voxel splitting:
        Voxel size = radius * voxel_factor.
        Centroids quantized with np.round — symmetric around zero.
        Anchor per voxel = mean of member centroids snapped to nearest real SP.
        All tree queries are batched and parallelised (workers=-1).

    Args:
        centroids:    (n_sp, 3) superpoint centroid array.
        radius:       Full neighborhood radius in metres.
        voxel_factor: Voxel size = radius * voxel_factor.
        tight_factor: Tight edge radius = radius * tight_factor.
        verbose:      Show tqdm progress bar with edge stats.

    Returns:
        edges:             (E, 2) int64 array of superpoint index pairs.
        voxel_assignments: list of lists; voxel_assignments[v] = edge indices
                           belonging to voxel v.
    """
    tree         = cKDTree(centroids)
    tight_radius = radius * tight_factor
    voxel_size   = radius * voxel_factor
    n_sp         = len(centroids)

    # --- two-radius edge set ---
    tight_pairs = tree.query_pairs(tight_radius, output_type='ndarray')
    all_pairs   = tree.query_pairs(radius,        output_type='ndarray')

    connected    = np.unique(tight_pairs) if len(tight_pairs) > 0 else np.array([], dtype=np.int64)
    isolated     = np.setdiff1d(np.arange(n_sp), connected)
    isolated_set = np.zeros(n_sp, dtype=bool)
    isolated_set[isolated] = True

    if len(all_pairs) > 0:
        fallback_mask  = isolated_set[all_pairs[:, 0]] | isolated_set[all_pairs[:, 1]]
        fallback_pairs = all_pairs[fallback_mask]
    else:
        fallback_pairs = np.empty((0, 2), dtype=np.int64)

    if len(tight_pairs) > 0 and len(fallback_pairs) > 0:
        edges = np.unique(np.vstack([tight_pairs, fallback_pairs]), axis=0).astype(np.int64)
    elif len(tight_pairs) > 0:
        edges = tight_pairs.astype(np.int64)
    elif len(fallback_pairs) > 0:
        edges = fallback_pairs.astype(np.int64)
    else:
        return np.empty((0, 2), dtype=np.int64), []

    # --- anchor computation: fully vectorized, one batched query ---
    voxel_coords           = np.round(centroids / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    n_voxels               = len(unique_voxels)

    voxel_sums   = np.zeros((n_voxels, 3), dtype=np.float64)
    voxel_counts = np.bincount(inverse, minlength=n_voxels)
    np.add.at(voxel_sums, inverse, centroids.astype(np.float64))
    voxel_means  = (voxel_sums / voxel_counts[:, None]).astype(np.float32)

    _, anchor_idx = tree.query(voxel_means, k=1, workers=-1)

    # --- voxel assignment: batched query_ball_point ---
    neighborhoods = tree.query_ball_point(centroids[anchor_idx], r=radius, workers=-1)

    edge_i            = edges[:, 0]
    edge_j            = edges[:, 1]
    in_neighborhood   = np.zeros(n_sp, dtype=bool)
    voxel_assignments = []

    iterator = tqdm(neighborhoods,
                    desc="Assigning voxels",
                    leave=False,
                    position=1,
                    disable=not verbose,
                    postfix={
                        "tight":       len(tight_pairs),
                        "fallback":    len(fallback_pairs),
                        "isolated":    len(isolated),
                        "total_edges": len(edges),
                    })

    for nbrs in iterator:
        nbrs = np.asarray(nbrs, dtype=np.int64)
        in_neighborhood[nbrs] = True
        mask = in_neighborhood[edge_i] & in_neighborhood[edge_j]
        voxel_assignments.append(np.where(mask)[0].tolist())
        in_neighborhood[nbrs] = False

    return edges, voxel_assignments

def edge_labels_binary(edges, sp_tree_ids):
    i = edges[:, 0]
    j = edges[:, 1]
    return (sp_tree_ids[i] == sp_tree_ids[j]).astype(np.float32)