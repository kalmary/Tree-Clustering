from typing import Union, List
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

def build_edges(centroids, radius: Union[float, list] = 1.5):
    if len(centroids) == 0:
        return []
    
    centroids = np.asarray(centroids)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids to be 2D array with 3 columns, got shape {centroids.shape}")
    
    tree = cKDTree(centroids)
    edges = []
    
    if isinstance(radius, list):
        for rad in radius:
            for i, c in enumerate(centroids):
                neighbors = tree.query_ball_point(c, rad)
                for j in neighbors:
                    if i < j:
                        edges.append((i, j))
    else:
        for i, c in enumerate(centroids):
            neighbors = tree.query_ball_point(c, radius)
            for j in neighbors:
                if i < j:
                    edges.append((i, j))
    
    # Remove duplicates and sort
    edges = list(set(edges))
    edges.sort()
    return edges


def _process_chunk_edges(
    centroids: np.ndarray,
    radius: float,
    start_idx: int,
    end_idx: int
):
    tree = cKDTree(centroids)
    radius_sq = radius * radius  # Avoid sqrt in distance comparison
    chunk_edges = []
    
    for i in range(start_idx, end_idx):
        neighbors = tree.query_ball_point(centroids[i], radius)
        
        # Vectorized distance calculation for all neighbors at once
        if len(neighbors) > 1:
            neighbor_coords = centroids[neighbors]
            diffs = neighbor_coords - centroids[i]
            dists_sq = np.sum(diffs * diffs, axis=1)
            
            # Filter neighbors: j > i and within radius
            for idx, j in enumerate(neighbors):
                if j > i and dists_sq[idx] <= radius_sq:
                    chunk_edges.append((i, j))
    
    return chunk_edges


def build_edges_mp(
    centroids: np.ndarray,
    radius: float = 1.5,
    n_jobs: int = -1
):
    """
    Build edges between centroids within a radius (optimized).
    
    Args:
        centroids: (N, 3) array
        radius: float, distance threshold for edge creation
        n_jobs: number of parallel jobs
    
    Returns:
        edges: numpy array of shape (M, 2) with edge indices
    """
    centroids = np.asarray(centroids, dtype=np.float32)  # Use float32 to save memory
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape (N, 3)")
    
    radius = float(radius)
    
    n_centroids = len(centroids)
    if n_centroids == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    
    # Optimize chunk size for better load balancing
    n_jobs = min(n_jobs, n_centroids)
    chunk_size = max(100, n_centroids // (n_jobs * 4))  # Smaller chunks for better balance
    chunks = [
        (i, min(i + chunk_size, n_centroids))
        for i in range(0, n_centroids, chunk_size)
    ]
    
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes=None
    )(
        delayed(_process_chunk_edges)(
            centroids,
            radius,
            start,
            end
        )
        for start, end in chunks
    )
    
    # Concatenate all edges directly into numpy array
    if not results or all(len(r) == 0 for r in results):
        return np.empty((0, 2), dtype=np.int32)
    
    # Flatten and convert to numpy array in one go
    total_edges = sum(len(r) for r in results)
    edges = np.empty((total_edges, 2), dtype=np.int32)
    
    idx = 0
    for chunk_edges in results:
        n = len(chunk_edges)
        if n > 0:
            edges[idx:idx+n] = chunk_edges
            idx += n
    
    # Remove duplicates efficiently using numpy
    edges = np.unique(edges, axis=0)
    
    return edges


def build_edges_multi_radius(
    centroids: np.ndarray,
    radius: Union[float, List[float]] = 1.5,
    use_mp: bool = True,
    n_jobs: int = -1
):
    """
    Build graph edges separately for each radius using existing build_edges_mp.

    Yields:
        (edges, radius_value) for each radius
    """
    if isinstance(radius, (float, int)):
        radii = [float(radius)]
    else:
        radii = list(radius)

    for r in radii:
        if use_mp and len(centroids) > 1000:
            edges = build_edges_mp(
                centroids,
                radius=r,
                n_jobs=n_jobs
            )
        else:
            edges = build_edges(
                centroids,
                radius=r
            )

        yield edges, r
