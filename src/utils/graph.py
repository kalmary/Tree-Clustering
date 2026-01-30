from typing import Union

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
        return []
    
    centroids = np.asarray(centroids)
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
    
    return all_edges