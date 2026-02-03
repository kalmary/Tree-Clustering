import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

def build_superpoints(points, radius=0.2, min_pts=30, max_pts=300):
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)

    for i in range(len(points)):
        if visited[i]:
            continue

        idx = tree.query_ball_point(points[i], radius)
        if len(idx) < min_pts:
            continue

        P = points[idx] - points[idx].mean(axis=0)
        _, S, _ = np.linalg.svd(P, full_matrices=False)

        linearity = S[0] / (S[1] + 1e-6)
        if linearity < 1.5:
            continue

        idx = idx[:max_pts]
        visited[idx] = True

        yield idx


def _process_chunk(points, tree, start_idx, end_idx, radius, min_pts, max_pts):
    chunk_superpoints = []
    visited_local = set()
    
    for i in range(start_idx, end_idx):
        if i in visited_local:
            continue
        
        idx = tree.query_ball_point(points[i], radius)
        if len(idx) < min_pts:
            continue
        
        P = points[idx] - points[idx].mean(axis=0)
        _, S, _ = np.linalg.svd(P, full_matrices=False)
        
        linearity = S[0] / (S[1] + 1e-6)
        if linearity < 1.5:
            continue
        
        idx = idx[:max_pts]
        visited_local.update(idx)
        chunk_superpoints.append((i, idx))
    
    return chunk_superpoints


def build_superpoints_mp(points, radius=0.2, min_pts=30, max_pts=300, n_jobs=-1):
    n_points = len(points)
    
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    
    tree = cKDTree(points)
    chunk_size = max(1, n_points // n_jobs)
    chunks = [(i, min(i + chunk_size, n_points)) for i in range(0, n_points, chunk_size)]
    
    results = Parallel(n_jobs=n_jobs, backend='loky', max_nbytes=None)(
        delayed(_process_chunk)(points, tree, start, end, radius, min_pts, max_pts)
        for start, end in chunks
    )
    
    all_superpoints = []
    for chunk_sps in results:
        all_superpoints.extend(chunk_sps)
    
    # all_superpoints.sort(key=lambda x: x[0])
    
    global_visited = set()
    final_superpoints = []
    for seed_idx, idx in all_superpoints:
        idx_set = set(idx)
        if not idx_set.intersection(global_visited):
            global_visited.update(idx_set)
            final_superpoints.append(idx)
    
    return final_superpoints