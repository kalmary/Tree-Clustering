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
    
    all_superpoints.sort(key=lambda x: x[0])
    
    global_visited = set()
    final_superpoints = []
    for seed_idx, idx in all_superpoints:
        idx_set = set(idx)
        if not idx_set.intersection(global_visited):
            global_visited.update(idx_set)
            final_superpoints.append(idx)
    
    return final_superpoints



def build_superpoints2(points, radius=0.2, min_pts=30, max_pts=300, max_overlap=3, 
                      min_density=0.5, n_jobs=-1):
    """
    Build superpoints with guaranteed coverage and controlled overlap.
    Uses density-based filtering instead of linearity.
    
    Args:
        points: (N, 3) array of point coordinates
        radius: Search radius for neighbors
        min_pts: Minimum points in a superpoint
        max_pts: Maximum points in a superpoint
        max_overlap: Maximum times a point can appear in different superpoints
        min_density: Minimum points per unit volume (points/radius³)
        n_jobs: Number of parallel jobs (-1 = all cores)
    
    Returns:
        superpoints: List of tuples (seed_idx, point_indices)
        point_membership: Array tracking superpoint membership count per point
    """
    n_points = len(points)
    use_parallel = n_points > 1000
    n_jobs_actual = n_jobs if use_parallel else 1
    
    # Calculate expected volume and minimum point threshold
    sphere_volume = (4/3) * np.pi * (radius ** 3)
    density_threshold = min_density * sphere_volume
    
    # Phase 1: Generate candidates in parallel
    def generate_candidate(i, pts, rad, min_p, dens_thresh):
        tree = cKDTree(pts)
        idx = tree.query_ball_point(pts[i], rad)
        if len(idx) < min_p:
            return None
        
        # Density check: reject sparse regions
        density = len(idx) / ((4/3) * np.pi * (rad ** 3))
        if density < min_density:
            return None
        
        return (i, idx)
    
    if use_parallel:
        candidates = Parallel(n_jobs=n_jobs_actual, batch_size='auto')(
            delayed(generate_candidate)(i, points, radius, min_pts, density_threshold) 
            for i in range(n_points)
        )
    else:
        tree = cKDTree(points)
        candidates = []
        for i in range(n_points):
            idx = tree.query_ball_point(points[i], radius)
            if len(idx) < min_pts:
                continue
            
            # Density check
            density = len(idx) / sphere_volume
            if density < min_density:
                continue
            
            candidates.append((i, idx))
    
    candidates = [c for c in candidates if c is not None]
    
    # Phase 2: Select superpoints with coverage-based point selection
    superpoints = []
    point_membership = np.zeros(n_points, dtype=int)
    
    for seed_idx, idx in candidates:
        if point_membership[seed_idx] >= max_overlap:
            continue
        
        idx_array = np.array(idx)
        
        # Coverage-based selection: prioritize points with lower membership
        if len(idx_array) > max_pts:
            memberships = point_membership[idx_array]
            priority = np.argsort(memberships)[:max_pts]
            idx_array = idx_array[priority]
        
        # Filter by overlap constraint
        valid_mask = point_membership[idx_array] < max_overlap
        valid_idx = idx_array[valid_mask]
        
        if len(valid_idx) < min_pts:
            continue
        
        point_membership[valid_idx] += 1
        superpoints.append((seed_idx, valid_idx.tolist()))
    
    # Phase 3: Handle uncovered points
    uncovered = np.where(point_membership == 0)[0]
    
    if len(uncovered) > 0:
        tree = cKDTree(points)
        
        # Build inverse index for fast lookup
        point_to_superpoints = [[] for _ in range(n_points)]
        for sp_idx, (seed, sp_points) in enumerate(superpoints):
            for pt in sp_points:
                point_to_superpoints[pt].append(sp_idx)
        
        superpoints_as_sets = [set(sp_points) for seed, sp_points in superpoints]
        
        def find_assignment(i, pts, rad, min_p):
            tree_local = cKDTree(pts)
            dists, neighbors = tree_local.query(pts[i], k=min(100, len(pts)))
            
            for neighbor in neighbors:
                sp_indices = point_to_superpoints[neighbor]
                if sp_indices:
                    for sp_idx in sp_indices:
                        seed, sp_points = superpoints[sp_idx]
                        if len(sp_points) < max_pts:
                            return (i, 'append', sp_idx, neighbor)
                    break
            
            idx = tree_local.query_ball_point(pts[i], rad)
            if len(idx) == 0:
                dists, idx = tree_local.query(pts[i], k=min(min_p, len(pts)))
                idx = idx.tolist()
            else:
                idx_array = np.array(idx)
                if len(idx_array) > max_pts:
                    memberships = point_membership[idx_array]
                    priority = np.argsort(memberships)[:max_pts]
                    idx = idx_array[priority].tolist()
            
            return (i, 'create', idx, None)
        
        if use_parallel and len(uncovered) > 100:
            assignments = Parallel(n_jobs=n_jobs_actual, batch_size='auto')(
                delayed(find_assignment)(i, points, radius, min_pts) for i in uncovered
            )
        else:
            assignments = [find_assignment(i, points, radius, min_pts) for i in uncovered]
        
        for assignment in assignments:
            i, action, data, extra = assignment
            
            if action == 'append':
                sp_idx = data
                seed, sp_points = superpoints[sp_idx]
                if len(sp_points) < max_pts and i not in superpoints_as_sets[sp_idx]:
                    sp_points.append(i)
                    superpoints_as_sets[sp_idx].add(i)
                    point_membership[i] = 1
            
            elif action == 'create':
                idx = data
                superpoints.append((i, idx))
                superpoints_as_sets.append(set(idx))
                point_membership[idx] += 1
    
    return superpoints, point_membership