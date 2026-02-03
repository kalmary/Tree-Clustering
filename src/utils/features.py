import numpy as np

def superpoint_features(points, indices):
    P = points[indices]
    centroid = P.mean(axis=0)

    Q = P - centroid
    _, S, Vt = np.linalg.svd(Q, full_matrices=False)

    pca_dir = Vt[0]
    verticality = abs(pca_dir[2])
    thickness = np.sqrt(S[1] * S[2])
    bbox_radius = 0.5 * np.linalg.norm(P.max(axis=0) - P.min(axis=0))

    return centroid, pca_dir, thickness, verticality, bbox_radius

def superpoint_features_batch(points, sp_indices_list, tree_ids):
    """Compute features for all superpoints at once, returning arrays."""
    n_sp = len(sp_indices_list)
    
    centroids = np.empty((n_sp, 3), dtype=np.float32)
    pca_dirs = np.empty((n_sp, 3), dtype=np.float32)
    thicknesses = np.empty(n_sp, dtype=np.float32)
    verticalities = np.empty(n_sp, dtype=np.float32)
    bbox_radii = np.empty(n_sp, dtype=np.float32)
    n_points = np.empty(n_sp, dtype=np.int32)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    
    for i, idx in enumerate(sp_indices_list):
        idx = np.asarray(idx, dtype=int)
        P = points[idx]
        
        centroid = P.mean(axis=0)
        Q = P - centroid
        _, S, Vt = np.linalg.svd(Q, full_matrices=False)
        
        pca_dir = Vt[0]
        verticality = abs(pca_dir[2])
        thickness = np.sqrt(S[1] * S[2])
        # bbox_radius = 0.5 * np.linalg.norm(P.max(axis=0) - P.min(axis=0))
        
        centroids[i] = centroid
        pca_dirs[i] = pca_dir
        thicknesses[i] = thickness
        verticalities[i] = verticality
        # bbox_radii[i] = bbox_radius
        # n_points[i] = len(idx)
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()
    
    return centroids, pca_dirs, thicknesses, verticalities, sp_tree_ids