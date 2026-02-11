import numpy as np

def edge_features(sp_a, sp_b):
    d = np.linalg.norm(sp_a.centroid - sp_b.centroid)
    angle = abs(np.dot(sp_a.pca_dir, sp_b.pca_dir))
    thickness_ratio = min(sp_a.thickness, sp_b.thickness) / max(sp_a.thickness, sp_b.thickness)
    vertical_diff = abs(sp_a.verticality - sp_b.verticality)
    vertical_offset = (sp_b.centroid - sp_a.centroid)[2]
    direction = (sp_b.centroid - sp_a.centroid) / (d + 1e-8)
    density_ratio = min(sp_a.n_points, sp_b.n_points) / max(sp_a.n_points, sp_b.n_points)
    height_ratio = min(sp_a.height_extent, sp_b.height_extent) / max(sp_a.height_extent, sp_b.height_extent)
    mean_height = (sp_a.centroid[2] + sp_b.centroid[2]) / 2



    return np.array(
        [d, angle, thickness_ratio, vertical_diff, vertical_offset, density_ratio, height_ratio, mean_height],
        dtype=np.float32
    )

import numpy as np

def edge_features_vectorized(edges, centroid, pca_dir, thickness, verticality, 
                             linearity, planarity, scattering, eps=1e-8):
    i, j = edges[:, 0], edges[:, 1]
    
    # 1. Edge Geometry
    diff = centroid[j] - centroid[i]
    dist = np.linalg.norm(diff, axis=1)
    unit_diff = diff / (dist[:, None] + eps)
    
    # 2. Alignment: Does the edge vector follow the PCA direction of the nodes?
    # This tells us if Node A and Node B are "stacked" along their growth axis.
    align_i = np.abs(np.sum(unit_diff * pca_dir[i], axis=1))
    align_j = np.abs(np.sum(unit_diff * pca_dir[j], axis=1))
    
    # 3. Shape Consistency
    # Do they have the same "vibe"? (e.g., both are linear)
    lin_diff = np.abs(linearity[i] - linearity[j])
    scat_avg = (scattering[i] + scattering[j]) / 2
    
    # 4. Vertical context (since we have no ground, use absolute Z and delta Z)
    z_min = np.minimum(centroid[i, 2], centroid[j, 2])
    z_diff = np.abs(centroid[i, 2] - centroid[j, 2])

    return np.column_stack([
        dist,               # Distance
        align_i, align_j,   # Directional Alignment
        lin_diff,           # Similarity in shape
        scat_avg,           # How "noisy" the area is
        z_diff,             # Vertical step
        z_min               # Absolute height (still useful for species/scaling)
    ])
