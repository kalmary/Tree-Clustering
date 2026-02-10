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

def edge_features_vectorized(
    edges,
    centroid,
    pca_dir,
    thickness,
    verticality,
    n_points,
    height_extent,
    eps=1e-8
):
    """
    edges: (E, 2) int
    returns: (E, F) float32
    """

    i = edges[:, 0]
    j = edges[:, 1]

    # --- centroids ---
    ci = centroid[i]        # (E, 3)
    cj = centroid[j]        # (E, 3)
    diff = cj - ci          # (E, 3)

    d = np.linalg.norm(diff, axis=1)                     # (E,)
    direction = diff / (d[:, None] + eps)                # (E, 3) if you ever need it

    # --- PCA angle ---
    angle = np.abs(np.sum(pca_dir[i] * pca_dir[j], axis=1))  # (E,)

    # --- ratios ---
    thickness_ratio = np.minimum(thickness[i], thickness[j]) / (
        np.maximum(thickness[i], thickness[j]) + eps
    )

    density_ratio = np.minimum(n_points[i], n_points[j]) / (
        np.maximum(n_points[i], n_points[j]) + eps
    )

    height_ratio = np.minimum(height_extent[i], height_extent[j]) / (
        np.maximum(height_extent[i], height_extent[j]) + eps
    )

    # --- vertical features ---
    vertical_diff = np.abs(verticality[i] - verticality[j])
    vertical_offset = diff[:, 2]

    # --- height ---
    mean_height = (ci[:, 2] + cj[:, 2]) * 0.5

    return np.stack(
        [
            d,
            angle,
            thickness_ratio,
            vertical_diff,
            vertical_offset,
            density_ratio,
            height_ratio,
            mean_height,
        ],
        axis=1
    ).astype(np.float32)
