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
    diff        = centroid[j] - centroid[i]
    dist        = np.linalg.norm(diff, axis=1)
    unit_diff   = diff / (dist[:, None] + eps)

    # 2. Directional alignment with PCA axis
    align_i = np.abs(np.sum(unit_diff * pca_dir[i], axis=1))
    align_j = np.abs(np.sum(unit_diff * pca_dir[j], axis=1))

    # 3. Shape consistency
    lin_diff  = np.abs(linearity[i]  - linearity[j])
    scat_avg  = (scattering[i] + scattering[j]) / 2

    # 4. Vertical context
    z_min  = np.minimum(centroid[i, 2], centroid[j, 2])
    z_diff = np.abs(centroid[i, 2] - centroid[j, 2])

    # 5. Horizontal vs vertical decomposition
    horiz_dist = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    h_v_ratio  = horiz_dist / (z_diff + eps)  # low = vertical = likely trunk

    # 6. Planarity — crown boundary edges connect two planar SPs
    plan_diff = np.abs(planarity[i] - planarity[j])
    plan_avg  = (planarity[i] + planarity[j]) / 2

    # 7. Thickness difference — same-tree SPs have gradual thickness change
    thick_diff = np.abs(thickness[i] - thickness[j])

    # 8. Verticality product — both vertical = likely same trunk segment
    vert_product = verticality[i] * verticality[j]

    return np.column_stack([
        dist,                    # 1  distance
        align_i, align_j,        # 2  directional alignment src, dst
        lin_diff,                # 4  linearity difference
        scat_avg,                # 5  average scattering
        z_diff,                  # 6  vertical step
        z_min,                   # 7  absolute height
        horiz_dist,              # 8  horizontal distance
        h_v_ratio,               # 9  horizontal/vertical ratio
        plan_diff,               # 10 planarity difference
        plan_avg,                # 11 average planarity
        thick_diff,              # 12 thickness difference
        vert_product,            # 13 verticality product
    ])