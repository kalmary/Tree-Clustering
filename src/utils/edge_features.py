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