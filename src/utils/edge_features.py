import numpy as np

def edge_features(sp_a, sp_b):
    d = np.linalg.norm(sp_a.centroid - sp_b.centroid)
    angle = abs(np.dot(sp_a.pca_dir, sp_b.pca_dir))
    thickness_ratio = min(sp_a.thickness, sp_b.thickness) / max(sp_a.thickness, sp_b.thickness)
    vertical_diff = abs(sp_a.verticality - sp_b.verticality)

    return np.array(
        [d, angle, thickness_ratio, vertical_diff],
        dtype=np.float32
    )