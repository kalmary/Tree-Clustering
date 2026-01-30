import numpy as np

def superpoint_features(points, indices):
    P = points[indices]
    centroid = P.mean(axis=0)

    Q = P - centroid
    _, S, Vt = np.linalg.svd(Q, full_matrices=False)

    pca_dir = Vt[0]
    verticality = abs(np.dot(pca_dir, np.array([0.0, 0.0, 1.0])))
    thickness = np.sqrt(S[1] * S[2])
    bbox_radius = 0.5 * np.linalg.norm(P.max(axis=0) - P.min(axis=0))

    return centroid, pca_dir, thickness, verticality, bbox_radius