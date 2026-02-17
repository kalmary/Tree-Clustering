import numpy as np

def superpoint_features(points: np.ndarray, indices: np.ndarray):
    P = points[indices]
    centroid = P.mean(axis=0)
    Q = P - centroid
    
    # S contains singular values: S[0] (main length), S[1] (width), S[2] (thickness)
    _, S, Vt = np.linalg.svd(Q, full_matrices=False)
    
    # Normalized Eigenvalues (Sum to 1)
    s_sum = np.sum(S) + 1e-8
    L1, L2, L3 = S[0]/s_sum, S[1]/s_sum, S[2]/s_sum
    
    # Geometric Descriptors
    linearity = (L1 - L2) / L1    # High for Trunks
    planarity = (L2 - L3) / L1    # High for dense leaf clusters
    scattering = L3 / L1          # High for chaotic twigs/foliage
    
    pca_dir = Vt[0]  # The primary axis of the shape
    verticality = abs(np.dot(pca_dir, np.array([0.0, 0.0, 1.0])))
    thickness = np.sqrt(S[1] * S[2])
    
    # Additional features
    eigenvalue_ratio = L2 / (L1 + 1e-8)
    omnivariance = (L1 * L2 * L3) ** (1/3)
    height_variation = np.std(P[:, 2])
    
    return centroid, pca_dir, thickness, verticality, linearity, planarity, scattering, eigenvalue_ratio, omnivariance, height_variation