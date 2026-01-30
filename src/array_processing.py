import numpy as np

from utils.structures import SuperPoint, UnionFind
from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges, build_edges_mp
from utils.edge_features import edge_features
from model_pipeline.affinity import geometric_affinity


def instance_segmentation(points, chunk_id=0, verbose=False, use_mp=False, n_jobs=-1):
    superpoints = []
    sp_indices = []
    
    if use_mp:
        sp_gen = build_superpoints_mp(points, n_jobs=n_jobs)
    else:
        sp_gen = build_superpoints(points)
    
    if verbose:
        from tqdm import tqdm
        sp_gen = tqdm(sp_gen, desc="Building superpoints", leave=False)
    
    for i, idx in enumerate(sp_gen):
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(points, idx)
        superpoints.append(SuperPoint(
            id=i,
            centroid=centroid,
            pca_dir=pca_dir,
            thickness=thickness,
            verticality=verticality,
            n_points=len(idx),
            bbox_radius=bbox_radius,
            chunk_id=chunk_id
        ))
        sp_indices.append(idx)
    
    centroids = np.array([sp.centroid for sp in superpoints])
    if use_mp and len(superpoints) > 1000:
        edges = build_edges_mp(centroids, radius=[0.3], n_jobs=n_jobs)
    else:
        edges = build_edges(centroids, radius=[0.3])

    uf = UnionFind(len(superpoints))

    edge_iterator = edges
    if verbose:
        from tqdm import tqdm
        edge_iterator = tqdm(edges, desc="Processing edges", leave=False)
    
    for i, j in edge_iterator:
        f = edge_features(superpoints[i], superpoints[j])
        score = geometric_affinity(f)
        if score > 0.5:
            uf.union(i, j)

    point_labels = np.zeros(len(points), dtype=int)
    for sp_id, idx in enumerate(sp_indices):
        tree_id = uf.find(sp_id)
        point_labels[idx] = tree_id

    return point_labels


def main():
    import laspy
    cloud = np.load("data/cut/A1N_trees_000000.npy")
    points = cloud[:, :3]  # Use only XYZ coordinates
    labels = cloud[:, -1]

    print("Ground truth labels:", np.unique(labels))
    labels = instance_segmentation(points, verbose=True, use_mp=True)
    print("Predicted labels:", np.unique(labels))

if __name__ == "__main__":
    main()