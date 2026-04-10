import pyvista as pv
import numpy as np
from typing import Optional

def plot_cloud(points: np.ndarray, labels: Optional[np.ndarray] = None):
    
    cloud = pv.PolyData(points)
    
    # Konfiguracja renderowania w przeglądarce
    pv.set_jupyter_backend('trame')  # Działa też w zwykłych skryptach .py
    
    # Tworzenie plottera
    plotter = pv.Plotter()
    
    if labels is not None:
        # Add cluster labels as scalar data
        cloud['cluster'] = labels
        plotter.add_mesh(
            cloud, 
            scalars='cluster',
            cmap='tab20',  # Good colormap for discrete clusters
            point_size=5, 
            render_points_as_spheres=True
        )
    else:
        # Original behavior - single color
        plotter.add_mesh(
            cloud, 
            color='green', 
            point_size=5, 
            render_points_as_spheres=True
        )
    
    # Wyświetlenie - to otworzy przeglądarkę
    plotter.show()
    
    plotter.close()
    plotter.deep_clean()

# from tqdm import tqdm
# def voxel_subsample_vectorized(xyz, feats, labels, voxel_size=0.25):
#     tqdm.write(f"  Starting voxel subsample: {xyz.shape[0], } pts...")
#     keys     = np.floor(xyz / voxel_size).astype(np.int32)
#     centers  = (keys + 0.5) * voxel_size
#     dists_sq = np.sum((xyz - centers) ** 2, axis=1)

#     keys_min  = keys.min(axis=0)
#     keys      = keys - keys_min
#     key_range = keys.max(axis=0) + 1

#     # explicit Python int — no numpy scalar overflow risk
#     rx, ry, rz = int(key_range[0]), int(key_range[1]), int(key_range[2])
#     assert (ry * rz * rx) < np.iinfo(np.int64).max, "key encoding overflow"

#     key_enc = (keys[:, 0].astype(np.int64) * ry * rz +
#                keys[:, 1].astype(np.int64) * rz +
#                keys[:, 2].astype(np.int64))



#     order      = np.lexsort((dists_sq, key_enc))
#     key_sorted = key_enc[order]
#     _, first   = np.unique(key_sorted, return_index=True)
#     chosen     = order[first]

#     tqdm.write(f"  voxel subsample: {len(xyz):,} → {len(chosen):,} pts")

#     return xyz[chosen], feats[chosen], labels[chosen]

# from sklearn.neighbors import KDTree

# def sample_local_patch(xyz: np.ndarray, labels: np.ndarray, n: int = 8192):
#     seed = np.random.randint(len(xyz))
#     tree = KDTree(xyz)
#     idx = tree.query(xyz[seed:seed+1], k=n, return_distance=False)[0]
#     return idx

# def test():
#     cloud = np.load("/mnt/DATA_SSD/BRIK/SEMANTIC_SEGM/decimated/14-28_tile_010_011.npy")
#     xyz, feats, labels = cloud[:, :3], cloud[:, 3:-1], cloud[:, -1]
#     print(xyz.shape)

#     plot_cloud(xyz, labels)

#     xyz, feats, labels = voxel_subsample_vectorized(xyz, feats, labels, voxel_size=0.3)

#     plot_cloud(xyz, labels)
#     print(xyz.shape)

#     idx = sample_local_patch(xyz, labels, n=16384)
#     new_labels = np.zeros(xyz.shape[0], dtype=np.int32)
#     new_labels[idx] = 1
#     plot_cloud(xyz, new_labels)


if __name__ == "__main__":
    test()