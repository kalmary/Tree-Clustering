import argparse
import gc
import json
import os
import pathlib as pth
import shutil
import tempfile
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from tqdm import tqdm

import raycloud_tools

if __package__:
    from .utils import cloud_geometry, cluster_processing, raycloud_io, wood_leaf
    from .utils.get_rays import get_las_ray_inputs, get_rays
else:
    from utils import cloud_geometry, cluster_processing, raycloud_io, wood_leaf
    from utils.get_rays import get_las_ray_inputs, get_rays

# @dataclass
# class TreeSegmRayConfig:
#     height_min:          float         = 2.0
#     max_diameter:        float         = 0.9
#     crop_length:         float         = 1.0
#     distance_limit:      float         = 0.3
#     girth_height_ratio:  float         = 0.12
#     gravity_factor:      float         = 0.75
#     global_taper:        Optional[float] = None # all below no need to change
#     global_taper_factor: Optional[float] = None
#     grid_width:          Optional[float] = None
#     use_rays:            bool          = False
#     ground_label:        Optional[int] = None
#     tree_label:          Optional[int] = None


class TreeSegmRay:
    def __init__(
        self,
        height_min: float = 2.0,
        max_diameter: float = 0.9,
        crop_length: float = 1.0,
        distance_limit: float = 0.3,
        girth_height_ratio: float = 0.12,
        gravity_factor: float = 0.75,
        global_taper: float | None = None,
        global_taper_factor: float | None = None,
        grid_width: float | None = None,
        use_rays: bool = False,
        ground_label: int | None = None,
        tree_label: int | None = None,
        verbose: bool = False,
        wood_distance: float = 0.02,
        wood_propagation_height: float = 1.35,
        wood_propagation_radius: float = 0.1,
        wood_propagation_min_points: int = 3,
    ):
        if not np.isfinite(wood_distance) or wood_distance < 0:
            raise ValueError("wood_distance must be finite and non-negative")
        if not np.isfinite(wood_propagation_height) or wood_propagation_height < 0:
            raise ValueError("wood_propagation_height must be finite and non-negative")
        if not np.isfinite(wood_propagation_radius) or wood_propagation_radius <= 0:
            raise ValueError("wood_propagation_radius must be finite and positive")
        if (
            isinstance(wood_propagation_min_points, (bool, np.bool_))
            or not isinstance(wood_propagation_min_points, (int, np.integer))
            or wood_propagation_min_points < 1
        ):
            raise ValueError("wood_propagation_min_points must be a positive integer")
        self.verbose = verbose
        self.height_min = height_min
        self.max_diameter = max_diameter
        self.crop_length = crop_length
        self.distance_limit = distance_limit
        self.girth_height_ratio = girth_height_ratio
        self.gravity_factor = gravity_factor
        self.global_taper = global_taper
        self.global_taper_factor = global_taper_factor
        self.grid_width = grid_width
        self.use_rays = use_rays
        self.tree_label = tree_label
        self.ground_label = ground_label
        self.wood_distance = wood_distance
        self.wood_propagation_height = wood_propagation_height
        self.wood_propagation_radius = wood_propagation_radius
        self.wood_propagation_min_points = wood_propagation_min_points

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any] | None = None,
        cfg_path: str | pth.Path | None = None,
        verbose: bool = False,
    ) -> "TreeSegmRay":
        if cfg is not None:
            return cls(**cfg, verbose=verbose)
        elif cfg is None and cfg_path is not None:
            cfg_path = pth.Path(cfg_path)
            with open(cfg_path, "r") as f:
                cfg = cast(dict[str, Any], json.load(f))
            return cls(**cfg, verbose=verbose)
        else:
            raise ValueError("Either cfg or cfg_path must be provided.")

    # ------------------------------------------------------------------
    # PLY helpers
    # ------------------------------------------------------------------

    _write_raycloud_ply = staticmethod(raycloud_io.write_raycloud_ply)

    _write_ground_mesh_ply = staticmethod(raycloud_io.write_ground_mesh_ply)

    _read_labels_from_segmented_ply = staticmethod(
        raycloud_io.read_labels_from_segmented_ply
    )

    _segment_wood_leaf = staticmethod(wood_leaf.segment_wood_leaf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_floating_clusters(
        self,
        tree_labels: np.ndarray,
        tree_xyz: np.ndarray,
        ground_xyz: np.ndarray,
        ground_z_threshold: float = 0.5,
        min_cluster_size: int = 5000,
        max_tilt_deg: float = 30.0,
    ) -> np.ndarray:
        return cluster_processing.connect_floating_clusters(
            tree_labels,
            tree_xyz,
            ground_xyz,
            ground_z_threshold,
            min_cluster_size,
            max_tilt_deg,
        )

    def _reduce_labels(self, labels: NDArray) -> NDArray:
        return cluster_processing.reduce_labels(labels)

    def _remove_small_clusters(
        self, tree_labels: np.ndarray, min_points: int = 100
    ) -> np.ndarray:
        return cluster_processing.remove_small_clusters(tree_labels, min_points)

    _connected_components_voxel = staticmethod(
        cloud_geometry.connected_components_voxel
    )

    _segment_xy_connected_components = staticmethod(
        cloud_geometry.segment_xy_connected_components
    )

    @classmethod
    def _filter_floating_clusters(
        cls,
        xyz: NDArray[np.float32],
        ground_xyz: NDArray[np.float32] | None = None,
        voxel_size: float = 0.5,
        max_gap: float = 0.4,
        boundary_margin: float = 0.5,
        nearest_ground_points: int = 10,
    ) -> NDArray[np.bool_]:
        """Return a point mask that excludes components floating above ground."""
        if len(xyz) == 0:
            return np.zeros(0, dtype=bool)
        if boundary_margin < 0:
            raise ValueError("boundary_margin must not be negative")
        if nearest_ground_points < 1:
            raise ValueError("nearest_ground_points must be at least 1")

        component_labels, min_heights, xy_bounds = cls._connected_components_voxel(
            xyz,
            voxel_size=voxel_size,
        )
        ground_gaps = np.empty(len(min_heights), dtype=np.float32)
        if ground_xyz is not None and len(ground_xyz) > 0:
            ground_tree = KDTree(ground_xyz[:, :2])
            for cluster_id, bounds in enumerate(xy_bounds):
                lower = bounds[:2] - boundary_margin
                upper = bounds[2:] + boundary_margin
                centre = (lower + upper) / 2.0
                radius = float(np.linalg.norm((upper - lower) / 2.0))
                nearby_indices = ground_tree.query_ball_point(centre, r=radius)
                if nearby_indices:
                    nearby_xy = ground_xyz[nearby_indices, :2]
                    inside_box = np.all(
                        (nearby_xy >= lower) & (nearby_xy <= upper),
                        axis=1,
                    )
                    box_indices = np.asarray(nearby_indices)[inside_box]
                else:
                    box_indices = np.zeros(0, dtype=np.int64)

                if len(box_indices) > 0:
                    ground_level = float(ground_xyz[box_indices, 2].mean())
                else:
                    neighbour_count = min(nearest_ground_points, len(ground_xyz))
                    _, nearest_indices = ground_tree.query(
                        centre,
                        k=neighbour_count,
                    )
                    nearest_indices = np.atleast_1d(nearest_indices)
                    ground_level = float(ground_xyz[nearest_indices, 2].mean())
                ground_gaps[cluster_id] = min_heights[cluster_id] - ground_level
        else:
            ground_gaps = min_heights - min_heights.min()

        grounded_components = ground_gaps <= max_gap
        filtered_component_labels = component_labels.copy()
        filtered_component_labels[~grounded_components[component_labels]] = -1
        retained_mask = filtered_component_labels >= 0

        return retained_mask

    _remove_partial_trunks = staticmethod(cloud_geometry.remove_partial_trunks)

    _estimate_ground = staticmethod(cloud_geometry.estimate_ground)

    # ------------------------------------------------------------------
    # Tile helpers
    # ------------------------------------------------------------------

    _voxel_tiles = staticmethod(cloud_geometry.voxel_tiles)

    _core_mask = staticmethod(cloud_geometry.core_mask)

    _tile_mask = staticmethod(cloud_geometry.tile_mask)

    _estimate_trunk_position = staticmethod(cloud_geometry.estimate_trunk_position)

    def _merge_close_trunks(
        self,
        tree_xyz: np.ndarray,
        tree_ids: np.ndarray,
        min_trunk_dist: float = 1.5,
        trunk_height_band: tuple[float, float] = (0.5, 2.0),
        min_points: int = 50,
    ) -> np.ndarray:

        unique_ids = np.unique(tree_ids)
        unique_ids = unique_ids[unique_ids >= 0]
        if len(unique_ids) < 2:
            return tree_ids.copy()

        # sort by tree_id for O(1) per-tree slicing instead of O(n) masking
        sort_idx = np.argsort(tree_ids, kind="stable")
        sorted_ids = tree_ids[sort_idx]
        sorted_xyz = tree_xyz[sort_idx]

        # find start index of each unique id in the sorted array
        boundaries = np.searchsorted(sorted_ids, unique_ids)

        trunk_xy = {}
        point_count = {}

        for i, tid in enumerate(unique_ids):
            start = int(boundaries[i])
            end = int(boundaries[i + 1]) if i + 1 < len(unique_ids) else len(sorted_ids)
            pts = sorted_xyz[start:end]
            count = end - start
            point_count[tid] = count
            trunk_xy[tid] = (
                self._estimate_trunk_position(pts, trunk_height_band)
                if count >= min_points
                else None
            )

        valid_ids = [tid for tid in unique_ids if trunk_xy[tid] is not None]
        if len(valid_ids) < 2:
            return tree_ids.copy()

        positions = np.array([trunk_xy[tid] for tid in valid_ids], dtype=np.float64)
        pairs = KDTree(positions).query_pairs(r=min_trunk_dist, output_type="ndarray")
        if len(pairs) == 0:
            return tree_ids.copy()

        parent = list(range(len(valid_ids)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if point_count[valid_ids[ra]] >= point_count[valid_ids[rb]]:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i, j in pairs:
            union(int(i), int(j))

        # vectorized remap: build lookup array indexed by tree id
        max_id = int(unique_ids.max())
        remap = np.arange(max_id + 1, dtype=np.int64)
        for idx, tid in enumerate(valid_ids):
            remap[tid] = valid_ids[find(idx)]

        new_ids = tree_ids.copy()
        valid_mask = new_ids >= 0
        new_ids[valid_mask] = remap[new_ids[valid_mask]]

        # re-index contiguously, preserving -1
        noise_mask = new_ids == -1
        _, new_ids[~noise_mask] = np.unique(new_ids[~noise_mask], return_inverse=True)
        new_ids[noise_mask] = -1

        return new_ids.astype(np.int64)

    def _segment_small(
        self,
        xyz: NDArray,
        labels: NDArray | None = None,
        rays: NDArray | None = None,
        debug: bool = False,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int8]]:

        xyz -= xyz.mean(
            axis=0
        )  # center for better numerical stability in raycloudtools

        if (
            labels is not None
            and self.tree_label is not None
            and self.ground_label is not None
        ):
            tree_mask = labels == self.tree_label
            ground_mask = labels == self.ground_label
            tree_xyz = xyz[tree_mask].copy()
            ground_xyz = xyz[ground_mask].copy()
        else:
            tree_xyz = xyz.copy()
            ground_xyz = None
            tree_mask = np.ones(len(xyz), dtype=bool)

        tree_rays = rays[tree_mask].copy() if rays is not None else None

        if tree_xyz.shape[0] == 0:
            empty_ids = np.zeros(0, dtype=np.int32)
            return empty_ids, empty_ids.copy(), np.full(0, -1, dtype=np.int8)

        if debug:
            if ground_xyz is not None:
                tqdm.write(
                    f"[debug] Trees: {len(tree_xyz):,} pts  Ground: {len(ground_xyz):,} pts"
                )
            else:
                tqdm.write(
                    f"[debug] Trees: {len(tree_xyz):,} pts  Ground: estimated from lowest points"
                )

        # xy_mean          = tree_xyz[:, :2].mean(axis=0)
        # tree_xyz[:, :2] -= xy_mean

        if ground_xyz is not None and len(ground_xyz) > 0:
            # ground_xyz[:, :2] -= xy_mean
            voxel_size = 1.0
            voxel_idx = np.floor(ground_xyz / voxel_size).astype(np.int32)
            _, unique = np.unique(voxel_idx, axis=0, return_index=True)
            ground_xyz = ground_xyz[unique]
        else:
            ground_xyz = self._estimate_ground(tree_xyz)
        if ground_xyz.shape[0] < 3:
            tree_instance_labels = np.full(tree_xyz.shape[0], -1, dtype=np.int32)
            shrub_instance_labels = self._segment_shrubs(
                tree_xyz, tree_instance_labels, ground_xyz=ground_xyz
            )
            return (
                tree_instance_labels,
                shrub_instance_labels,
                np.full(len(tree_xyz), -1, dtype=np.int8),
            )

        tmpdir = tempfile.mkdtemp(prefix="treesegmray_")

        cloud_ply = os.path.join(tmpdir, "cloud.ply")
        ground_ply = os.path.join(tmpdir, "ground.ply")

        try:
            self._write_raycloud_ply(tree_xyz, cloud_ply, tree_rays)
            self._write_ground_mesh_ply(ground_xyz, ground_ply)

            seg_ply = raycloud_tools.run_trees(
                cloud_ply,
                ground_ply,
                height_min=self.height_min,
                max_diameter=self.max_diameter,
                crop_length=self.crop_length,
                distance_limit=self.distance_limit,
                girth_height_ratio=self.girth_height_ratio,
                gravity_factor=self.gravity_factor,
                global_taper=self.global_taper,
                global_taper_factor=self.global_taper_factor,
                grid_width=self.grid_width,
                use_rays=tree_rays is not None,
                segment_branches=False,
                workdir=tmpdir,
            )
            if not seg_ply.exists():
                raise RuntimeError(
                    f"Segmented output not found at {seg_ply}.\n"
                    "Run with verbose=True to inspect raycloudtools output."
                )

            tree_instance_labels = self._read_labels_from_segmented_ply(
                os.fspath(seg_ply)
            )
            tree_instance_labels = self._connect_floating_clusters(
                tree_instance_labels,
                tree_xyz,
                ground_xyz,
                ground_z_threshold=1.5,
                min_cluster_size=500,
            )
            tree_instance_labels = self._remove_small_clusters(
                tree_instance_labels, min_points=5000
            )
            tree_instance_labels = self._reduce_labels(tree_instance_labels)
            wood_leaf_labels = self._segment_wood_leaf(
                tree_xyz,
                tree_instance_labels,
                pth.Path(cloud_ply).with_name("cloud_trees.txt"),
                self.wood_distance,
            )
            wood_leaf_labels = wood_leaf.propagate_wood(
                tree_xyz,
                tree_instance_labels,
                wood_leaf_labels,
                ground_xyz,
                height=self.wood_propagation_height,
                radius=self.wood_propagation_radius,
                min_points=self.wood_propagation_min_points,
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        tree_instance_labels = tree_instance_labels.astype(np.int32, copy=False)
        shrub_instance_labels = self._segment_shrubs(
            tree_xyz, tree_instance_labels, ground_xyz=ground_xyz
        )
        return tree_instance_labels, shrub_instance_labels, wood_leaf_labels

    _filter_shrub_clusters = staticmethod(cluster_processing.filter_shrub_clusters)

    def _segment_shrubs(
        self,
        tree_xyz: np.ndarray,
        tree_ids: np.ndarray,
        ground_xyz: np.ndarray | None = None,
        xy_voxel_size: float = 0.5,
        outlier_neighbors: int = 48,
        outlier_std_ratio: float = 0.5,
    ) -> np.ndarray:
        """Return shrub instance IDs for vegetation not assigned to a tree.

        ``tree_xyz`` and ``tree_ids`` contain only points selected by the
        vegetation-class mask. Clusters rejected by ground-contact, width,
        or shape filtering remain at shrub ID ``-1``.
        """
        if tree_xyz.shape[0] != tree_ids.shape[0]:
            raise ValueError(
                "tree_xyz and tree_ids length mismatch: "
                f"{tree_xyz.shape[0]} != {tree_ids.shape[0]}"
            )

        shrub_ids = np.full(tree_ids.shape, -1, dtype=np.int32)
        if len(tree_xyz) == 0:
            return shrub_ids
        if outlier_neighbors < 1:
            raise ValueError("outlier_neighbors must be at least 1")
        if outlier_std_ratio < 0:
            raise ValueError("outlier_std_ratio must not be negative")

        shrub_point_indices = np.flatnonzero(tree_ids == -1)
        shrub_xyz = tree_xyz[shrub_point_indices]
        if len(shrub_xyz) <= 1:
            return shrub_ids

        shrub_instance_ids = self._segment_xy_connected_components(
            shrub_xyz,
            voxel_size=xy_voxel_size,
        )
        retained_mask = self._filter_shrub_clusters(
            shrub_xyz, shrub_instance_ids, ground_xyz
        )
        shrub_point_indices = shrub_point_indices[retained_mask]
        shrub_xyz = shrub_xyz[retained_mask]
        shrub_instance_ids = shrub_instance_ids[retained_mask]
        retained_mask = self._remove_partial_trunks(
            shrub_xyz,
            shrub_instance_ids,
        )

        shrub_point_indices = shrub_point_indices[retained_mask]
        shrub_xyz = shrub_xyz[retained_mask]
        shrub_instance_ids = shrub_instance_ids[retained_mask]
        if len(shrub_xyz) == 0:
            return shrub_ids

        shrub_instance_ids = self._reduce_labels(shrub_instance_ids)
        shrub_ids[shrub_point_indices] = shrub_instance_ids

        # plot_cloud(shrub_xyz, shrub_instance_ids, title="Retained shrub points")

        return shrub_ids

    def _postprocess_tree_group(
        self,
        tree_xyz: NDArray,
        tree_ids: NDArray[np.int32],
        shrub_ids: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """Postprocess a Birch group only when it contains no shrubs."""
        if tree_xyz.shape[0] != tree_ids.shape[0] or tree_ids.shape != shrub_ids.shape:
            raise ValueError("Tree-group points, tree IDs, and shrub IDs must align")
        if np.any(shrub_ids >= 0):
            return tree_ids

        tree_ids = self._merge_close_trunks(
            tree_xyz,
            tree_ids,
            min_trunk_dist=0.3,
            trunk_height_band=(0.5, 1.0),
        )
        return self._remove_small_clusters(tree_ids, min_points=5000)

    def _segment_birch(
        self,
        xyz: NDArray,
        labels: NDArray,
        rays: NDArray | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int8]]:
        from sklearn.cluster import Birch

        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            empty_ids = np.full(0, -1, dtype=np.int32)
            return empty_ids, empty_ids.copy(), np.full(0, -1, dtype=np.int8)

        tree_xyz = xyz[tree_mask]

        n_clusters = int(tree_xyz.shape[0] / 4e6)
        n_clusters = max(2, int(n_clusters))

        with tqdm(
            desc="Subsampling PCD for coarse tree clusterization",
            unit="step",
            total=1,
            leave=False,
            position=2,
            disable=not self.verbose,
        ) as pbar:
            tree_xyz_lr_mask = self.voxel_subsample_vectorized(tree_xyz, voxel_size=0.3)
            tree_xyz_lr = tree_xyz[tree_xyz_lr_mask]
            pbar.update(1)

        model = Birch(threshold=2.5, branching_factor=128, n_clusters=n_clusters)

        chunk_size = int(2e6)
        pbar = range(0, tree_xyz_lr.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse clustering fit", leave=False, position=2)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz_lr.shape[0])
            model.partial_fit(tree_xyz_lr[start:end])

        model.partial_fit()

        group_ids_lr = np.empty(tree_xyz_lr.shape[0], dtype=np.int32)
        pbar = range(0, tree_xyz_lr.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse clustering predict", leave=False, position=2)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz_lr.shape[0])
            group_ids_lr[start:end] = model.predict(tree_xyz_lr[start:end])

        del model

        chunk_size = int(2e6)
        kdtree = KDTree(tree_xyz_lr)
        group_ids = np.empty(tree_xyz.shape[0], dtype=np.int32)
        pbar = range(0, tree_xyz.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse label upsampling", leave=False, position=2)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz.shape[0])
            _, idx = kdtree.query(tree_xyz[start:end], k=1)
            group_ids[start:end] = group_ids_lr[idx]
        del tree_xyz_lr, tree_xyz_lr_mask, group_ids_lr, kdtree
        gc.collect()

        full_tree_ids = np.full(len(tree_xyz), -1, dtype=np.int32)
        full_shrub_ids = np.full(len(tree_xyz), -1, dtype=np.int32)
        full_wood_leaf_labels = np.full(len(tree_xyz), -1, dtype=np.int8)
        tree_indices = np.flatnonzero(tree_mask)
        tree_id_offset = 0
        shrub_id_offset = 0

        group_labels = np.unique(group_ids)
        pbar = (
            tqdm(group_labels, desc="Fine tree clustering", leave=False, position=2)
            if self.verbose
            else group_labels
        )
        bbox_chunk_size = int(2e6)
        for group_id in pbar:
            group_positions = np.flatnonzero(group_ids == group_id)
            group_tree = tree_xyz[group_positions]
            if group_tree.shape[0] == 0:
                continue

            max_xyz, min_xyz = group_tree.max(axis=0), group_tree.min(axis=0)
            group_index_chunks = []
            for start in range(0, xyz.shape[0], bbox_chunk_size):
                end = min(start + bbox_chunk_size, xyz.shape[0])
                xyz_chunk = xyz[start:end]
                chunk_mask = (
                    (xyz_chunk[:, 0] >= min_xyz[0])
                    & (xyz_chunk[:, 0] <= max_xyz[0])
                    & (xyz_chunk[:, 1] >= min_xyz[1])
                    & (xyz_chunk[:, 1] <= max_xyz[1])
                    & (xyz_chunk[:, 2] >= min_xyz[2])
                    & (xyz_chunk[:, 2] <= max_xyz[2])
                )
                if chunk_mask.any():
                    group_index_chunks.append(np.flatnonzero(chunk_mask) + start)
            if len(group_index_chunks) == 0:
                continue

            group_indices = np.concatenate(group_index_chunks)
            group_voxel = xyz[group_indices]
            group_voxel_labels = labels[group_indices]
            group_voxel_rays = rays[group_indices] if rays is not None else None
            group_voxel_tree_mask = group_voxel_labels == self.tree_label

            tree_ids_voxel, shrub_ids_voxel, wood_leaf_voxel = self._segment_small(
                group_voxel, group_voxel_labels, group_voxel_rays
            )

            group_voxel_tree_indices = group_indices[group_voxel_tree_mask]
            tree_positions_in_voxel = np.searchsorted(
                tree_indices, group_voxel_tree_indices
            )
            group_tree_mask_in_voxel = group_ids[tree_positions_in_voxel] == group_id

            group_tree_ids = tree_ids_voxel[group_tree_mask_in_voxel].astype(
                np.int32, copy=True
            )
            group_shrub_ids = shrub_ids_voxel[group_tree_mask_in_voxel].astype(
                np.int32, copy=True
            )
            target_positions = tree_positions_in_voxel[group_tree_mask_in_voxel]
            group_tree_ids = self._postprocess_tree_group(
                tree_xyz[target_positions],
                group_tree_ids,
                group_shrub_ids,
            )

            valid = group_tree_ids >= 0
            if valid.any():
                group_tree_ids[valid] += tree_id_offset
                tree_id_offset = int(group_tree_ids[valid].max()) + 1

            valid_shrub = group_shrub_ids >= 0
            if valid_shrub.any():
                group_shrub_ids[valid_shrub] += shrub_id_offset
                shrub_id_offset = int(group_shrub_ids[valid_shrub].max()) + 1

            full_tree_ids[target_positions] = group_tree_ids
            full_shrub_ids[target_positions] = group_shrub_ids
            group_wood_leaf = wood_leaf_voxel[group_tree_mask_in_voxel].copy()
            group_wood_leaf[group_tree_ids < 0] = -1
            full_wood_leaf_labels[target_positions] = group_wood_leaf

            del (
                group_index_chunks,
                group_indices,
                group_voxel,
                group_voxel_labels,
                group_voxel_rays,
                tree_ids_voxel,
                shrub_ids_voxel,
                wood_leaf_voxel,
            )
            gc.collect()

        full_tree_ids = self._reduce_labels(full_tree_ids)

        return (
            full_tree_ids.astype(np.int32, copy=False),
            full_shrub_ids,
            full_wood_leaf_labels,
        )

    voxel_subsample_vectorized = staticmethod(
        cloud_geometry.voxel_subsample_vectorized
    )

    _merge_instance_ids = staticmethod(cluster_processing.merge_instance_ids)

    def segment(
        self,
        xyz: NDArray,
        labels: NDArray,
        rays: NDArray | None = None,
        gps_time: NDArray | None = None,
        point_source_id: NDArray | None = None,
        scan_angle_deg: NDArray | None = None,
        scan_angle_rank: NDArray | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.int8], NDArray[np.int8]]:
        """Return instance IDs, initial species, and wood/leaf labels (-1 non-tree, 0 wood, 1 leaf)."""
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] not in {3, 6, 7}:
            raise ValueError("xyz must have shape (n, 3), (n, 6), or (n, 7)")
        if rays is None and xyz.shape[1] == 6:
            rays = xyz[:, 3:6]
        elif rays is None and xyz.shape[1] == 7:
            rays = xyz[:, 4:7]
        xyz = xyz[:, :3]

        if rays is not None:
            rays = np.asarray(rays)
            if rays.shape != xyz.shape:
                raise ValueError(f"rays must have shape {xyz.shape}, not {rays.shape}")
            if not np.isfinite(rays).all():
                raise ValueError("rays must contain only finite values")

        full_tree_ids = np.full(len(xyz), -1, dtype=np.int32)
        full_shrub_ids = np.full(len(xyz), -1, dtype=np.int32)
        full_wood_leaf_labels = np.full(len(xyz), -1, dtype=np.int8)
        if xyz.shape[0] == 0:
            return (
                *self._merge_instance_ids(full_tree_ids, full_shrub_ids),
                full_wood_leaf_labels,
            )
        if xyz.shape[0] != labels.shape[0]:
            raise ValueError(
                f"xyz and labels length mismatch: {xyz.shape[0]} != {labels.shape[0]}"
            )

        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            return (
                *self._merge_instance_ids(full_tree_ids, full_shrub_ids),
                full_wood_leaf_labels,
            )

        if rays is None and self.use_rays:
            if scan_angle_deg is not None and scan_angle_rank is not None:
                raise ValueError(
                    "Provide scan angles using either scan_angle_deg or "
                    "scan_angle_rank, not both"
                )
            if scan_angle_deg is None:
                scan_angle_deg = scan_angle_rank
            rays = get_rays(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                gps_time=gps_time,
                point_source_id=point_source_id,
                scan_angle_deg=scan_angle_deg,
            )

        xyz = (xyz - xyz.mean(axis=0)).astype(np.float32)

        if xyz[tree_mask].shape[0] > 1e7:  # threshold checked
            tree_ids, shrub_ids, wood_leaf_labels = self._segment_birch(
                xyz.copy(), labels, rays
            )
        else:
            tree_ids, shrub_ids, wood_leaf_labels = self._segment_small(
                xyz, labels, rays
            )
        full_tree_ids[tree_mask] = tree_ids
        full_shrub_ids[tree_mask] = shrub_ids
        full_wood_leaf_labels[tree_mask] = wood_leaf_labels

        return (
            *self._merge_instance_ids(full_tree_ids, full_shrub_ids),
            full_wood_leaf_labels,
        )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


def argparser(argv=None):
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Segment tree and shrub instances in a LAZ file."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=module_dir / "final_files" / "config_RE.json",
    )
    parser.add_argument("--output-path", type=Path)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = argparser(argv)

    import laspy

    if __package__:
        from .utils.save_laz import save_laz
    else:
        from utils.save_laz import save_laz

    seg = TreeSegmRay.from_config(
        cfg_path=args.config_path,
        verbose=args.verbose,
    )

    path = args.input_path
    las = laspy.read(path)
    xyz = np.column_stack((np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)))
    labels = np.asarray(las.classification)

    ray_inputs = get_las_ray_inputs(las, use_rays=seg.use_rays)
    merged_instance_ids, _initial_model_species, wood_leaf_labels = seg.segment(
        xyz,
        labels,
        **ray_inputs,
    )

    output_path = args.output_path or path.with_name(f"{path.stem}_segmented.laz")
    save_laz(las, merged_instance_ids, output_path)
    if args.plot:
        if __package__:
            from .utils.plot_cloud import plot_cloud
        else:
            from utils.plot_cloud import plot_cloud

        tree_mask = (merged_instance_ids >= 0) & (wood_leaf_labels >= 0)
        for tree_id in np.unique(merged_instance_ids[tree_mask]):
            mask = tree_mask & (merged_instance_ids == tree_id)
            plot_cloud(
                points=xyz[mask],
                feature=wood_leaf_labels[mask],
                title="Wood/leaf segmentation (0: wood, 1: leaf)",
            )


if __name__ == "__main__":
    main()
