import gc
import json
import os
import pathlib as pth
import shutil
import struct
import subprocess
import tempfile
import uuid
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import (
    Delaunay,
    KDTree,
)
from tqdm import tqdm

try:
    from .utils.plot_cloud import plot_cloud
except ImportError:
    from utils.plot_cloud import plot_cloud


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
#     segment_branches:    bool          = False
#     ground_label:        Optional[int] = None
#     tree_label:          Optional[int] = None



class TreeSegmRay:
    def __init__(
        self,
        height_min:          float         = 2.0,
        max_diameter:        float         = 0.9,
        crop_length:         float         = 1.0,
        distance_limit:      float         = 0.3,
        girth_height_ratio:  float         = 0.12,
        gravity_factor:      float         = 0.75,
        global_taper:        float | None = None,
        global_taper_factor: float | None = None,
        grid_width:          float | None = None,
        use_rays:            bool          = False,
        segment_branches:    bool          = False,
        ground_label:        int | None = None,
        tree_label:          int | None = None,
        verbose:             bool          = False
    ):
        self.verbose             = verbose
        self.height_min          = height_min
        self.max_diameter        = max_diameter
        self.crop_length         = crop_length
        self.distance_limit      = distance_limit
        self.girth_height_ratio  = girth_height_ratio
        self.gravity_factor      = gravity_factor
        self.global_taper        = global_taper
        self.global_taper_factor = global_taper_factor
        self.grid_width          = grid_width
        self.use_rays            = use_rays
        self.segment_branches    = segment_branches
        self.tree_label          = tree_label
        self.ground_label        = ground_label

        self._container_name = None
        self._shared_tmpdir  = None
        self._backend        = self._detect_backend()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, cfg_path: str | pth.Path | None = None, verbose: bool = False) -> "TreeSegmRay":
        if cfg is not None:
            return cls(**cfg, verbose=verbose)
        elif cfg is None and cfg_path is not None:
            cfg_path = pth.Path(cfg_path)
            with open(cfg_path, 'r') as f:
                cfg = cast(dict[str, Any], json.load(f))
            return cls(**cfg, verbose=verbose)
        else:
            raise ValueError("Either cfg or cfg_path must be provided.")

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    def start_container(self):
        if self._backend != "docker":
            return
        if self._container_name is not None:
            return

        self._shared_tmpdir  = tempfile.mkdtemp(prefix="treesegmray_persistent_", dir=os.path.expanduser("~"))
        self._container_name = f"treesegmray_{uuid.uuid4().hex[:8]}"
        subprocess.run([
            "docker", "run", "-d",
            "--name", self._container_name,
            "-v",     f"{self._shared_tmpdir}:/data",
            "ghcr.io/csiro-robotics/raycloudtools:latest",
            "sleep", "infinity",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def rm_container(self):
        if self._container_name:
            subprocess.run(
                ["docker", "rm", "-f", self._container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._container_name = None

        if self._shared_tmpdir:
            shutil.rmtree(self._shared_tmpdir, ignore_errors=True)
            self._shared_tmpdir = None

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        if shutil.which("rayextract"):
            return "native"
        if shutil.which("docker"):
            if subprocess.run(
                ["docker", "info"], capture_output=True, check=False
            ).returncode != 0:
                subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
                subprocess.run(["docker", "info"], check=True)

            r = subprocess.run(
                ["docker", "image", "inspect",
                "ghcr.io/csiro-robotics/raycloudtools:latest"],
                capture_output=True,
                check=False,
            )
            if r.returncode == 0:
                return "docker"
            raise OSError(
                "Docker found but raycloudtools image not pulled.\n"
            )
        raise OSError(
            "raycloudtools not found"
        )

    def _run(self, cmd: list, workdir: str):
        if self._backend == "docker":
            if self._container_name:
                def to_running_container(arg):
                    if os.path.isabs(arg):
                        rel = os.path.relpath(arg, self._shared_tmpdir)
                        return "/data/" + rel
                    return arg
                cmd = ["docker", "exec", self._container_name] + \
                    [to_running_container(a) for a in cmd]
            else:
                def to_ephemeral_container(arg):
                    if os.path.isabs(arg):
                        return "/data/" + os.path.basename(arg)
                    return arg
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{workdir}:/data",
                    "ghcr.io/csiro-robotics/raycloudtools:latest",
                ] + [to_ephemeral_container(a) for a in cmd]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=workdir, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rayextract failed (exit {result.returncode}):\n"
                f"{result.stdout or ''}{result.stderr or '(no output)'}"
            )

    # ------------------------------------------------------------------
    # PLY helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_raycloud_ply(points: NDArray, path: str):
        n      = len(points)
        pts    = points.astype(np.float32)
        nxyz   = np.tile(np.array([0, 0, 10], dtype=np.float32), (n, 1))
        times  = np.zeros(n, dtype=np.float64)
        colors = np.full((n, 4), 128, dtype=np.uint8)

        with open(path, "wb") as f:
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                "comment generated by TreeSegmRay\n"
                f"element vertex {n:010d}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property double time\n"
                "property float nx\nproperty float ny\nproperty float nz\n"
                "property uchar red\nproperty uchar green\n"
                "property uchar blue\nproperty uchar alpha\nend_header\n"
            ).encode("ascii"))
            for i in range(n):
                f.write(pts[i].tobytes())
                f.write(times[i].tobytes())
                f.write(nxyz[i].tobytes())
                f.write(colors[i].tobytes())

    @staticmethod
    def _write_ground_mesh_ply(ground_xyz: NDArray, path: str):
        if ground_xyz.shape[0] < 3:
            raise ValueError("Cannot build ground mesh from fewer than 3 points")

        tri   = Delaunay(ground_xyz[:, :2])
        verts = ground_xyz.astype(np.float32)
        faces = tri.simplices.astype(np.int32)

        with open(path, "wb") as f:
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                "comment generated by TreeSegmRay\n"
                f"element vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            ).encode("ascii"))
            f.write(verts.tobytes())
            f.writelines(
                struct.pack("<B3i", 3, face[0], face[1], face[2])
                for face in faces
            )

    @staticmethod
    def _read_labels_from_segmented_ply(path: str) -> NDArray:
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii").strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            n_points, props = 0, []
            for line in header_lines:
                if line.startswith("element vertex"):
                    n_points = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    props.append((parts[1], parts[2]))

            type_map = {
                "float": "f", "float32": "f", "double": "d", "float64": "d",
                "int": "i",   "int32": "i",   "uint": "I",   "uint32": "I",
                "short": "h", "ushort": "H",  "uchar": "B",  "uint8": "B",
                "char": "b",  "int8": "b",
            }
            names  = [n for _, n in props]
            fmt    = "<" + "".join(type_map[t] for t, _ in props)
            stride = struct.calcsize(fmt)
            ri, gi, bi = names.index("red"), names.index("green"), names.index("blue")
            raw = f.read(n_points * stride)

        records = struct.iter_unpack(fmt, raw)
        colors  = np.array([(r[ri], r[gi], r[bi]) for r in records], dtype=np.int32)
        packed  = colors[:, 0] << 16 | colors[:, 1] << 8 | colors[:, 2]
        _, labels = np.unique(packed, return_inverse=True)
        return labels.astype(np.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_floating_clusters(self, tree_labels: NDArray, tree_xyz: NDArray,
                                    ground_xyz: NDArray,
                                    ground_z_threshold: float = 0.5,
                                    min_cluster_size: int = 5000,
                                    max_tilt_deg: float = 30.0) -> NDArray:
        if tree_xyz.shape[0] == 0 or ground_xyz.shape[0] == 0 or tree_labels.shape[0] == 0:
            return tree_labels

        ground_z_max  = ground_xyz[:, 2].max()
        unique_labels = np.unique(tree_labels)

        grounded, floating = [], []
        for lbl in unique_labels:
            mask        = tree_labels == lbl
            cluster_pts = tree_xyz[mask]
            is_large    = mask.sum() >= min_cluster_size
            is_grounded = cluster_pts[:, 2].min() <= ground_z_max + ground_z_threshold
            if is_grounded or is_large:
                grounded.append(lbl)
            else:
                floating.append(lbl)

        if len(floating) == 0 or len(grounded) == 0:
            return tree_labels

        grounded = np.array(grounded)
        floating = np.array(floating)

        grounded_centroids = np.array([
            tree_xyz[tree_labels == lbl].mean(axis=0) for lbl in grounded
        ], dtype=np.float32)
        floating_centroids = np.array([
            tree_xyz[tree_labels == lbl].mean(axis=0) for lbl in floating
        ], dtype=np.float32)

        tilt_tolerance = np.tan(np.deg2rad(max_tilt_deg))
        result  = tree_labels.copy()
        kdtree  = KDTree(grounded_centroids)

        for i, lbl in enumerate(floating):
            fc         = floating_centroids[i]
            below_mask = grounded_centroids[:, 2] < fc[2]

            if below_mask.any():
                candidates    = grounded_centroids[below_mask]
                candidate_ids = grounded[below_mask]
                dz            = fc[2] - candidates[:, 2]
                dxy           = np.linalg.norm(fc[:2] - candidates[:, :2], axis=1)
                tilt_score    = dxy - tilt_tolerance * dz
                target        = candidate_ids[np.argmin(tilt_score)]
            else:
                _, nn  = kdtree.query(fc, k=1)
                target = grounded[nn]

            result[result == lbl] = target

        return result

    def _reduce_labels(self, labels: NDArray) -> NDArray:
        valid_mask = labels != -1
        _, labels[valid_mask] = np.unique(
            labels[valid_mask],
            return_inverse=True,
        )
        return labels

    def _remove_small_clusters(self, tree_labels: NDArray,
                                min_points: int = 100) -> NDArray:
        result = tree_labels.copy()
        for lbl in np.unique(tree_labels):
            if (tree_labels == lbl).sum() < min_points:
                result[tree_labels == lbl] = -1
        return result

    @staticmethod
    def _connected_components_voxel(
        xyz: NDArray[np.float32],
        voxel_size: float = 0.3,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
        """Cluster a dense voxel grid using 26-neighbour connectivity.

        Returns per-point labels, cluster minimum heights, and cluster XY
        bounds ordered as ``x_min, y_min, x_max, y_max``.
        """
        if xyz.shape[0] == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )
        if voxel_size <= 0:
            raise ValueError("voxel_size must be greater than zero")

        from scipy.ndimage import label as ndimage_label

        point_voxels = np.floor(xyz / voxel_size).astype(np.int32)
        point_voxels -= point_voxels.min(axis=0)
        grid_shape = tuple(
            int(axis_size) for axis_size in point_voxels.max(axis=0) + 1
        )

        occupied_grid = np.zeros(grid_shape, dtype=bool)
        occupied_grid[
            point_voxels[:, 0],
            point_voxels[:, 1],
            point_voxels[:, 2],
        ] = True
        labeled_grid, n_clusters = ndimage_label(  # type: ignore[misc]
            occupied_grid,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        del occupied_grid

        point_labels = (
            labeled_grid[
                point_voxels[:, 0],
                point_voxels[:, 1],
                point_voxels[:, 2],
            ]
            - 1
        ).astype(np.int32)
        del labeled_grid, point_voxels
        min_heights = np.full(n_clusters, np.inf, dtype=np.float32)
        np.minimum.at(min_heights, point_labels, xyz[:, 2])

        xy_bounds = np.empty((n_clusters, 4), dtype=np.float32)
        xy_bounds[:, :2] = np.inf
        xy_bounds[:, 2:] = -np.inf
        np.minimum.at(xy_bounds[:, 0], point_labels, xyz[:, 0])
        np.minimum.at(xy_bounds[:, 1], point_labels, xyz[:, 1])
        np.maximum.at(xy_bounds[:, 2], point_labels, xyz[:, 0])
        np.maximum.at(xy_bounds[:, 3], point_labels, xyz[:, 1])

        return point_labels, min_heights, xy_bounds

    @staticmethod
    def _segment_xy_connected_components(
        xyz: NDArray[np.float32],
        voxel_size: float = 0.5,
    ) -> NDArray[np.int32]:
        """Segment points by 8-connected occupied cells in the XY plane."""
        if len(xyz) == 0:
            return np.zeros(0, dtype=np.int32)
        if voxel_size <= 0:
            raise ValueError("voxel_size must be greater than zero")

        from scipy.ndimage import label as ndimage_label

        point_cells = np.floor(xyz[:, :2] / voxel_size).astype(np.int32)
        point_cells -= point_cells.min(axis=0)
        grid_shape = tuple(
            int(axis_size) for axis_size in point_cells.max(axis=0) + 1
        )
        occupied_grid = np.zeros(grid_shape, dtype=bool)
        occupied_grid[point_cells[:, 0], point_cells[:, 1]] = True
        labeled_grid, _ = ndimage_label(  # type: ignore[misc]
            occupied_grid,
            structure=np.ones((3, 3), dtype=bool),
        )
        point_labels = (
            labeled_grid[point_cells[:, 0], point_cells[:, 1]] - 1
        ).astype(np.int32)
        return point_labels

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

    @staticmethod
    def _remove_partial_trunks(
        xyz: NDArray[np.float32],
        cluster_ids: NDArray[np.int32],
    ) -> NDArray[np.bool_]:
        """Diagnose clusters and exclude vertically linear trunk fragments."""
        if len(xyz) != len(cluster_ids):
            raise ValueError("xyz and cluster_ids must have the same length")
        if len(xyz) == 0:
            return np.zeros(0, dtype=bool)
        if np.any(cluster_ids < 0):
            raise ValueError("cluster_ids must not contain negative labels")

        cluster_count = int(cluster_ids.max()) + 1
        counts = np.bincount(cluster_ids, minlength=cluster_count)
        eligible_ids = np.flatnonzero(counts >= 100)
        linearity = np.zeros(cluster_count, dtype=np.float64)
        z_alignment = np.zeros(cluster_count, dtype=np.float64)

        if len(eligible_ids) > 0:
            coordinates = xyz - xyz[0]
            x, y, z = coordinates.T
            sums = np.column_stack((
                np.bincount(cluster_ids, weights=x, minlength=cluster_count),
                np.bincount(cluster_ids, weights=y, minlength=cluster_count),
                np.bincount(cluster_ids, weights=z, minlength=cluster_count),
            ))
            eligible_counts = counts[eligible_ids]
            means = sums[eligible_ids] / eligible_counts[:, None]

            covariances = np.empty((len(eligible_ids), 3, 3), dtype=np.float64)
            covariances[:, 0, 0] = (
                np.bincount(cluster_ids, weights=x * x, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 0] ** 2
            )
            covariances[:, 0, 1] = covariances[:, 1, 0] = (
                np.bincount(cluster_ids, weights=x * y, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 0] * means[:, 1]
            )
            covariances[:, 0, 2] = covariances[:, 2, 0] = (
                np.bincount(cluster_ids, weights=x * z, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 0] * means[:, 2]
            )
            covariances[:, 1, 1] = (
                np.bincount(cluster_ids, weights=y * y, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 1] ** 2
            )
            covariances[:, 1, 2] = covariances[:, 2, 1] = (
                np.bincount(cluster_ids, weights=y * z, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 1] * means[:, 2]
            )
            covariances[:, 2, 2] = (
                np.bincount(cluster_ids, weights=z * z, minlength=cluster_count)[
                    eligible_ids
                ]
                / eligible_counts
                - means[:, 2] ** 2
            )

            eigenvalues, eigenvectors = np.linalg.eigh(covariances)
            largest_eigenvalues = np.maximum(eigenvalues[:, 2], 0.0)
            second_eigenvalues = np.maximum(eigenvalues[:, 1], 0.0)
            valid_eigenvalues = largest_eigenvalues > 0.0
            eligible_linearity = np.zeros(len(eligible_ids), dtype=np.float64)
            eligible_linearity[valid_eigenvalues] = (
                largest_eigenvalues[valid_eigenvalues]
                - second_eigenvalues[valid_eigenvalues]
            ) / largest_eigenvalues[valid_eigenvalues]
            linearity[eligible_ids] = eligible_linearity
            z_alignment[eligible_ids] = np.abs(eigenvectors[:, 2, 2])

        is_trunk = (
            (counts >= 100)
            & (linearity >= 0.55)
            & (z_alignment >= 0.70)
        )

        # point_order = np.argsort(cluster_ids, kind="stable")
        # ordered_xyz = xyz[point_order]
        # cluster_ends = np.cumsum(counts)
        # cluster_starts = cluster_ends - counts
        # for cluster_id in np.flatnonzero(counts):
        #     cluster_xyz = ordered_xyz[
        #         cluster_starts[cluster_id]:cluster_ends[cluster_id]
        #     ]
        #     print(
        #         f"cluster_id: {cluster_id}\n"
        #         f"point_count: {counts[cluster_id]}\n"
        #         f"linearity: {linearity[cluster_id]:.6g}\n"
        #         "principal_axis_z_alignment: "
        #         f"{z_alignment[cluster_id]:.6g}\n"
        #         f"is_trunk: {is_trunk[cluster_id]}"
        #     )
        #     plot_cloud(cluster_xyz, title=f"Cluster {cluster_id}")

        return ~is_trunk[cluster_ids]

    @staticmethod
    def _estimate_ground(tree_xyz: NDArray, grid_size: float = 2.0) -> NDArray:
        if tree_xyz.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        xs = tree_xyz[:, 0]
        ys = tree_xyz[:, 1]
        x_bins = np.arange(xs.min(), xs.max() + grid_size, grid_size)
        y_bins = np.arange(ys.min(), ys.max() + grid_size, grid_size)

        ground_pts = []
        for xi in range(len(x_bins) - 1):
            for yi in range(len(y_bins) - 1):
                mask = (
                    (xs >= x_bins[xi]) & (xs < x_bins[xi + 1]) &
                    (ys >= y_bins[yi]) & (ys < y_bins[yi + 1])
                )
                if mask.sum() > 0:
                    ground_pts.append(tree_xyz[mask][tree_xyz[mask, 2].argmin()])
        return np.array(ground_pts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Tile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _voxel_tiles(xyz: NDArray, voxel_size: float = 50.0, overlap: float = 3.0):
        if xyz.shape[0] == 0:
            return

        xs, ys = xyz[:, 0], xyz[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        for xs0 in np.arange(x_min, x_max, voxel_size):
            for ys0 in np.arange(y_min, y_max, voxel_size):
                yield {
                    "cx_min": xs0,           "cx_max": xs0 + voxel_size,
                    "cy_min": ys0,           "cy_max": ys0 + voxel_size,
                    "x_min":  xs0 - overlap, "x_max":  xs0 + voxel_size + overlap,
                    "y_min":  ys0 - overlap, "y_max":  ys0 + voxel_size + overlap,
                }

    @staticmethod
    def _core_mask(xyz: NDArray, tile: dict) -> NDArray:
        return (
            (xyz[:, 0] >= tile["cx_min"]) & (xyz[:, 0] < tile["cx_max"]) &
            (xyz[:, 1] >= tile["cy_min"]) & (xyz[:, 1] < tile["cy_max"])
        )

    @staticmethod
    def _tile_mask(xyz: NDArray, tile: dict) -> NDArray:
        return (
            (xyz[:, 0] >= tile["x_min"]) & (xyz[:, 0] < tile["x_max"]) &
            (xyz[:, 1] >= tile["y_min"]) & (xyz[:, 1] < tile["y_max"])
        )
    
    def _ensure_container(self):
        if self._backend != "docker" or self._container_name is None:
            return
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", self._container_name],
            capture_output=True, text=True, check=False
        )
        status = r.stdout.strip()
        if r.returncode != 0 or status != "running":
            tqdm.write(f"[container] status='{status}', restarting...")
            # preserve shared tmpdir, just recreate the container with same mount
            self._container_name = f"treesegmray_{uuid.uuid4().hex[:8]}"
            subprocess.run([
                "docker", "run", "-d",
                "--name", self._container_name,
                "-v", f"{self._shared_tmpdir}:/data",
                "ghcr.io/csiro-robotics/raycloudtools:latest",
                "sleep", "infinity",
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------
    # Trunk helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_trunk_position(
            tree_xyz: NDArray,
            trunk_height_band: tuple[float, float] = (0.5, 2.0)) -> NDArray | None:
        z_min = tree_xyz[:, 2].min()
        band  = tree_xyz[
            (tree_xyz[:, 2] >= z_min + trunk_height_band[0]) &
            (tree_xyz[:, 2] <= z_min + trunk_height_band[1])
        ]
        if len(band) == 0:
            return None
        return band[:, :2].mean(axis=0)

    def _merge_close_trunks(
        self,
        tree_xyz: NDArray,
        tree_ids: NDArray,
        min_trunk_dist: float = 1.5,
        trunk_height_band: tuple[float, float] = (0.5, 2.0),
        min_points: int = 50) -> NDArray:

        unique_ids = np.unique(tree_ids)
        unique_ids = unique_ids[unique_ids >= 0]
        if len(unique_ids) < 2:
            return tree_ids.copy()

        # sort by tree_id for O(1) per-tree slicing instead of O(n) masking
        sort_idx   = np.argsort(tree_ids, kind="stable")
        sorted_ids = tree_ids[sort_idx]
        sorted_xyz = tree_xyz[sort_idx]

        # find start index of each unique id in the sorted array
        boundaries = np.searchsorted(sorted_ids, unique_ids)

        trunk_xy    = {}
        point_count = {}

        for i, tid in enumerate(unique_ids):
            start = int(boundaries[i])
            end   = int(boundaries[i + 1]) if i + 1 < len(unique_ids) else len(sorted_ids)
            pts   = sorted_xyz[start:end]
            count = end - start
            point_count[tid] = count
            trunk_xy[tid]    = self._estimate_trunk_position(pts, trunk_height_band) if count >= min_points else None

        valid_ids = [tid for tid in unique_ids if trunk_xy[tid] is not None]
        if len(valid_ids) < 2:
            return tree_ids.copy()

        positions = np.array([trunk_xy[tid] for tid in valid_ids], dtype=np.float64)
        pairs     = KDTree(positions).query_pairs(r=min_trunk_dist, output_type="ndarray")
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
        remap  = np.arange(max_id + 1, dtype=np.int64)
        for idx, tid in enumerate(valid_ids):
            remap[tid] = valid_ids[find(idx)]

        new_ids            = tree_ids.copy()
        valid_mask         = new_ids >= 0
        new_ids[valid_mask] = remap[new_ids[valid_mask]]

        # re-index contiguously, preserving -1
        noise_mask = new_ids == -1
        _, new_ids[~noise_mask] = np.unique(new_ids[~noise_mask], return_inverse=True)
        new_ids[noise_mask] = -1

        return new_ids.astype(np.int64)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment_watershed(self, tree_xyz: NDArray, resolution: float = 0.15) -> NDArray:
        """2D watershed fallback on XY crown projection."""
        from scipy.ndimage import label
        from skimage.feature import peak_local_max  # type: ignore[import-not-found]
        from skimage.segmentation import watershed  # type: ignore[import-not-found]

        xy = tree_xyz[:, :2]
        xy_min = xy.min(axis=0)
        grid_shape = ((xy.max(axis=0) - xy_min) / resolution).astype(int) + 1

        # density map
        density = np.zeros(grid_shape, dtype=np.float32)
        idx = ((xy - xy_min) / resolution).astype(int)
        np.add.at(density, (idx[:, 0], idx[:, 1]), 1)

        # smooth + watershed
        from scipy.ndimage import gaussian_filter

        density_smooth = gaussian_filter(density, sigma=1.5)

        coords     = peak_local_max(density_smooth, min_distance=int(1.5 / resolution), threshold_abs=5)
        mask       = np.zeros(grid_shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = label(mask)  # type: ignore[misc]
        ws_labels  = watershed(-density_smooth, markers, mask=density > 0)

        # map back to points
        point_labels = ws_labels[idx[:, 0], idx[:, 1]].astype(np.int64) - 1  # 0-indexed, -1 = unlabelled
        return point_labels

    def _segment_small(
        self,
        xyz: NDArray,
        labels: NDArray | None = None,
        debug: bool = False,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:

        self.start_container()

        xyz -= xyz.min(axis=0)  # shift minimum XYZ to zero for raycloudtools

        if labels is not None and self.tree_label is not None and self.ground_label is not None:
            tree_mask   = labels == self.tree_label
            ground_mask = labels == self.ground_label
            tree_xyz    = xyz[tree_mask].copy()
            ground_xyz  = xyz[ground_mask].copy()
        else:
            tree_xyz   = xyz.copy()
            ground_xyz = None
            tree_mask  = np.ones(len(xyz), dtype=bool)

        if tree_xyz.shape[0] == 0:
            empty_ids = np.zeros(0, dtype=np.int32)
            return empty_ids, empty_ids.copy()

        if debug:
            if ground_xyz is not None:
                tqdm.write(f"[debug] Trees: {len(tree_xyz):,} pts  Ground: {len(ground_xyz):,} pts")
            else:
                tqdm.write(f"[debug] Trees: {len(tree_xyz):,} pts  Ground: estimated from lowest points")

        # xy_mean          = tree_xyz[:, :2].mean(axis=0)
        # tree_xyz[:, :2] -= xy_mean

        if ground_xyz is not None and len(ground_xyz) > 0:
            # ground_xyz[:, :2] -= xy_mean
            voxel_size = 1.0
            voxel_idx  = np.floor(ground_xyz / voxel_size).astype(np.int32)
            _, unique  = np.unique(voxel_idx, axis=0, return_index=True)
            ground_xyz = ground_xyz[unique]
        else:
            ground_xyz = self._estimate_ground(tree_xyz)
        if ground_xyz.shape[0] < 3:
            tree_instance_labels = np.full(tree_xyz.shape[0], -1, dtype=np.int32)
            bush_instance_labels = self._segment_bushes(tree_xyz, tree_instance_labels, ground_xyz=ground_xyz)
            return tree_instance_labels, bush_instance_labels

        # Each call gets its own subdirectory so concurrent tiles never
        # overwrite each other's cloud.ply / ground.ply inside the container.
        own_tmpdir = self._shared_tmpdir is None
        if own_tmpdir:
            tmpdir = tempfile.mkdtemp(prefix="treesegmray_", dir=os.path.expanduser("~"))
        else:
            tmpdir = tempfile.mkdtemp(prefix="tile_", dir=self._shared_tmpdir)

        cloud_ply  = os.path.join(tmpdir, "cloud.ply")
        ground_ply = os.path.join(tmpdir, "ground.ply")

        try:
            self._write_raycloud_ply(tree_xyz, cloud_ply)
            self._write_ground_mesh_ply(ground_xyz, ground_ply)

            cmd = [
                "rayextract", "trees", cloud_ply, ground_ply,
                "--height_min",         str(self.height_min),
                "--max_diameter",       str(self.max_diameter),
                "--crop_length",        str(self.crop_length),
                "--distance_limit",     str(self.distance_limit),
                "--girth_height_ratio", str(self.girth_height_ratio),
                "--gravity_factor",     str(self.gravity_factor),
            ]
            if self.global_taper is not None:
                cmd += ["--global_taper",        str(self.global_taper)]
            if self.global_taper_factor is not None:
                cmd += ["--global_taper_factor", str(self.global_taper_factor)]
            if self.grid_width is not None:
                cmd += ["--grid_width",          str(self.grid_width)]
            if self.use_rays:
                cmd.append("--use_rays")
            if self.segment_branches:
                cmd.append("--branch_segmentation")

            self._run(cmd, workdir=tmpdir)

            seg_ply = os.path.join(tmpdir, "cloud_segmented.ply")
            if not os.path.exists(seg_ply):
                raise RuntimeError(
                    f"Segmented output not found at {seg_ply}.\n"
                    "Run with verbose=True to inspect raycloudtools output."
                )

            tree_instance_labels = self._read_labels_from_segmented_ply(seg_ply)
            tree_instance_labels = self._connect_floating_clusters(
                tree_instance_labels, tree_xyz, ground_xyz,
                ground_z_threshold=1.5,
                min_cluster_size=500,
            )
            tree_instance_labels = self._remove_small_clusters(tree_instance_labels, min_points=5000)
            tree_instance_labels = self._reduce_labels(tree_instance_labels)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            self.rm_container()



        tree_instance_labels = tree_instance_labels.astype(np.int32, copy=False)
        bush_instance_labels = self._segment_bushes(tree_xyz, tree_instance_labels, ground_xyz=ground_xyz)
        return tree_instance_labels, bush_instance_labels

    def _segment_bushes(
        self,
        tree_xyz: NDArray[np.float32],
        tree_ids: NDArray[np.int32],
        ground_xyz: NDArray[np.float32] | None = None,
        xy_voxel_size: float = 0.5,
        outlier_neighbors: int = 48,
        outlier_std_ratio: float = 0.5,
    ) -> NDArray[np.int32]:
        """Return bush instance IDs for vegetation not assigned to a tree.

        ``tree_xyz`` and ``tree_ids`` contain only points selected by the
        vegetation-class mask. Floating components and vertically elongated
        clusters are rejected and remain at bush ID ``-1``.
        """
        if tree_xyz.shape[0] != tree_ids.shape[0]:
            raise ValueError(
                "tree_xyz and tree_ids length mismatch: "
                f"{tree_xyz.shape[0]} != {tree_ids.shape[0]}"
            )

        bush_ids = np.full(tree_ids.shape, -1, dtype=np.int32)
        if len(tree_xyz) == 0:
            return bush_ids
        if outlier_neighbors < 1:
            raise ValueError("outlier_neighbors must be at least 1")
        if outlier_std_ratio < 0:
            raise ValueError("outlier_std_ratio must not be negative")

        bush_point_indices = np.flatnonzero(tree_ids == -1)
        bush_xyz = tree_xyz[bush_point_indices]
        if len(bush_xyz) <= 1:
            return bush_ids

        if len(bush_xyz) == 0:
            return bush_ids

        retained_mask = self._filter_floating_clusters(
            bush_xyz,
            ground_xyz=ground_xyz,
            voxel_size=0.5,
            max_gap=1.0,
        )
        bush_point_indices = bush_point_indices[retained_mask]
        bush_xyz = bush_xyz[retained_mask]

        bush_instance_ids = self._segment_xy_connected_components(
            bush_xyz,
            voxel_size=xy_voxel_size,
        )
        retained_mask = self._remove_partial_trunks(
            bush_xyz,
            bush_instance_ids,
        )
        
        bush_point_indices = bush_point_indices[retained_mask]
        bush_xyz = bush_xyz[retained_mask]
        bush_instance_ids = bush_instance_ids[retained_mask]
        if len(bush_xyz) == 0:
            return bush_ids

        bush_instance_ids = self._reduce_labels(bush_instance_ids)
        bush_ids[bush_point_indices] = bush_instance_ids

        plot_cloud(bush_xyz, bush_instance_ids, title="Retained bush points")


        return bush_ids


    def _segment_birch(
        self,
        xyz: NDArray,
        labels: NDArray,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        from sklearn.cluster import Birch


        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            empty_ids = np.full(0, -1, dtype=np.int32)
            return empty_ids, empty_ids.copy()
        
        tree_xyz = xyz[tree_mask]

        n_clusters = int(tree_xyz.shape[0] / 4e6)
        n_clusters = max(2, int(n_clusters))

        with tqdm(desc="Subsampling PCD for coarse tree clusterization", unit="step", total=1, leave=False, position=1, disable=not self.verbose) as pbar:
            tree_xyz_lr_mask = self.voxel_subsample_vectorized(tree_xyz, voxel_size=0.3)
            tree_xyz_lr = tree_xyz[tree_xyz_lr_mask]
            pbar.update(1)
        
        model = Birch(
            threshold=2.5,
            branching_factor=128,
            n_clusters=n_clusters
        )

        chunk_size = int(2e6)
        pbar = range(0, tree_xyz_lr.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse clustering fit", leave=False, position=1)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz_lr.shape[0])
            model.partial_fit(tree_xyz_lr[start:end])

        model.partial_fit()

        group_ids_lr = np.empty(tree_xyz_lr.shape[0], dtype=np.int32)
        pbar = range(0, tree_xyz_lr.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse clustering predict", leave=False, position=1)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz_lr.shape[0])
            group_ids_lr[start:end] = model.predict(tree_xyz_lr[start:end])

        del model

        chunk_size = int(2e6)
        kdtree = KDTree(tree_xyz_lr)
        group_ids = np.empty(tree_xyz.shape[0], dtype=np.int32)
        pbar = range(0, tree_xyz.shape[0], chunk_size)
        if self.verbose:
            pbar = tqdm(pbar, desc="Coarse label upsampling", leave=False, position=1)
        for start in pbar:
            end = min(start + chunk_size, tree_xyz.shape[0])
            _, idx = kdtree.query(tree_xyz[start:end], k=1)
            group_ids[start:end] = group_ids_lr[idx]
        del tree_xyz_lr, tree_xyz_lr_mask, group_ids_lr, kdtree
        gc.collect()

        full_tree_ids = np.full(len(tree_xyz), -1, dtype=np.int32)
        full_bush_ids = np.full(len(tree_xyz), -1, dtype=np.int32)
        tree_indices = np.flatnonzero(tree_mask)
        tree_id_offset = 0
        bush_id_offset = 0

        group_labels = np.unique(group_ids)
        pbar = tqdm(group_labels, desc="Fine tree clustering", leave=False, position=1) if self.verbose else group_labels
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
                    (xyz_chunk[:, 0] >= min_xyz[0]) & (xyz_chunk[:, 0] <= max_xyz[0]) &
                    (xyz_chunk[:, 1] >= min_xyz[1]) & (xyz_chunk[:, 1] <= max_xyz[1]) &
                    (xyz_chunk[:, 2] >= min_xyz[2]) & (xyz_chunk[:, 2] <= max_xyz[2])
                )
                if chunk_mask.any():
                    group_index_chunks.append(np.flatnonzero(chunk_mask) + start)
            if len(group_index_chunks) == 0:
                continue

            group_indices = np.concatenate(group_index_chunks)
            group_voxel = xyz[group_indices]
            group_voxel_labels = labels[group_indices]
            group_voxel_tree_mask = group_voxel_labels == self.tree_label

            self.rm_container()
            try:
                tree_ids_voxel, bush_ids_voxel = self._segment_small(
                    group_voxel, group_voxel_labels
                )
            except Exception:  # noqa: BLE001
                max_tree_dim = 3.0
                group_voxel_tree = group_voxel[group_voxel_tree_mask]
                xy_extent = np.ptp(group_voxel_tree[:, :2], axis=0)
                is_single_tree = (xy_extent <= max_tree_dim).all()
                tree_ids_voxel = (
                    np.zeros(group_voxel_tree_mask.sum(), dtype=np.int32)
                    if is_single_tree
                    else np.full(group_voxel_tree_mask.sum(), -1, dtype=np.int32)
                )

                if group_voxel_labels is not None and self.ground_label is not None:
                    group_ground_xyz = group_voxel[group_voxel_labels == self.ground_label]
                else:
                    group_ground_xyz = None

                bush_ids_voxel = self._segment_bushes(
                    group_voxel_tree, tree_ids_voxel, ground_xyz=group_ground_xyz
                )

            group_voxel_tree_indices = group_indices[group_voxel_tree_mask]
            tree_positions_in_voxel = np.searchsorted(tree_indices, group_voxel_tree_indices)
            group_tree_mask_in_voxel = group_ids[tree_positions_in_voxel] == group_id

            group_tree_ids = tree_ids_voxel[group_tree_mask_in_voxel].astype(np.int32, copy=True)
            valid = group_tree_ids >= 0
            if valid.any():
                group_tree_ids[valid] += tree_id_offset
                tree_id_offset = int(group_tree_ids[valid].max()) + 1

            group_bush_ids = bush_ids_voxel[group_tree_mask_in_voxel].astype(np.int32, copy=True)
            valid_bush = group_bush_ids >= 0
            if valid_bush.any():
                group_bush_ids[valid_bush] += bush_id_offset
                bush_id_offset = int(group_bush_ids[valid_bush].max()) + 1

            target_positions = tree_positions_in_voxel[group_tree_mask_in_voxel]
            full_tree_ids[target_positions] = group_tree_ids
            full_bush_ids[target_positions] = group_bush_ids

            del group_index_chunks, group_indices, group_voxel, group_voxel_labels, tree_ids_voxel, bush_ids_voxel
            gc.collect()

        full_tree_ids = self._merge_close_trunks(tree_xyz, full_tree_ids,
                                                 min_trunk_dist=0.3,
                                                 trunk_height_band=(0.5, 1.0))
        full_tree_ids = self._remove_small_clusters(full_tree_ids, min_points=5000)
        full_tree_ids = self._reduce_labels(full_tree_ids)

        return full_tree_ids.astype(np.int32, copy=False), full_bush_ids

    @staticmethod
    def voxel_subsample_vectorized(xyz, voxel_size=0.25):
        if xyz.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        keys     = np.floor(xyz / voxel_size).astype(np.int32)
        centers  = (keys + 0.5) * voxel_size
        dists_sq = np.sum((xyz - centers) ** 2, axis=1)
    
        keys_min  = keys.min(axis=0)
        keys      = keys - keys_min
        key_range = keys.max(axis=0) + 1
    
        key_range = key_range.astype(np.int64)
        assert np.prod(key_range) < np.iinfo(np.int64).max, "key encoding overflow"
        strides = np.cumprod(np.r_[1, key_range[:0:-1]], dtype=np.int64)[::-1]
        key_enc = keys.astype(np.int64) @ strides
        
        order      = np.lexsort((dists_sq, key_enc))
        key_sorted = key_enc[order]
        _, first   = np.unique(key_sorted, return_index=True)
        chosen     = order[first]
    
        mask = np.zeros(xyz.shape[0], dtype=bool)
        mask[chosen] = True
        return mask


    def segment(self, xyz: NDArray, labels: NDArray) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        full_tree_ids = np.full(len(xyz), -1, dtype=np.int32)
        full_bush_ids = np.full(len(xyz), -1, dtype=np.int32)
        if xyz.shape[0] == 0:
            return full_tree_ids, full_bush_ids
        if xyz.shape[0] != labels.shape[0]:
            raise ValueError(f"xyz and labels length mismatch: {xyz.shape[0]} != {labels.shape[0]}")
        
        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            return full_tree_ids, full_bush_ids

        xyz = (xyz - xyz.mean(axis=0)).astype(np.float32)


        if xyz[tree_mask].shape[0] > 1e7: # threshold checked
            tree_ids, bush_ids = self._segment_birch(xyz.copy(), labels)
        else:
            tree_ids, bush_ids = self._segment_small(xyz, labels)
        full_tree_ids[tree_mask] = tree_ids
        full_bush_ids[tree_mask] = bush_ids

        return full_tree_ids, full_bush_ids


def test_segment_bushes_contract():
    tree_xyz = np.zeros((3, 3), dtype=np.float32)
    tree_ids = np.array([0, -1, 1], dtype=np.int32)

    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    bush_ids = segmenter._segment_bushes(tree_xyz=tree_xyz, tree_ids=tree_ids)

    assert bush_ids.shape == tree_ids.shape
    assert bush_ids.dtype == np.int32
    assert np.all(bush_ids[tree_ids >= 0] == -1)


def test_connected_components_voxel_two_clusters():
    rng = np.random.default_rng(42)
    cluster_a = rng.uniform(0.0, 1.0, (50, 3)).astype(np.float32)
    cluster_b = rng.uniform(10.0, 11.0, (30, 3)).astype(np.float32)
    cluster_b[:, 2] += 5.0  # shift z so min_height differs
    xyz = np.concatenate([cluster_a, cluster_b])

    labels, min_heights, xy_bounds = TreeSegmRay._connected_components_voxel(
        xyz, voxel_size=0.3,
    )

    assert labels.shape == (80,)
    assert labels.dtype == np.int32
    assert len(np.unique(labels)) == 2
    assert np.all(labels[:50] == labels[0])
    assert np.all(labels[50:] == labels[50])
    assert labels[0] != labels[50]

    la, lb = labels[0], labels[50]
    assert min_heights[la] < 1.1
    assert min_heights[lb] > 14.0
    assert xy_bounds.shape == (2, 4)
    np.testing.assert_allclose(
        xy_bounds[la],
        [cluster_a[:, 0].min(), cluster_a[:, 1].min(),
         cluster_a[:, 0].max(), cluster_a[:, 1].max()],
    )
    np.testing.assert_allclose(
        xy_bounds[lb],
        [cluster_b[:, 0].min(), cluster_b[:, 1].min(),
         cluster_b[:, 0].max(), cluster_b[:, 1].max()],
    )


def test_segment_xy_connected_components_ignores_height():
    xyz = np.array([
        [0.1, 0.1, 0.0],
        [0.6, 0.6, 10.0],
        [3.0, 3.0, 0.0],
    ], dtype=np.float32)

    labels = TreeSegmRay._segment_xy_connected_components(
        xyz,
        voxel_size=0.5,
    )

    assert labels.tolist() == [0, 0, 1]


def test_remove_partial_trunks_filters_vertical_linear_cluster():
    point_count = 120
    heights = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    trunk = np.column_stack((
        np.sin(heights) * 0.02,
        np.cos(heights) * 0.02,
        heights,
    )).astype(np.float32)
    bush = np.column_stack((
        np.linspace(2.0, 4.0, point_count, dtype=np.float32),
        np.tile(np.array([0.0, 1.0], dtype=np.float32), point_count // 2),
        np.tile(np.array([0.0, 0.2], dtype=np.float32), point_count // 2),
    ))
    xyz = np.concatenate((trunk, bush))
    cluster_ids = np.repeat(np.array([0, 1], dtype=np.int32), point_count)
    plotted_clusters = []

    original_plot_cloud = globals()["plot_cloud"]
    globals()["plot_cloud"] = lambda points, **_: plotted_clusters.append(points)
    try:
        retained_mask = TreeSegmRay._remove_partial_trunks(xyz, cluster_ids)
    finally:
        globals()["plot_cloud"] = original_plot_cloud

    assert retained_mask.dtype == np.bool_
    assert not retained_mask[:point_count].any()
    assert retained_mask[point_count:].all()
    assert len(plotted_clusters) == 2


def test_connected_components_voxel_empty():
    labels, min_h, xy_bounds = TreeSegmRay._connected_components_voxel(
        np.zeros((0, 3), dtype=np.float32),
    )
    assert labels.shape == (0,)
    assert min_h.shape == (0,)
    assert xy_bounds.shape == (0, 4)


def test_filter_floating_clusters_uses_bounds_and_nearest_ground_mean():
    xyz = np.array([
        [0.0, 0.0, 0.5],
        [10.0, 10.0, 3.0],
    ], dtype=np.float32)
    ground_xyz = np.array([
        [0.0, 0.0, 0.0],
        [9.0, 10.0, 0.0],
        [10.0, 9.0, 0.0],
    ], dtype=np.float32)

    retained_mask = TreeSegmRay._filter_floating_clusters(
        xyz,
        ground_xyz=ground_xyz,
        voxel_size=0.5,
        max_gap=1.0,
        nearest_ground_points=2,
    )

    assert retained_mask.tolist() == [True, False]


def test_reduce_labels_removes_gaps_without_discarding_zero():
    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    labels = np.array([-1, 4, 4, 9], dtype=np.int32)

    reduced = segmenter._reduce_labels(labels)

    assert reduced.tolist() == [-1, 0, 0, 1]


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def main():
    import laspy

    seg = TreeSegmRay(ground_label=1,
                      tree_label=7, verbose=True)

    seg = TreeSegmRay.from_config(cfg_path="src/final_files/config_RE.json", verbose=True)

    for path in ["/Users/michalsiniarski/Documents/PROGRAMMING/BRIK-data-processing/src/TreeClustering/fixtures/BIG_CLOUD.laz"]:
        las    = laspy.read(path)
        xyz    = np.column_stack(
            (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
        )
        labels = np.asarray(las.classification)

        _tree_ids, _bush_ids = seg.segment(xyz, labels)


if __name__ == "__main__":
    main()
