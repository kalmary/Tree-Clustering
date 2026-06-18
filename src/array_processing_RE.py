import shutil
import subprocess
import tempfile
import os
import uuid
import struct

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils.plot_cloud import plot_cloud
from pprint import pprint

import gc

from dataclasses import dataclass, field, asdict
from typing import Optional, Union
import pathlib as pth
import json

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
        global_taper:        Optional[float] = None,
        global_taper_factor: Optional[float] = None,
        grid_width:          Optional[float] = None,
        use_rays:            bool          = False,
        segment_branches:    bool          = False,
        ground_label:        Optional[int] = None,
        tree_label:          Optional[int] = None,
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
    def from_config(cls, cfg: Optional[dict] = None, cfg_path: Optional[Union[str, pth.Path]] = None, verbose: bool = False) -> "TreeSegmRay":
        if cfg is not None:
            return cls(**cfg, verbose=verbose)
        elif cfg is None and cfg_path is not None:
            cfg_path = pth.Path(cfg_path)
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
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
            subprocess.run(["docker", "rm", "-f", self._container_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
            if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
                subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
                subprocess.run(["docker", "info"], check=True)

            r = subprocess.run(
                ["docker", "image", "inspect",
                "ghcr.io/csiro-robotics/raycloudtools:latest"],
                capture_output=True,
            )
            if r.returncode == 0:
                return "docker"
            raise EnvironmentError(
                "Docker found but raycloudtools image not pulled.\n"
            )
        raise EnvironmentError(
            "raycloudtools not found"
        )

    def _run(self, cmd: list, workdir: str):
        if self._backend == "docker":
            if self._container_name:
                def to_container(arg):
                    if os.path.isabs(arg):
                        rel = os.path.relpath(arg, self._shared_tmpdir)
                        return "/data/" + rel
                    return arg
                cmd = ["docker", "exec", self._container_name] + \
                    [to_container(a) for a in cmd]
            else:
                def to_container(arg):
                    if os.path.isabs(arg):
                        return "/data/" + os.path.basename(arg)
                    return arg
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{workdir}:/data",
                    "ghcr.io/csiro-robotics/raycloudtools:latest",
                ] + [to_container(a) for a in cmd]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=workdir,
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
    def _write_raycloud_ply(points: np.ndarray, path: str):
        n      = len(points)
        pts    = points.astype(np.float32)
        nxyz   = np.tile(np.float32([0, 0, 10]), (n, 1))
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
    def _write_ground_mesh_ply(ground_xyz: np.ndarray, path: str):
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
            for face in faces:
                f.write(struct.pack("<B3i", 3, face[0], face[1], face[2]))

    @staticmethod
    def _read_labels_from_segmented_ply(path: str) -> np.ndarray:
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

    def _connect_floating_clusters(self, tree_labels: np.ndarray, tree_xyz: np.ndarray,
                                    ground_xyz: np.ndarray,
                                    ground_z_threshold: float = 0.5,
                                    min_cluster_size: int = 5000,
                                    max_tilt_deg: float = 30.0) -> np.ndarray:
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
        kdtree  = cKDTree(grounded_centroids)

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

    def _reduce_labels(self, labels: np.ndarray) -> np.ndarray:
        _, labels[labels != -1] = np.unique(labels[labels != -1], return_inverse=True)
        labels[labels != -1] -= 1
        return labels

    def _remove_small_clusters(self, tree_labels: np.ndarray,
                                min_points: int = 100) -> np.ndarray:
        result = tree_labels.copy()
        for lbl in np.unique(tree_labels):
            if (tree_labels == lbl).sum() < min_points:
                result[tree_labels == lbl] = -1
        return result

    @staticmethod
    def _estimate_ground(tree_xyz: np.ndarray, grid_size: float = 2.0) -> np.ndarray:
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
    def _voxel_tiles(xyz: np.ndarray, voxel_size: float = 50.0, overlap: float = 3.0):
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
    def _core_mask(xyz: np.ndarray, tile: dict) -> np.ndarray:
        return (
            (xyz[:, 0] >= tile["cx_min"]) & (xyz[:, 0] < tile["cx_max"]) &
            (xyz[:, 1] >= tile["cy_min"]) & (xyz[:, 1] < tile["cy_max"])
        )

    @staticmethod
    def _tile_mask(xyz: np.ndarray, tile: dict) -> np.ndarray:
        return (
            (xyz[:, 0] >= tile["x_min"]) & (xyz[:, 0] < tile["x_max"]) &
            (xyz[:, 1] >= tile["y_min"]) & (xyz[:, 1] < tile["y_max"])
        )
    
    def _ensure_container(self):
        if self._backend != "docker" or self._container_name is None:
            return
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", self._container_name],
            capture_output=True, text=True
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
            tree_xyz: np.ndarray,
            trunk_height_band: tuple[float, float] = (0.5, 2.0)) -> np.ndarray | None:
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
        tree_xyz: np.ndarray,
        tree_ids: np.ndarray,
        min_trunk_dist: float = 1.5,
        trunk_height_band: tuple[float, float] = (0.5, 2.0),
        min_points: int = 50) -> np.ndarray:

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
        pairs     = cKDTree(positions).query_pairs(r=min_trunk_dist, output_type="ndarray")
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

    def _segment_watershed(self, tree_xyz: np.ndarray, resolution: float = 0.15) -> np.ndarray:
        """2D watershed fallback on XY crown projection."""
        from scipy.ndimage import label
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

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
        markers, _ = label(mask)
        ws_labels  = watershed(-density_smooth, markers, mask=density > 0)

        # map back to points
        point_labels = ws_labels[idx[:, 0], idx[:, 1]].astype(np.int64) - 1  # 0-indexed, -1 = unlabelled
        return point_labels

    def _segment_small(self,
                       xyz: np.ndarray,
                       labels: np.ndarray = None,
                       debug: bool = False) -> np.ndarray:

        self.start_container()

        xyz -= xyz.mean(axis=0)  # center for better numerical stability in raycloudtools

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
            return np.zeros(0, dtype=np.int64)

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
            return np.full(tree_xyz.shape[0], -1, dtype=np.int64)

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



        return tree_instance_labels

    def _segment_big(self, xyz: np.ndarray, labels: np.ndarray,
                    voxel_size: float = 40.0, overlap: float = 5.0) -> np.ndarray:

        if labels is not None and self.tree_label is not None and self.ground_label is not None:
            tree_mask = labels == self.tree_label
            tree_xyz  = xyz[tree_mask]
        else:
            tree_mask = np.ones(len(xyz), dtype=bool)
            tree_xyz  = xyz

        if tree_xyz.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)

        tree_ids      = np.full(len(tree_xyz), -1, dtype=np.int64)
        treeID_offset = 0

        tiles = list(self._voxel_tiles(tree_xyz, voxel_size=voxel_size, overlap=overlap))
        pbar = tqdm(tiles, desc="Voxel tiles - tree clustering", leave=False, position=1) if self.verbose else tiles

        for tile in pbar:

            ext_mask  = self._tile_mask(xyz, tile)
            mini_xyz  = xyz[ext_mask]
            print(mini_xyz.shape)

            if mini_xyz.shape[0] == 0:
                continue

            mini_labels = labels[ext_mask] if labels is not None else None

            if mini_labels is not None and self.tree_label is not None:
                mini_tree_mask = mini_labels == self.tree_label
            else:
                mini_tree_mask = np.ones(len(mini_xyz), dtype=bool)

            if mini_tree_mask.sum() < int(5e4):
                continue

            core_in_tree = self._core_mask(tree_xyz, tile)
            if core_in_tree.sum() == 0:
                continue
            
            self.rm_container()
            try:
                chunk_tree_ids = self._segment_small(mini_xyz, mini_labels)
            except Exception as e:

                max_tree_dim = 3. # meters
                mini_tree_xyz  = mini_xyz[mini_tree_mask]
                xy_extent      = np.ptp(mini_tree_xyz[:, :2], axis=0)
                is_single_tree = (xy_extent <= max_tree_dim).all()
                chunk_tree_ids = np.zeros(mini_tree_mask.sum(), dtype=np.int64) if is_single_tree else np.full(mini_tree_mask.sum(), -1, dtype=np.int64)

            core_chunk_ids = chunk_tree_ids[self._core_mask(mini_xyz[mini_tree_mask], tile)].copy()

            valid     = core_chunk_ids >= 0
            valid_all = chunk_tree_ids >= 0
            core_chunk_ids[valid] += treeID_offset
            if valid_all.any():
                treeID_offset = int(chunk_tree_ids[valid_all].max()) + treeID_offset + 1

            assign_mask  = core_chunk_ids >= 0
            core_indices = np.where(core_in_tree)[0]
            if assign_mask.any():
                target_indices = core_indices[assign_mask]
                unassigned     = tree_ids[target_indices] == -1
                tree_ids[target_indices[unassigned]] = core_chunk_ids[assign_mask][unassigned]

                del ext_mask, mini_xyz, mini_labels, chunk_tree_ids
                gc.collect()

        tree_ids = self._merge_close_trunks(tree_xyz, tree_ids,
                                            min_trunk_dist=0.3,
                                            trunk_height_band=(0.5, 1.0))
        tree_ids = self._remove_small_clusters(tree_ids, min_points=5000)
        tree_ids = self._reduce_labels(tree_ids)

        return tree_ids

    def _segment_birch(self, xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
        from sklearn.cluster import Birch


        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            return np.full(0, -1, dtype=np.int32)
        
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
        kdtree = cKDTree(tree_xyz_lr)
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
        tree_indices = np.flatnonzero(tree_mask)
        tree_id_offset = 0

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
                tree_ids_voxel = self._segment_small(group_voxel, group_voxel_labels)
            except Exception:
                max_tree_dim = 3.0
                group_voxel_tree = group_voxel[group_voxel_tree_mask]
                xy_extent = np.ptp(group_voxel_tree[:, :2], axis=0)
                is_single_tree = (xy_extent <= max_tree_dim).all()
                tree_ids_voxel = (
                    np.zeros(group_voxel_tree_mask.sum(), dtype=np.int32)
                    if is_single_tree
                    else np.full(group_voxel_tree_mask.sum(), -1, dtype=np.int32)
                )

            group_voxel_tree_indices = group_indices[group_voxel_tree_mask]
            tree_positions_in_voxel = np.searchsorted(tree_indices, group_voxel_tree_indices)
            group_tree_mask_in_voxel = group_ids[tree_positions_in_voxel] == group_id

            group_tree_ids = tree_ids_voxel[group_tree_mask_in_voxel].astype(np.int32, copy=True)
            valid = group_tree_ids >= 0
            if valid.any():
                group_tree_ids[valid] += tree_id_offset
                tree_id_offset = int(group_tree_ids[valid].max()) + 1
            full_tree_ids[tree_positions_in_voxel[group_tree_mask_in_voxel]] = group_tree_ids

            del group_index_chunks, group_indices, group_voxel, group_voxel_labels, tree_ids_voxel
            gc.collect()

        full_tree_ids = self._merge_close_trunks(tree_xyz, full_tree_ids,
                                                 min_trunk_dist=0.3,
                                                 trunk_height_band=(0.5, 1.0))
        full_tree_ids = self._remove_small_clusters(full_tree_ids, min_points=5000)
        full_tree_ids = self._reduce_labels(full_tree_ids)

        return full_tree_ids.astype(np.int32, copy=False)

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


    def segment(self, xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
        full_tree_ids = np.full(len(xyz), -1, dtype=np.int32)
        if xyz.shape[0] == 0:
            return full_tree_ids
        if xyz.shape[0] != labels.shape[0]:
            raise ValueError(f"xyz and labels length mismatch: {xyz.shape[0]} != {labels.shape[0]}")
        
        tree_mask = labels == self.tree_label
        if tree_mask.sum() == 0:
            return full_tree_ids

        xyz = (xyz - xyz.mean(axis=0)).astype(np.float32)


        if xyz[tree_mask].shape[0] > 1e7: # threshold checked
            tree_ids = self._segment_birch(xyz.copy(), labels)
            # tree_ids = self._segment_big(xyz, labels)
        else:
            tree_ids = self._segment_small(xyz, labels)
        full_tree_ids[tree_mask] = tree_ids
        return full_tree_ids


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def main():
    import laspy

    seg = TreeSegmRay(ground_label=1,
                      tree_label=7, verbose=True)

    seg = TreeSegmRay.from_config(cfg_path="src/final_files/config_RE.json", verbose=True)

    for path in ["/Users/michalsiniarski/Documents/DATA/BRIK/ITWL_Grajewo21_mod.laz"]:
        las    = laspy.read(path)
        xyz    = np.vstack([las.x, las.y, las.z]).T
        labels = np.asarray(las.classification)

        tree_xyz    = xyz[labels == seg.tree_label]


        labels = seg.segment(xyz, labels)



        for tree_label in np.unique(labels):
            if tree_label == -1:
                continue
            fake_labels = np.zeros_like(labels)
            mask = labels == tree_label
            fake_labels[mask] = 1

        for tree_xyz in [xyz[labels == tree_label] for tree_label in np.unique(labels) if tree_label != -1]:
            plot_cloud(tree_xyz)


if __name__ == "__main__":
    main()
