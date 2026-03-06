import shutil
import subprocess
import tempfile
import os
import uuid
import struct

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import yaml



class TreeSegmRay:
    def __init__(
        self,
        height_min: float = 2.0,
        max_diameter: float = 0.9,
        crop_length: float = 1.0,
        distance_limit: float = 1.0,
        girth_height_ratio: float = 0.12,
        gravity_factor: float = 0.3,
        global_taper: float = None,
        global_taper_factor: float = None,
        grid_width: float = None,
        use_rays: bool = False,
        segment_branches: bool = False,
        ground_label: int = None,
        tree_label: int = None,
        verbose: bool = False
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

        if self.verbose:
            print(f"[TreeSegmRay] Backend: {self._backend}")

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    def start_container(self):
        """
        Start a persistent Docker container so segment() calls reuse it
        instead of spinning up a new container each time (~1-2s saved per call).
        Call rm_container() when you are done.
        No-op if backend is native or container is already running.
        """
        if self._backend != "docker":
            if self.verbose:
                print("[TreeSegmRay] start_container() ignored — using native backend.")
            return
        if self._container_name is not None:
            if self.verbose:
                print(f"[TreeSegmRay] Container '{self._container_name}' already running.")
            return

        self._shared_tmpdir  = tempfile.mkdtemp(prefix="treesegmray_persistent_", dir=os.path.expanduser("~"))
        self._container_name = f"treesegmray_{uuid.uuid4().hex[:8]}"
        subprocess.run([
            "docker", "run", "-d",
            "--name", self._container_name,
            "-v",     f"{self._shared_tmpdir}:/data",
            "ghcr.io/csiro-robotics/raycloudtools:latest",
            "sleep", "infinity",
        ], check=True, capture_output=not self.verbose)

        if self.verbose:
            print(f"[TreeSegmRay] Container '{self._container_name}' started.")

    def rm_container(self):
        """
        Stop and remove the persistent Docker container and its temp directory.
        Safe to call even if no container is running.
        """
        if self._container_name:
            subprocess.run(["docker", "rm", "-f", self._container_name],
                           capture_output=True)
            if self.verbose:
                print(f"[TreeSegmRay] Container '{self._container_name}' removed.")
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
            # start daemon if not running
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
            def to_container(arg):
                if os.path.isabs(arg):
                    return "/data/" + os.path.basename(arg)
                return arg

            if self._container_name:
                cmd = ["docker", "exec", self._container_name] + \
                      [to_container(a) for a in cmd]
            else:
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{workdir}:/data",
                    "ghcr.io/csiro-robotics/raycloudtools:latest",
                ] + [to_container(a) for a in cmd]

        if self.verbose:
            print(f"[TreeSegmRay] $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd, capture_output=not self.verbose,
            text=True, cwd=workdir,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"rayextract failed (exit {result.returncode}):\n"
                f"{result.stderr or '(no stderr)'}"
            )

    # ------------------------------------------------------------------
    # Internal PLY helpers — purely a transport format for rayextract
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
        """Write a binary PLY mesh — rayextract only accepts binary format."""
        tri   = Delaunay(ground_xyz[:, :2])
        verts = ground_xyz.astype(np.float32)
        faces = tri.simplices.astype(np.int32)

        with open(path, "wb") as f:
            # header (ASCII)
            f.write((
                "ply\nformat binary_little_endian 1.0\n"
                "comment generated by TreeSegmRay\n"
                f"element vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {len(faces)}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            ).encode("ascii"))
            # vertices — 3 × float32 each
            f.write(verts.tobytes())
            # faces — uchar count (always 3) + 3 × int32
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

    def _connect_floating_clusters(self, tree_labels: np.ndarray, tree_xyz: np.ndarray,
                                ground_xyz: np.ndarray,
                                ground_z_threshold: float = 0.5,
                                min_cluster_size: int = 5000,
                                max_tilt_deg: float = 30.0) -> np.ndarray:
        """
        max_tilt_deg: maximum tilt angle from vertical (degrees) to still
                    consider a grounded cluster as belonging below a floating one.
        """
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

        tilt_tolerance = np.tan(np.deg2rad(max_tilt_deg))  # XY per unit Z
        result  = tree_labels.copy()
        kdtree  = cKDTree(grounded_centroids)

        for i, lbl in enumerate(floating):
            fc         = floating_centroids[i]
            below_mask = grounded_centroids[:, 2] < fc[2]

            if below_mask.any():
                candidates    = grounded_centroids[below_mask]
                candidate_ids = grounded[below_mask]

                dz          = fc[2] - candidates[:, 2]
                dxy         = np.linalg.norm(fc[:2] - candidates[:, :2], axis=1)
                tilt_score  = dxy - tilt_tolerance * dz
                best        = np.argmin(tilt_score)
                target      = candidate_ids[best]
            else:
                _, nn  = kdtree.query(fc, k=1)
                target = grounded[nn]

            result[result == lbl] = target

        return result
    
    def _reduce_labels(self, labels: np.ndarray) -> np.ndarray:
        _, labels[labels!=-1] = np.unique(labels[labels!=-1], return_inverse=True) # move from 0 to -1
        labels[labels!=-1] -= 1
        
        return labels
    
    def _remove_small_clusters(self, tree_labels: np.ndarray,
                                min_points: int = 100) -> np.ndarray:
        result = tree_labels.copy()
        for lbl in np.unique(tree_labels):
            mask = tree_labels == lbl
            if mask.sum() < min_points:
                result[mask] = -1
        return result

    @staticmethod
    def _estimate_ground(tree_xyz: np.ndarray, grid_size: float = 2.0) -> np.ndarray:
        """
        Estimate ground surface from lowest points in a 2D grid.
        Used when no ground labels are available.
        """
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
                    lowest = tree_xyz[mask][tree_xyz[mask, 2].argmin()]
                    ground_pts.append(lowest)

        return np.array(ground_pts, dtype=np.float32)


    def load_from_config(self, config: dict):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
            
        return self


    def segment(self, xyz: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        if labels is not None and self.tree_label is not None and self.ground_label is not None:
            tree_mask   = labels == self.tree_label
            ground_mask = labels == self.ground_label
            tree_xyz    = xyz[tree_mask].copy()
            ground_xyz  = xyz[ground_mask].copy()
        else:
            tree_xyz   = xyz.copy()
            ground_xyz = None
            tree_mask  = np.ones(len(xyz), dtype=bool)

        if self.verbose:
            if ground_xyz is not None:
                print(f"[TreeSegmRay] Trees: {len(tree_xyz):,} pts  Ground: {len(ground_xyz):,} pts")
            else:
                print(f"[TreeSegmRay] Trees: {len(tree_xyz):,} pts  Ground: estimated from lowest points")

        xy_mean           = tree_xyz[:, :2].mean(axis=0)
        tree_xyz[:, :2]  -= xy_mean

        if ground_xyz is not None and len(ground_xyz) > 0:
            ground_xyz[:, :2] -= xy_mean
        else:
            ground_xyz = self._estimate_ground(tree_xyz)

        own_tmpdir = self._shared_tmpdir is None
        tmpdir     = tempfile.mkdtemp(prefix="treesegmray_", dir=os.path.expanduser("~")) if own_tmpdir \
                    else self._shared_tmpdir

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
            if self.verbose:
                cmd.append("--verbose")

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
                ground_z_threshold=0.5,
                min_cluster_size=500
            )
            tree_instance_labels = self._remove_small_clusters(tree_instance_labels, min_points=1500)
            tree_instance_labels = self._reduce_labels(tree_instance_labels)

        finally:
            if own_tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return tree_instance_labels

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

def main():
    import laspy
    from utils.plot_cloud import plot_cloud

    seg = TreeSegmRay(height_min=0.9, max_diameter=0.8, distance_limit=0.25, gravity_factor=0.6, use_rays=False, ground_label=1, tree_label=7, verbose = False)
    seg.start_container()

    for path in ["data/split/ITWL_Grajewo19_cut_small.laz"]:
        las = laspy.read(path)
        xyz   = np.vstack([las.x, las.y, las.z]).T
        labels = np.asarray(las.classification)

        # plot_cloud(xyz, labels)

        tree_xyz    = xyz[labels == seg.tree_label]
        tree_labels = seg.segment(xyz, labels)
        # plot_cloud(tree_xyz, treeqqq_labels)

        for tree in np.unique(tree_labels):
            mask = tree_labels == tree
            print(f"Tree {tree}: {mask.sum()} points")
            plot_cloud(tree_xyz[mask], tree_labels[mask])

    seg.rm_container()


if __name__ == "__main__":
    main()