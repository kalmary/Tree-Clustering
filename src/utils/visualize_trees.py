"""
visualize_trees.py
==================
Renders 5 depth-map views per tree to a PDF report.
One tree = one page.

Layout (A4 landscape):
  ┌─────────────┬─────────────┬─────────────┐
  │  Front (XZ) │  Back  (XZ) │   Top  (XY) │
  ├─────────────┼─────────────┼─────────────┤
  │  Left  (YZ) │  Right (YZ) │  Statistics │
  └─────────────┴─────────────┴─────────────┘

Usage
-----
    python visualize_trees.py --path2src forest.laz --output trees.pdf
    python visualize_trees.py --path2src ./scans/   --output trees.pdf
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Depth-map renderer
# ──────────────────────────────────────────────────────────────────────────────

def cloud2sideViews_torch(points: torch.Tensor, resolution_xy: int, margin_ratio: float = 0.05) -> torch.Tensor:
    points = points.type(torch.float64)
    points -= points.mean(dim=0, keepdim=True)

    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values

    center = (min_xyz + max_xyz) / 2
    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2 * (1 + 2 * margin_ratio)

    cube_min = center - cube_half
    cube_max = center + cube_half


    def to_grid(val, min_val, max_val):
        return torch.clamp(
            ((val - min_val) / (max_val - min_val + 1e-8) * (resolution_xy - 1)).long(),
            0, resolution_xy - 1
        )

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    gx = to_grid(x, cube_min[0], cube_max[0])
    gy = to_grid(y, cube_min[1], cube_max[1])
    gz = to_grid(z, cube_min[2], cube_max[2])

    views = []

    def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
        y_idx, x_idx = indices_2d
        if flip_y:
            y_idx = resolution_xy - 1 - y_idx
        if flip_x:
            x_idx = resolution_xy - 1 - x_idx

        flat_indices = y_idx * resolution_xy + x_idx
        depth_map = torch.full((resolution_xy * resolution_xy,), float('inf'), dtype=torch.float64,
                               device=distances.device)

        # Use scatter_reduce to keep the minimum distance per pixel
        depth_map = torch.scatter_reduce(depth_map, 0, flat_indices, distances, reduce='amin', include_self=True)

        img = depth_map.view(resolution_xy, resolution_xy)
        img[img == float('inf')] = 0  # Replace untouched pixels

        # Normalize non-zero pixels to [0, 1]
        nonzero_mask = img > 0
        if torch.any(nonzero_mask):
            values = img[nonzero_mask]
            min_val = values.min()
            max_val = values.max()
            img[nonzero_mask] = (values - min_val) / (max_val - min_val + 1e-8)

        return img

    # Compute distance from each wall
    dist_top = cube_max[2] - z
    views.append(build_depth_map((gy, gx), dist_top))

    dist_front = cube_max[1] - y
    views.append(build_depth_map((gz, gx), dist_front, flip_y=True))


    dist_back = y - cube_min[1]
    views.append(build_depth_map((gz, gx), dist_back, flip_y=True, flip_x=True))

    dist_left = cube_max[0] - x
    views.append(build_depth_map((gz, gy), dist_left, flip_y=True))

    dist_right = x - cube_min[0]
    views.append(build_depth_map((gz, gy), dist_right, flip_y=True, flip_x=True))

    return torch.stack(views, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# PDF generator
# ──────────────────────────────────────────────────────────────────────────────

VIEW_TITLES = ["TOP", "FRONT", "BACK", "Left", "Right"]
CMAP        = "viridis"


def save_tree_projections_pdf(
    points:              np.ndarray,
    labels:              np.ndarray,
    las_idx:             Optional[int],
    source_name:         str,
    output_name:         str | pathlib.Path,
    output_dir:          str | pathlib.Path,
    resolution:          int          = 256,
    margin_ratio:        float        = 0.05,
    max_points_per_tree: int          = 100000,
    dpi:                 int          = 150,
    device:              str          = torch.device("cpu"),
) -> None:
    
    output_path = output_dir / output_name
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cloud_dir = output_dir / "clouds"
    cloud_dir.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)

    unique_labels = np.sort(np.unique(labels[labels >= 0]))
    n_trees       = len(unique_labels)

    if n_trees == 0:
        print("[VIZ] No trees found (labels >= 0 required).")
        return

    print(f"[VIZ] {n_trees} trees | device={device} | res={resolution}px")
    print(f"[VIZ] Output → {output_path.resolve()}")

    with PdfPages(output_path) as pdf:
        for tree_no, tree_id in enumerate(unique_labels, start=1):

            mask     = labels == tree_id
            tree_pts = points[mask].copy()
            n_raw    = len(tree_pts)

            tree_name = cloud_dir.joinpath(f"{las_idx}_{tree_id}.npy")

            if n_raw > max_points_per_tree:
                idx      = np.random.choice(n_raw, max_points_per_tree, replace=False)
                tree_pts = tree_pts[idx]

            np.save(tree_name, tree_pts)

            # stats before centring so Z base is meaningful
            h      = float(tree_pts[:, 2].max() - tree_pts[:, 2].min())
            wx     = float(tree_pts[:, 0].max() - tree_pts[:, 0].min())
            wy     = float(tree_pts[:, 1].max() - tree_pts[:, 1].min())
            z_base = float(tree_pts[:, 2].min())

            t_pts = torch.from_numpy(tree_pts).to(device)
            with torch.no_grad():
                views = cloud2sideViews_torch(t_pts, resolution, margin_ratio)
            views_np = views.cpu().numpy()

            fig = plt.figure(figsize=(15.0, 8.5), facecolor="white")
            fig.suptitle(
                f"Tree {tree_id}   ({tree_no} / {n_trees})",
                fontsize=15, fontweight="bold", y=0.98,
            )
            fig.text(
                0.5, 0.935,
                (f"File: {source_name}   |   pts: {n_raw:,}   "
                 f"H: {h:.1f} m   W-X: {wx:.1f} m   W-Y: {wy:.1f} m   "
                 f"Z base: {z_base:.1f} m"),
                ha="center", fontsize=8.5, color="#444444",
            )

            gs = gridspec.GridSpec(
                2, 3, figure=fig,
                left=0.03, right=0.955, top=0.88, bottom=0.04,
                wspace=0.05, hspace=0.22,
            )

            for vi, (row, col) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1)]):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(views_np[vi], cmap=CMAP, origin="upper",
                          interpolation="nearest", vmin=0, vmax=1)
                ax.set_title(VIEW_TITLES[vi], fontsize=9.5, fontweight="bold", pad=5)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor("#bbbbbb"); sp.set_linewidth(0.7)

            # stats panel
            ax_s = fig.add_subplot(gs[1, 2])
            ax_s.axis("off")
            stat_rows = [
                ("Tree ID",       f"{str(tree_id)}"),
                ("Tree file name", tree_name.name),
                ("Points (raw)",  f"{n_raw:,}"),
                ("Points (used)", f"{len(tree_pts):,}"),
                ("Height",        f"{h:.2f} m"),
                ("Width X",       f"{wx:.2f} m"),
                ("Width Y",       f"{wy:.2f} m"),
                ("Z base",        f"{z_base:.2f} m"),
                ("Resolution",    f"{resolution} x {resolution} px")
            ]
            ax_s.text(0.08, 0.94, "Statistics", transform=ax_s.transAxes,
                      fontsize=10, fontweight="bold", va="top")
            y_cur = 0.82
            for lbl_txt, val_txt in stat_rows:
                ax_s.text(0.08, y_cur, f"{lbl_txt}:", transform=ax_s.transAxes,
                          fontsize=8, color="#555555", va="top")
                ax_s.text(0.58, y_cur, val_txt, transform=ax_s.transAxes,
                          fontsize=8, fontweight="bold", va="top")
                y_cur -= 0.088

            # colorbar
            # cbar_ax = fig.add_axes([0.962, 0.04, 0.012, 0.84])
            # sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
            # sm.set_array([])
            # cb = fig.colorbar(sm, cax=cbar_ax)
            # cb.set_label("Normalised depth", fontsize=7, color="#444444")
            # cb.ax.tick_params(labelsize=6, colors="#444444")

            pdf.savefig(fig, dpi=dpi, facecolor="white")
            plt.close(fig)

            if tree_no % 10 == 0 or tree_no == n_trees:
                print(f"  {tree_no}/{n_trees} rendered")

    print(f"[VIZ] Saved → {output_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import laspy
    import pathlib as pth
    
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from array_processing_RE import TreeSegmRay

    path   = "data/split/ITWL_Grajewo19_cut_small.laz"
    path = pth.Path(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg = TreeSegmRay(
        height_min=0.9, max_diameter=0.8, distance_limit=0.25,
        gravity_factor=0.6, use_rays=False,
        ground_label=1, tree_label=7, verbose=True,
    )
    seg.start_container()

    try:
        las         = laspy.read(path)
        pts         = np.vstack([las.x, las.y, las.z]).T
        cls         = np.asarray(las.classification, dtype=np.int32)

        tree_labels = seg.segment(pts, cls)
        tree_xyz    = pts[cls == seg.tree_label]

        save_tree_projections_pdf(
            points      = tree_xyz[tree_labels != -1],
            labels      = tree_labels[tree_labels != -1],
            las_idx     = 0,
            source_name = pathlib.Path(path).name,
            output_dir  = path.parent,
            output_path = "output.pdf",
            resolution  = 350,
            device      = device
        )
    finally:
        seg.rm_container()


if __name__ == "__main__":
    main()