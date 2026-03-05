"""
visualize_trees.py
==================
Generates a PDF report with 5 depth-map views per tree, rendered on GPU
using cloud2sideViews_torch.  One tree = one PDF page.

Layout per page (A4 landscape):
  ┌─────────────┬─────────────┬─────────────┐
  │  Front (XZ) │  Side  (YZ) │   Top  (XY) │   row 1
  ├─────────────┼─────────────┼─────────────┤
  │  Diag +45°  │  Diag −45°  │  Statistics │   row 2
  └─────────────┴─────────────┴─────────────┘

Usage
-----
    from visualize_trees import save_tree_projections_pdf

    save_tree_projections_pdf(
        points      = vegetation,
        labels      = tree_labels,
        source_name = "forest.las",
        output_path = "trees.pdf",
        resolution  = 256,
        device      = torch.device("cuda"),
    )

CLI:
    python visualize_trees.py --las forest.las --output trees.pdf
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


# ──────────────────────────────────────────────────────────────────────────────
# Depth-map renderer
# ──────────────────────────────────────────────────────────────────────────────

def _shift_positive(t: torch.Tensor) -> torch.Tensor:
    """Shift tensor values so the minimum is 1.0  (fixes negative-depth bug)."""
    return t - t.min() + 1.0


def cloud2sideViews_torch(
    points: torch.Tensor,
    resolution_xy: int,
    margin_ratio: float = 0.05,
) -> torch.Tensor:
    """
    Original cloud2sideViews_torch with one minimal fix:
      depth tensors are shifted to [1, …] before being passed to
      build_depth_map so that the  `nonzero_mask = img > 0`  check
      works correctly even when raw coordinates are negative (e.g. Z = -12 m).

    build_depth_map itself is left byte-for-byte identical to the original.
    Diagonal views use the same cube_min/cube_max as the original.
    """
    points = points.type(torch.float32)

    min_xyz   = points.min(dim=0).values
    max_xyz   = points.max(dim=0).values
    center    = (min_xyz + max_xyz) / 2

    # Centre the cloud so all projections (including rotated diagonals)
    # are symmetric around the image centre.
    points    = points - center

    max_range = (max_xyz - min_xyz).max()
    cube_half = max_range / 2 * (1 + 2 * margin_ratio)
    cube_min  = torch.full((3,), -cube_half.item(), device=points.device)
    cube_max  = torch.full((3,),  cube_half.item(), device=points.device)

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

    # ── original build_depth_map, untouched ──────────────────────────────────
    def build_depth_map(indices_2d, distances, flip_y=False, flip_x=False):
        y_idx, x_idx = indices_2d
        if flip_y:
            y_idx = resolution_xy - 1 - y_idx
        if flip_x:
            x_idx = resolution_xy - 1 - x_idx

        flat_indices = y_idx * resolution_xy + x_idx
        depth_map = torch.full(
            (resolution_xy * resolution_xy,), float('inf'),
            dtype=torch.float32, device=distances.device
        )
        depth_map = torch.scatter_reduce(
            depth_map, 0, flat_indices, distances,
            reduce='amin', include_self=True
        )

        img = depth_map.view(resolution_xy, resolution_xy)
        img[img == float('inf')] = 0  # Replace untouched pixels

        # Normalize non-zero pixels to [0, 1]
        nonzero_mask = img > 0
        if torch.any(nonzero_mask):
            values  = img[nonzero_mask]
            min_val = values.min()
            max_val = values.max()
            img[nonzero_mask] = (values - min_val) / (max_val - min_val + 1e-8)

        return img
    # ─────────────────────────────────────────────────────────────────────────

    # ── 1. Front  XZ  (depth = Y) ────────────────────────────────────────────
    views.append(build_depth_map((gz, gx), _shift_positive(y), flip_y=True))

    # ── 2. Side   YZ  (depth = X) ────────────────────────────────────────────
    views.append(build_depth_map((gz, gy), _shift_positive(x), flip_y=True))

    # ── 3. Top    XY  (depth = Z, brighter = higher) ─────────────────────────
    views.append(build_depth_map((gy, gx), _shift_positive(z), flip_y=False))

    # ── 4. Diagonal +45° (original cube bounds, same as original) ────────────
    angle        = math.pi / 4
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    xr =  cos_a * x + sin_a * y
    yr = -sin_a * x + cos_a * y
    gxr = to_grid(xr, cube_min[0], cube_max[0])   # original bounds, no stretch
    views.append(build_depth_map((gz, gxr), _shift_positive(yr), flip_y=True))

    # ── 5. Diagonal -45° ─────────────────────────────────────────────────────
    xr2 =  cos_a * x - sin_a * y
    yr2 =  sin_a * x + cos_a * y
    gxr2 = to_grid(xr2, cube_min[0], cube_max[0])  # original bounds
    views.append(build_depth_map((gz, gxr2), _shift_positive(yr2), flip_y=True))

    return torch.stack(views)   # (5, R, R)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

VIEW_TITLES = [
    "Front  (XZ)",
    "Side   (YZ)",
    "Top    (XY)",
    "Left  +45°",
    "Right  −45°",
]

DEPTH_CMAP = "viridis"


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def save_tree_projections_pdf(
    points:              np.ndarray | torch.Tensor,
    labels:              np.ndarray | torch.Tensor,
    source_name:         str = "point_cloud",
    output_path:         str | pathlib.Path = "trees.pdf",
    resolution:          int   = 256,
    margin_ratio:        float = 0.05,
    max_points_per_tree: int   = 80_000,
    dpi:                 int   = 180,
    device:              torch.device = torch.device("cpu"),
) -> None:
    """
    Render 5 GPU depth-map views per tree, one tree per PDF page.

    Parameters
    ----------
    points              : (N, 3) XYZ array.
    labels              : (N,)   cluster IDs; values > 0 = trees.
    source_name         : filename shown as subtitle on every page.
    output_path         : destination PDF.
    resolution          : depth-map side length in pixels.
    margin_ratio        : padding around the point cloud cube.
    max_points_per_tree : subsample threshold (keeps GPU memory low).
    dpi                 : PDF render resolution.
    device              : torch device — use cuda for speed.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels)

    unique_labels = np.sort(np.unique(labels[labels > 0]))
    n_trees       = len(unique_labels)

    if n_trees == 0:
        print("[VIZ] No trees found (labels > 0 required).")
        return

    print(f"[VIZ] {n_trees} trees  |  device={device}  |  res={resolution}px")
    print(f"[VIZ] Output → {output_path.resolve()}")

    fig_w, fig_h = 15.0, 8.5   # A4 landscape inches

    with PdfPages(output_path) as pdf:
        for tree_no, tree_id in enumerate(unique_labels, start=1):

            # ── extract & optionally subsample ───────────────────────────────
            mask     = labels == tree_id
            tree_pts = points[mask].copy()
            n_raw    = len(tree_pts)

            if n_raw > max_points_per_tree:
                idx      = np.random.choice(n_raw, max_points_per_tree, replace=False)
                tree_pts = tree_pts[idx]

            # ── GPU rendering ─────────────────────────────────────────────────
            t_pts = torch.from_numpy(tree_pts).to(device)
            with torch.no_grad():
                views = cloud2sideViews_torch(t_pts, resolution, margin_ratio)
            views_np = views.cpu().numpy()   # (5, R, R)  values in [0, 1]

            # ── stats ─────────────────────────────────────────────────────────
            h      = float(tree_pts[:, 2].max() - tree_pts[:, 2].min())
            wx     = float(tree_pts[:, 0].max() - tree_pts[:, 0].min())
            wy     = float(tree_pts[:, 1].max() - tree_pts[:, 1].min())
            z_base = float(tree_pts[:, 2].min())

            # ── figure ────────────────────────────────────────────────────────
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

            fig.suptitle(
                f"Tree  {tree_id}   ({tree_no} / {n_trees})",
                fontsize=15, fontweight="bold", color="#111111", y=0.98,
            )
            fig.text(
                0.5, 0.935,
                (f"Source: {source_name}   |   points: {n_raw:,}   "
                 f"height: {h:.1f} m   width X: {wx:.1f} m   "
                 f"width Y: {wy:.1f} m   Z base: {z_base:.1f} m"),
                ha="center", va="top", fontsize=8.5, color="#444444",
            )

            gs = gridspec.GridSpec(
                2, 3,
                figure=fig,
                left=0.03, right=0.955,
                top=0.88,  bottom=0.04,
                wspace=0.05, hspace=0.22,
            )

            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

            for vi, (row, col) in enumerate(positions):
                ax  = fig.add_subplot(gs[row, col])
                img = views_np[vi]

                ax.imshow(img, cmap=DEPTH_CMAP, origin="upper",
                          interpolation="nearest", vmin=0, vmax=1)
                ax.set_title(VIEW_TITLES[vi], fontsize=9.5, color="#111111",
                             pad=5, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#bbbbbb")
                    spine.set_linewidth(0.7)

            # ── stats panel (bottom-right) ────────────────────────────────────
            ax_s = fig.add_subplot(gs[1, 2])
            ax_s.axis("off")
            for spine in ax_s.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#dddddd")
                spine.set_linewidth(0.6)

            stat_rows = [
                ("Tree ID",        str(tree_id)),
                ("Points (raw)",   f"{n_raw:,}"),
                ("Points (used)",  f"{len(tree_pts):,}"),
                ("Height",         f"{h:.2f} m"),
                ("Width X",        f"{wx:.2f} m"),
                ("Width Y",        f"{wy:.2f} m"),
                ("Z base",         f"{z_base:.2f} m"),
                ("Resolution",     f"{resolution} \u00d7 {resolution} px"),
                ("Device",         str(device)),
            ]

            ax_s.text(0.08, 0.94, "Statistics",
                      transform=ax_s.transAxes,
                      fontsize=10, fontweight="bold", color="#111111", va="top")

            y_cur = 0.82
            for lbl_txt, val_txt in stat_rows:
                ax_s.text(0.08, y_cur, f"{lbl_txt}:",
                          transform=ax_s.transAxes,
                          fontsize=8, color="#555555", va="top")
                ax_s.text(0.58, y_cur, val_txt,
                          transform=ax_s.transAxes,
                          fontsize=8, color="#111111", va="top", fontweight="bold")
                y_cur -= 0.088

            # ── colourbar ─────────────────────────────────────────────────────
            cbar_ax = fig.add_axes([0.962, 0.04, 0.012, 0.84])
            sm      = plt.cm.ScalarMappable(
                cmap=DEPTH_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cbar_ax)
            cb.set_label("Normalised depth", fontsize=7, color="#444444")
            cb.ax.tick_params(labelsize=6, colors="#444444")
            cb.outline.set_edgecolor("#cccccc")

            pdf.savefig(fig, dpi=dpi, facecolor="white")
            plt.close(fig)

            if tree_no % 10 == 0 or tree_no == n_trees:
                print(f"  {tree_no}/{n_trees} rendered")

    print(f"[VIZ] Saved → {output_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="5-view GPU depth-map PDF for segmented trees."
    )

    parser.add_argument("--path2src", required=True, help="Input path .las file.")
    parser.add_argument("--output",     default="trees.pdf")
    parser.add_argument("--veg-class",  type=int,   default=7)
    parser.add_argument("--resolution", type=int,   default=256)
    parser.add_argument("--max-pts",    type=int,   default=80_000)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import laspy
    import sys
    import os
    src_path = pathlib.Path(__file__).parent.parent
    os.path.append(str(src_path))

    from src.array_processing_RE import TreeSegmRay

    args.path2src = pathlib.Path(args.path2src)
    args.pth2src = [path for path in args.path2src.iterdir()] if args.path2src.is_dir() else [args.path2src]

    seg = TreeSegmRay(height_min=1.5, max_diameter=0.95, distance_limit=0.2, use_rays=True, ground_label=1, tree_label=7, verbose = True)
    seg.start_container()

    for path in args.pth2src:
        las   = laspy.read(path)
        pts   = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
        cls   = np.array(las.classification).astype(np.int32)
        veg   = pts[cls == args.veg_class]
        dev   = torch.device(args.device)
        fname = pathlib.Path(args.las).name

        treeids = seg.segment(veg, cls)

        save_tree_projections_pdf(
            points=veg,
            labels=treeids,
            source_name=fname,
            output_path=args.output,
            resolution=args.resolution,
            max_points_per_tree=args.max_pts,
            device=dev,
        )

    seg.rm_container()


if __name__ == "__main__":
    _cli()