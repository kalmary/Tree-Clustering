"""Diagnostic visualization for bush-to-ground filtering decisions."""

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.spatial import KDTree


def plot_bush_diagnostic(
    bush_xyz: NDArray[np.float32],
    cluster_labels: NDArray[np.int32],
    min_heights: NDArray[np.float32],
    xy_bounds: NDArray[np.float32],
    ground_xyz: NDArray[np.float32] | None = None,
    max_gap: float = 1.0,
    boundary_margin: float = 0.5,
    nearest_ground_points: int = 10,
) -> None:
    """Plot clusters retained by ground filtering and their ground gaps."""
    retained_mask = cluster_labels >= 0
    if not retained_mask.any():
        return

    retained_xyz = bush_xyz[retained_mask]
    retained_labels = cluster_labels[retained_mask]
    retained_cluster_ids = np.unique(retained_labels)

    ground_tree = (
        KDTree(ground_xyz[:, :2])
        if ground_xyz is not None and len(ground_xyz) > 0
        else None
    )
    fallback_ground_level = float(min_heights.min())
    plotter = pv.Plotter()
    plotter.add_title(
        f"Bush clusters retained after filtering, max gap={max_gap:.2f} m"
    )
    plotter.add_points(
        retained_xyz,
        scalars=retained_labels,
        cmap="tab20",
        point_size=5,
        render_points_as_spheres=True,
    )

    for cluster_id in retained_cluster_ids:
        bounds = xy_bounds[cluster_id]
        lower = bounds[:2] - boundary_margin
        upper = bounds[2:] + boundary_margin
        centre_xy = (lower + upper) / 2.0

        if ground_tree is not None and ground_xyz is not None:
            radius = float(np.linalg.norm((upper - lower) / 2.0))
            nearby_indices = ground_tree.query_ball_point(centre_xy, r=radius)
            nearby_xy = ground_xyz[nearby_indices, :2]
            inside_box = np.all(
                (nearby_xy >= lower) & (nearby_xy <= upper),
                axis=1,
            )
            box_indices = np.asarray(nearby_indices)[inside_box]
            if len(box_indices) > 0:
                assumed_ground = float(ground_xyz[box_indices, 2].mean())
            else:
                neighbour_count = min(nearest_ground_points, len(ground_xyz))
                _, nearest_indices = ground_tree.query(
                    centre_xy,
                    k=neighbour_count,
                )
                nearest_indices = np.atleast_1d(nearest_indices)
                assumed_ground = float(ground_xyz[nearest_indices, 2].mean())
        else:
            assumed_ground = fallback_ground_level

        min_height = float(min_heights[cluster_id])
        gap = min_height - assumed_ground
        accepted = gap <= max_gap
        line = pv.Line(
            (float(centre_xy[0]), float(centre_xy[1]), assumed_ground),
            (float(centre_xy[0]), float(centre_xy[1]), min_height),
        )
        plotter.add_mesh(
            line,
            color="green" if accepted else "red",
            line_width=5,
        )

    try:
        plotter.show()
    finally:
        plotter.close()
        plotter.deep_clean()
