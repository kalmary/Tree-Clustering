"""
Plots a point cloud using pyvista.

1. Receives an input point cloud as a numpy.ndarray(shape=(n, 3), dtype=numpy.float32)
2. Optionally, receives a feature as a one-dimensional numpy.ndarray or a
   numpy.ndarray(shape=(n, 1)).
3. Optionally, displays a title above the plot.
4. Colors displayed:
    - Integer features use the existing discrete tab20 colors.
    - Floating-point features use gradient coloring.
    - If no feature is provided, all points use the same color (green).
    - Background color is always set to white.
5. Splits large point clouds into buffers before rendering to avoid exceeding
   data-transfer limits. The same buffering works on Linux and macOS.
"""

import sys
from typing import Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

BUFFER = 500_000

def plot_cloud(
    points: NDArray[np.float32],
    feature: NDArray[np.number[Any]] | None = None,
    title: str | None = None,
    *,
    buffer_size: int = BUFFER,
) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than zero")

    values: NDArray[np.number[Any]] | None = None
    if feature is not None:
        values = np.asarray(feature)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise ValueError("feature must have shape (n,) or (n, 1)")
        if values.shape[0] != points.shape[0]:
            raise ValueError("feature must contain one value per point")

    if "ipykernel" in sys.modules:
        pv.set_jupyter_backend("trame")
    plotter = pv.Plotter()
    if title is not None:
        plotter.add_title(title)

    color_limits = None
    continuous = values is not None and np.issubdtype(values.dtype, np.floating)
    if values is not None and values.size:
        valid_values = values[np.isfinite(values)] if continuous else values
        if valid_values.size:
            color_limits = (float(valid_values.min()), float(valid_values.max()))

    for start in range(0, points.shape[0], buffer_size):
        end = min(start + buffer_size, points.shape[0])
        point_buffer = points[start:end]
        vertices = pv.CellArray.from_regular_cells(
            np.arange(point_buffer.shape[0], dtype=pv.ID_TYPE).reshape(-1, 1)
        )
        cloud = pv.PolyData(point_buffer, verts=vertices)

        if values is None:
            plotter.add_mesh(
                cloud,
                color="green",
                point_size=5,
                render_points_as_spheres=True,
            )
            continue

        scalar_name = "feature" if continuous else "cluster"
        plotter.add_mesh(
            cloud,
            scalars=values[start:end],
            clim=color_limits,
            cmap="viridis" if continuous else "tab20",
            point_size=5,
            render_points_as_spheres=True,
            scalar_bar_args={"title": scalar_name},
            show_scalar_bar=start == 0,
        )

    try:
        plotter.show()
    finally:
        plotter.close()
        plotter.deep_clean()
