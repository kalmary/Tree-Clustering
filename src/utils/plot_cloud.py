import sys
import time
from typing import Optional

import numpy as np
import pyvista as pv


DEFAULT_BUFFER_SIZE = 500_000


def _show_native(plotter: pv.Plotter) -> None:
    if sys.platform != "darwin":
        plotter.show()
        return

    closed = {"value": False}

    def mark_closed(*_):
        closed["value"] = True

    plotter.iren.interactor.AddObserver("ExitEvent", mark_closed)
    plotter.iren.interactor.AddObserver("DeleteEvent", mark_closed)

    plotter.show(interactive_update=True, auto_close=False)

    try:
        while not closed["value"]:
            plotter.update(stime=10, force_redraw=False)
            time.sleep(0.01)
    except KeyboardInterrupt:
        raise
    finally:
        plotter.close()


def _add_point_buffer(
    plotter: pv.Plotter,
    points: np.ndarray,
    labels: Optional[np.ndarray],
    origin: np.ndarray,
    point_size: int,
    render_points_as_spheres: bool,
) -> None:
    local_points = np.ascontiguousarray(points - origin, dtype=np.float32)
    cloud = pv.PolyData(local_points)

    point_kwargs = {
        "point_size": point_size,
        "render_points_as_spheres": render_points_as_spheres,
    }

    if labels is not None:
        cloud["cluster"] = np.ascontiguousarray(labels, dtype=np.float32)
        plotter.add_mesh(
            cloud,
            scalars="cluster",
            cmap="tab20",
            **point_kwargs,
        )
    else:
        plotter.add_mesh(
            cloud,
            color="green",
            **point_kwargs,
        )


def _add_point_buffers(
    plotter: pv.Plotter,
    points: np.ndarray,
    labels: Optional[np.ndarray],
    origin: np.ndarray,
    buffer_size: int,
    point_size: int,
    render_points_as_spheres: bool,
    verbose: bool,
) -> None:
    for start in range(0, len(points), buffer_size):
        stop = min(start + buffer_size, len(points))
        if verbose and len(points) > buffer_size:
            print(f"Adding point buffer {start:,}-{stop:,} / {len(points):,}")
        chunk_labels = labels[start:stop] if labels is not None else None
        _add_point_buffer(
            plotter,
            points[start:stop],
            chunk_labels,
            origin,
            point_size,
            render_points_as_spheres,
        )


def plot_cloud(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    point_size: int = 3,
    render_points_as_spheres: bool = False,
    verbose: bool = False,
):
    points = np.asarray(points)
    labels = np.asarray(labels) if labels is not None else None

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if labels is not None and len(labels) != len(points):
        raise ValueError(
            f"labels must have the same length as points: {len(labels)} != {len(points)}"
        )
    if buffer_size <= 0:
        raise ValueError(f"buffer_size must be positive, got {buffer_size}")

    plotter = pv.Plotter(notebook=False, off_screen=False)
    origin = points.mean(axis=0)
    if verbose:
        print(f"Plot origin subtracted before float32 conversion: {origin}")

    _add_point_buffers(
        plotter,
        points,
        labels,
        origin,
        buffer_size,
        point_size,
        render_points_as_spheres,
        verbose,
    )
    
    plotter.reset_camera()
    _show_native(plotter)
