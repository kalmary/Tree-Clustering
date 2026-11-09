import pathlib as pth

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree


def segment_wood_leaf(
    xyz: NDArray,
    tree_ids: NDArray,
    tree_path: str | pth.Path,
    distance: float,
) -> NDArray[np.int8]:
    """Split tree points against Raycloudtools branch cylinders expanded by distance."""
    labels = np.full(len(xyz), -1, dtype=np.int8)
    tree_indices = np.flatnonzero(tree_ids >= 0)
    if len(tree_indices) == 0:
        return labels
    labels[tree_indices] = 1
    points = xyz[tree_indices].astype(np.float64)
    index = KDTree(points)

    with open(tree_path) as file:
        lines = (
            line.strip()
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        )
        fields = next(lines, "").split(",")
        if fields[:6] != ["x", "y", "z", "radius", "parent_id", "section_id"]:
            raise ValueError(f"Invalid Raycloudtools tree header in {tree_path}")
        tree_count = 0
        for line in lines:
            values = np.array(line.split(","), dtype=np.float64)
            if values.size % len(fields) != 0 or not np.isfinite(values).all():
                raise ValueError(f"Invalid Raycloudtools tree data in {tree_path}")
            segments = values.reshape(-1, len(fields))
            parents = segments[:, 4]
            if (
                len(segments) < 2
                or parents[0] != -1
                or np.any(parents[1:] < 0)
                or np.any(parents[1:] >= np.arange(1, len(segments)))
                or np.any(parents[1:] != np.floor(parents[1:]))
                or np.any(segments[:, 3] < 0)
            ):
                raise ValueError(
                    f"Invalid Raycloudtools branch geometry in {tree_path}"
                )
            tree_count += 1
            for segment, parent in zip(segments[1:], parents[1:].astype(np.intp)):
                start = segments[parent, :3]
                direction = segment[:3] - start
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                direction /= length
                radius = segment[3] + distance
                candidates = np.asarray(
                    index.query_ball_point(
                        (start + segment[:3]) / 2,
                        np.hypot(length / 2 + distance, radius),
                    ),
                    dtype=np.intp,
                )
                offsets = points[candidates] - start
                along = offsets @ direction
                radial = offsets - along[:, None] * direction
                inside = (
                    (along >= -distance)
                    & (along <= length + distance)
                    & (np.einsum("ij,ij->i", radial, radial) < radius**2)
                )
                labels[tree_indices[candidates[inside]]] = 0
        if tree_count == 0:
            raise ValueError(
                f"No branch geometry for assigned tree points in {tree_path}"
            )
    return labels
