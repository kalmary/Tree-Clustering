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


def propagate_wood(
    xyz: NDArray,
    tree_ids: NDArray,
    labels: NDArray[np.int8],
    ground_xyz: NDArray,
    height: float,
    radius: float,
    min_points: int,
) -> NDArray[np.int8]:
    """Grow wood labels within each tree's lower band until no leaves qualify."""
    result = labels.copy()
    indices = np.flatnonzero((tree_ids >= 0) & (labels >= 0))
    if len(indices) == 0:
        return result
    if len(ground_xyz) == 0:
        raise ValueError("Wood propagation requires ground points")

    _, nearest = KDTree(ground_xyz[:, :2]).query(xyz[indices, :2])
    heights = xyz[indices, 2] - ground_xyz[nearest, 2]
    indices = indices[heights <= height]
    for tree_id in np.unique(tree_ids[indices]):
        tree_indices = indices[tree_ids[indices] == tree_id]
        wood = tree_indices[result[tree_indices] == 0]
        leaves = tree_indices[result[tree_indices] == 1]
        while len(wood) >= min_points and len(leaves) > 0:
            distances, _ = KDTree(xyz[wood]).query(xyz[leaves], k=[min_points])
            promote = distances[:, 0] <= radius
            if not promote.any():
                break
            result[leaves[promote]] = 0
            wood = np.concatenate((wood, leaves[promote]))
            leaves = leaves[~promote]
    return result
