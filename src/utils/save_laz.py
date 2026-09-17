# pyright: basic

from pathlib import Path

import laspy
import numpy as np


def save_laz(
    las: laspy.LasData,
    ids: np.ndarray,
    output_path: str | Path,
    field_name: str = "tree_ids",
) -> Path:
    """Save a LAZ file with an integer point-ID field."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or len(ids) != len(las.points):
        raise ValueError(
            f"ids must contain one value per point: {ids.shape} for {len(las.points)} points"
        )

    if field_name not in las.point_format.dimension_names:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name=field_name,
                type=np.dtype(np.int32),
                description="Tree instance ID",
            )
        )

    las[field_name] = ids.astype(np.int32, copy=False)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
    return path
