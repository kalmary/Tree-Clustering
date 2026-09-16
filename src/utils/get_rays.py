# pyright: basic

import numpy as np

CHUNK_SIZE = 5_000_000


def _has_valid_angles(
    scan_angle_rank: np.ndarray,
    indices: np.ndarray | None = None,
) -> bool:
    length = len(scan_angle_rank) if indices is None else len(indices)
    for start in range(0, length, CHUNK_SIZE):
        if indices is None:
            angles = scan_angle_rank[start:start + CHUNK_SIZE]
        else:
            angles = scan_angle_rank[indices[start:start + CHUNK_SIZE]]
        if np.any(np.abs(angles) > 5.0):
            return True
    return False


def _search_time(gps_time: np.ndarray, indices: np.ndarray, value: float) -> int:
    left = 0
    right = len(indices)
    while left < right:
        middle = (left + right) // 2
        if gps_time[indices[middle]] < value:
            left = middle + 1
        else:
            right = middle
    return left


def _flightline_bounds(
    point_source_id: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    changes = []
    previous = None

    for start in range(0, len(order), CHUNK_SIZE):
        sources = point_source_id[order[start:start + CHUNK_SIZE]]
        if previous is not None and sources[0] != previous:
            changes.append(start)
        changes.extend((np.flatnonzero(np.diff(sources) != 0) + start + 1).tolist())
        previous = sources[-1]

    boundaries = np.asarray(changes, dtype=np.int64)
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return starts, ends


def _fill_missing_heights(heights: np.ndarray) -> None:
    previous = np.empty(len(heights), dtype=np.int64)
    following = np.empty(len(heights), dtype=np.int64)

    nearest = -1
    for index in range(len(heights)):
        if np.isfinite(heights[index]):
            nearest = index
        previous[index] = nearest

    nearest = -1
    for index in range(len(heights) - 1, -1, -1):
        if np.isfinite(heights[index]):
            nearest = index
        following[index] = nearest

    for index in np.flatnonzero(~np.isfinite(heights)):
        before = previous[index]
        after = following[index]
        if before >= 0 and after >= 0:
            heights[index] = (heights[before] + heights[after]) / 2.0
        elif before >= 0:
            heights[index] = heights[before]
        else:
            heights[index] = heights[after]


def get_rays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    gps_time: np.ndarray | None,
    point_source_id: np.ndarray | None,
    scan_angle_rank: np.ndarray | None,
) -> np.ndarray | None:
    """
    Computes ray vectors (pointing from the ground point towards the sensor)
    for a LiDAR point cloud.
    
    Args:
        x, y, z: 1D numpy arrays of point coordinates.
        gps_time: 1D numpy array of GPS times.
        point_source_id: 1D numpy array of flightline IDs.
        scan_angle_rank: 1D numpy array of scan angles in degrees.
        
    Returns:
        A (N, 3) numpy array of raw ray vectors (nx, ny, nz) pointing from the
        end point towards the sensor. Returns None when the required flight data
        is missing or insufficient. Vectors are not normalized — the magnitude
        is the actual distance to the estimated sensor position.
    """
    if gps_time is None or point_source_id is None or scan_angle_rank is None:
        return None
        
    n_points = len(x)
    if n_points == 0:
        return None
    if not _has_valid_angles(scan_angle_rank):
        return None

    # This permutation is the only full-size working array. Coordinate and
    # metadata arrays are indexed directly and processed one time bin at a time.
    order = np.lexsort((gps_time, point_source_id))
    if n_points <= np.iinfo(np.uint32).max:
        compact_order = order.astype(np.uint32)
        del order
        order = compact_order

    # Find where each flightline starts/ends in the sorted array
    fl_starts, fl_ends = _flightline_bounds(point_source_id, order)
    for fl_s, fl_e in zip(fl_starts, fl_ends):
        if not _has_valid_angles(scan_angle_rank, order[fl_s:fl_e]):
            return None

    rays = np.empty((n_points, 3), dtype=np.float32)

    time_bin_size = 1.0

    for fl_s, fl_e in zip(fl_starts, fl_ends):
        flight_indices = order[fl_s:fl_e]
        n_fl = fl_e - fl_s

        min_t = gps_time[flight_indices[0]]
        max_t = gps_time[flight_indices[-1]]

        if max_t - min_t < time_bin_size:
            n_bins = 1
        else:
            n_bins = int((max_t - min_t) / time_bin_size) + 1

        bin_edges = np.linspace(min_t, max_t, n_bins + 1)
        bin_splits = np.fromiter(
            (_search_time(gps_time, flight_indices, edge) for edge in bin_edges[1:-1]),
            dtype=np.int64,
            count=max(0, n_bins - 1),
        )
        b_starts = np.concatenate(([0], bin_splits))
        b_ends   = np.concatenate((bin_splits, [n_fl]))
        sensor_heights = np.full(n_bins, np.nan, dtype=np.float64)

        for bin_index, (bs, be) in enumerate(zip(b_starts, b_ends)):
            if bs == be:
                continue

            indices = flight_indices[bs:be]
            bx = x[indices]
            by = y[indices]
            bz = z[indices]
            ba = scan_angle_rank[indices]

            cx = np.mean(bx)
            cy = np.mean(by)
            rays[indices, 0] = cx - bx
            rays[indices, 1] = cy - by

            b_valid = np.abs(ba) > 5.0
            if np.any(b_valid):
                ba_rad = np.radians(np.abs(ba[b_valid]))
                d2d = np.sqrt((bx[b_valid] - cx)**2 + (by[b_valid] - cy)**2)
                h = d2d / np.tan(ba_rad)
                sensor_heights[bin_index] = np.mean(bz[b_valid]) + np.median(h)

        _fill_missing_heights(sensor_heights)
        for bin_index, (bs, be) in enumerate(zip(b_starts, b_ends)):
            if bs == be:
                continue
            indices = flight_indices[bs:be]
            rays[indices, 2] = sensor_heights[bin_index] - z[indices]

    return rays
