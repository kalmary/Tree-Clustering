import numpy as np

from src.utils.raycloud_io import read_labels_from_segmented_ply, write_raycloud_ply


def test_read_labels_preserves_black_points_as_noise(tmp_path):
    path = tmp_path / "segmented.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(
        header.encode("ascii") + bytes([0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 128, 0])
    )

    labels = read_labels_from_segmented_ply(str(path))

    assert labels[[0, 2]].tolist() == [-1, -1]
    assert set(labels[[1, 3]]) == {0, 1}


def test_write_raycloud_ply_writes_supplied_or_default_rays(tmp_path):
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    rays = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    point_path = tmp_path.joinpath("points.ply")
    ray_path = tmp_path.joinpath("rays.ply")

    write_raycloud_ply(points, str(point_path))
    write_raycloud_ply(points, str(ray_path), rays)

    point_data = point_path.read_bytes()
    ray_data = ray_path.read_bytes()
    assert b"property float nx\n" in point_data
    assert b"property float nx\n" in ray_data
    assert np.array([0.0, 0.0, 10.0], dtype=np.float32).tobytes() in point_data
    assert rays[0].tobytes() in ray_data
