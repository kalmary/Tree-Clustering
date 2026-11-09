import numpy as np

from src.utils.get_rays import get_las_ray_inputs, get_rays


def test_get_las_ray_inputs_uses_legacy_angles_without_scaling():
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(2, header=header)
    las.gps_time = np.array([1.0, 2.0])
    las.point_source_id = np.array([4, 4], dtype=np.uint16)
    las.scan_angle_rank = np.array([-10, 12], dtype=np.int8)

    inputs = get_las_ray_inputs(las, use_rays=True)

    np.testing.assert_array_equal(inputs["scan_angle_deg"], [-10.0, 12.0])


def test_get_las_ray_inputs_scales_modern_angles_to_degrees():
    import laspy

    header = laspy.LasHeader(point_format=6, version="1.4")
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(2, header=header)
    las.gps_time = np.array([1.0, 2.0])
    las.point_source_id = np.array([4, 4], dtype=np.uint16)
    las.scan_angle = np.array([-1000, 2000], dtype=np.int16)

    inputs = get_las_ray_inputs(las, use_rays=True)

    np.testing.assert_allclose(inputs["scan_angle_deg"], [-6.0, 12.0])


def test_get_las_ray_inputs_does_not_read_las_when_rays_are_disabled():
    class LasWithoutReadableMetadata:
        @property
        def point_format(self):
            raise AssertionError("point format must not be read")

    assert get_las_ray_inputs(LasWithoutReadableMetadata(), use_rays=False) == {}


def test_get_las_ray_inputs_returns_empty_when_required_dimensions_are_missing():
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(2, header=header)

    assert get_las_ray_inputs(las, use_rays=True) == {}


def test_get_rays_rejects_mismatched_input_lengths():
    try:
        get_rays(
            x=np.zeros(2),
            y=np.zeros(1),
            z=np.zeros(2),
            gps_time=np.zeros(2),
            point_source_id=np.zeros(2),
            scan_angle_deg=np.full(2, 10.0),
        )
    except ValueError as exc:
        assert "y must contain one value per point" in str(exc)
    else:
        raise AssertionError("Expected mismatched input lengths to be rejected")
