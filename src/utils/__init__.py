from .get_rays import get_las_ray_inputs as get_las_ray_inputs
from .get_rays import get_rays as get_rays


def __getattr__(name):
    if name == "plot_cloud":
        from .plot_cloud import plot_cloud

        return plot_cloud
    raise AttributeError(name)
