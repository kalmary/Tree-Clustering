import pyvista as pv
import numpy as np
from typing import Optional

def plot_cloud(points: np.ndarray, labels: Optional[np.ndarray] = None):
    
    cloud = pv.PolyData(points)
    
    # Konfiguracja renderowania w przeglądarce
    pv.set_jupyter_backend('trame')  # Działa też w zwykłych skryptach .py
    
    # Tworzenie plottera
    plotter = pv.Plotter()
    
    if labels is not None:
        # Add cluster labels as scalar data
        cloud['cluster'] = labels
        plotter.add_mesh(
            cloud, 
            scalars='cluster',
            cmap='tab20',  # Good colormap for discrete clusters
            point_size=5, 
            render_points_as_spheres=True
        )
    else:
        # Original behavior - single color
        plotter.add_mesh(
            cloud, 
            color='green', 
            point_size=5, 
            render_points_as_spheres=True
        )
    
    # Wyświetlenie - to otworzy przeglądarkę
    plotter.show()
    
    plotter.close()
    plotter.deep_clean()
