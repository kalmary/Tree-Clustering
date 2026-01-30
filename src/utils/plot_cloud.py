
def plot_cloud(points: np.ndarray):
    
        cloud = pv.PolyData(points)

        # Konfiguracja renderowania w przeglądarce
        pv.set_jupyter_backend('trame') # Działa też w zwykłych skryptach .py

        # Tworzenie plottera
        plotter = pv.Plotter()
        plotter.add_mesh(cloud, color='green', point_size=5, render_points_as_spheres=True)

        # Wyświetlenie - to otworzy przeglądarkę
        plotter.show()

        plotter.close()
        plotter.deep_clean()
