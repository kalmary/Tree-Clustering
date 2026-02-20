import numpy as np
import pyvista as pv
from torch_geometric.data import Data


def visualize_graph(data: Data, edge_labels: np.ndarray = None, point_size: float = 8.0):
    """
    Visualize a PyG Data object as a 3D graph using PyVista.

    Args:
        data:         PyG Data with pos (N, 3) and edge_index (2, E).
        edge_labels:  Optional int array (E,) with values 0 or 1.
                      0 -> red  (inter-tree edge)
                      1 -> green (same-tree edge)
                      None -> all edges white.
        point_size:   Sphere size for nodes.
    """
    pos        = data.pos.numpy()        # (N, 3)
    edge_index = data.edge_index.numpy() # (2, E)

    # --- nodes ---
    node_cloud = pv.PolyData(pos)

    # --- edges: deduplicate since to_undirected doubles them ---
    src = edge_index[0]
    dst = edge_index[1]
    mask = src < dst
    src, dst = src[mask], dst[mask]
    if edge_labels is not None:
        labels_masked = edge_labels[mask]

    n_edges   = len(src)
    cells     = np.empty((n_edges, 3), dtype=np.int_)
    cells[:, 0] = 2
    cells[:, 1] = src
    cells[:, 2] = dst

    lines = pv.PolyData()
    lines.points = pos
    lines.lines  = cells.ravel()

    if edge_labels is not None:
        lines.cell_data["label"] = labels_masked.astype(np.float32)

    # --- stats ---
    n_nodes       = len(pos)
    n_edges_total = n_edges
    extent        = pos.max(axis=0) - pos.min(axis=0)

    if edge_labels is not None:
        n_same  = int((labels_masked == 1).sum())
        n_inter = int((labels_masked == 0).sum())
        label_line = f"Same-tree: {n_same}  Inter-tree: {n_inter}"
    else:
        label_line = ""

    stats = (
        f"Nodes : {n_nodes}\n"
        f"Edges : {n_edges_total}\n"
        f"X span: {extent[0]:.1f} m\n"
        f"Y span: {extent[1]:.1f} m\n"
        f"Z span: {extent[2]:.1f} m\n"
        + label_line
    )

    # --- bounding box ---
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    bbox   = pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))

    # --- plotter ---
    pl = pv.Plotter()
    pl.set_background("black")

    # wireframe bounding box gives instant sense of spatial scale
    pl.add_mesh(bbox, style="wireframe", color="gray", line_width=1.0, opacity=0.4)

    # XYZ axes widget in bottom-left corner
    pl.add_axes(line_width=3, labels_off=False)

    # nodes
    pl.add_mesh(node_cloud, color="white", point_size=point_size,
                render_points_as_spheres=True)

    # edges
    if edge_labels is not None:
        pl.add_mesh(lines, scalars="label", cmap=["red", "green"],
                    clim=[0, 1], line_width=1.5, show_scalar_bar=False)
    else:
        pl.add_mesh(lines, color="white", line_width=1.5)

    # stats text overlay
    pl.add_text(stats, position="upper_left", font_size=11,
                color="white", font="courier")

    pl.show()