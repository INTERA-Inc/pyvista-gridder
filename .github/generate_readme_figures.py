#%%
import numpy as np
import pyvista as pv
import pvgridder as pvg


# Example 1: anticline
mesh = (
    pvg.MeshStack2D(pv.Line([-3.14, 0.0, 0.0], [3.14, 0.0, 0.0], 41))
    .add(0.0)
    .add(lambda x, y, z: np.cos(x) + 1.0, 4, group="Layer 1")
    .add(lambda x, y, z: np.cos(x) + 1.5, 2, group="Layer 2")
    .add(lambda x, y, z: np.cos(x) + 2.0, 2, group="Layer 3")
    .add(lambda x, y, z: np.cos(x) + 2.5, 2, group="Layer 4")
    .add(lambda x, y, z: np.full_like(x, 3.4), 4, group="Layer 5")
    .generate_mesh()
)

p = pv.Plotter(off_screen=True)
p.add_mesh(mesh, show_edges=True)
p.camera_position = [
    (0.0, -11.40162342415554, 1.7000000476837158),
    (0.0, 0.0, 1.7000000476837158),
    (0.0, 0.0, 1.0),
]
p.add_axes()
p.screenshot("anticline.png", transparent_background=True, return_img=False)

# Example 2: nightmare fuel
smile_radius = 0.64
smile_points = [
    (smile_radius * np.cos(theta), smile_radius * np.sin(theta), 0.0)
    for theta in np.deg2rad(np.linspace(200.0, 340.0, 32))
]
mesh = (
    pvg.VoronoiMesh2D(pvg.Annulus(0.0, 1.0, 16, 32), default_group="Face")
    .add_circle(0.16, resolution=16, center=(-0.32, 0.32, 0.0), group="Eye")
    .add_circle(0.16, resolution=16, center=(0.32, 0.32, 0.0), group="Eye")
    .add_polyline(smile_points, width=0.05, group="Mouth")
    .generate_mesh()
)

group_map = {v: k for k, v in mesh.user_dict["CellGroup"].items()}
p = pv.Plotter(off_screen=True)
p.add_mesh(
    mesh,
    scalars=[group_map[i] for i in mesh.cell_data["CellGroup"]],
    show_edges=True,
)
p.add_axes()
p.view_xy()
p.screenshot("nightmare_fuel.png", transparent_background=True, return_img=False)
