import numpy as np
import pyvista as pv

import pvgridder as pvg


# Example 1: anticline
mesh = (
    pvg.MeshStack2D(pv.Line([-3.14, 0.0, 0.0], [3.14, 0.0, 0.0], resolution=41))
    .add(0.0)
    .add(lambda x, y, z: np.cos(x) + 1.0, 4, group="Layer 1")
    .add(0.5, 2, group="Layer 2")
    .add(0.5, 2, group="Layer 3")
    .add(0.5, 2, group="Layer 4")
    .add(lambda x, y, z: np.full_like(x, 3.4), 4, group="Layer 5")
    .generate_mesh()
)

p = pv.Plotter(off_screen=True)
p.theme.font.color = "#808080"
p.add_mesh(
    mesh,
    scalars=pvg.get_cell_group(mesh),
    show_edges=True,
)
p.camera_position = [
    (0.0, -11.40162342415554, 1.7000000476837158),
    (0.0, 0.0, 1.7000000476837158),
    (0.0, 0.0, 1.0),
]
p.add_axes()  # type: ignore
p.screenshot("anticline.png", transparent_background=True, return_img=False)

# Example 2: nightmare fuel
smile_radius = 0.64
smile_points = [
    (smile_radius * np.cos(theta), smile_radius * np.sin(theta), 0.0)
    for theta in np.deg2rad(np.linspace(200.0, 340.0, 32))
]
mesh = (
    pvg.VoronoiMesh2D(pvg.Annulus(0.0, 1.0, 16, 32), default_group="Face")
    .add_circle(
        0.16, plain=False, resolution=16, center=(-0.32, 0.32, 0.0), group="Eye"
    )
    .add_circle(0.16, plain=True, resolution=16, center=(0.32, 0.32, 0.0), group="Eye")
    .add_polyline(smile_points, width=0.05, group="Mouth")
    .generate_mesh()
)

p = pv.Plotter(off_screen=True)
p.theme.font.color = "#808080"
p.add_mesh(
    mesh,
    scalars=pvg.get_cell_group(mesh),
    show_edges=True,
)
p.add_axes()  # type: ignore
p.view_xy()  # type: ignore
p.screenshot("nightmare_fuel.png", transparent_background=True, return_img=False)

# Example 3: conformal QuadTree
bmesh = pvg.Rectangle(5000.0, 10000.0, 20, 40)
qtree = pvg.QuadTree(bmesh, max_depth=2)
qtree = qtree.add_boundary_polygon(pvg.examples.freyberg.load_outcrop())
qtree = qtree.add_polyline(pvg.examples.freyberg.load_river(), group="River")

for well in pvg.examples.freyberg.load_wells():
    qtree = qtree.add_point(well, group="Well")

mesh = qtree.generate_mesh(balance=True, conformal=False)
mesh_conformal = qtree.generate_mesh(balance=True, conformal=True)
connectivity = pvg.get_connectivity(mesh)
connectivity_conformal = pvg.get_connectivity(mesh_conformal)

p = pv.Plotter(
    window_size=[1200, 800],
    shape=(1, 3),
    off_screen=True,
    border=False,
)
p.theme.font.color = "#808080"
p.link_views()

p.subplot(0, 0)
p.add_mesh(mesh, style="wireframe", line_width=1.6, color="#7D8590")
p.add_title("Mesh", font_size=12)

p.subplot(0, 1)
p.add_mesh(mesh, style="wireframe", line_width=1.6, color="#7D8590")
p.add_mesh(connectivity, color="#46D3CF", line_width=1.2, opacity=0.8)
p.add_title("Nonconformal", font_size=12)

p.subplot(0, 2)
p.add_mesh(mesh_conformal, style="wireframe", line_width=1.6, color="#7D8590")
p.add_mesh(connectivity_conformal, color="#46D3CF", line_width=1.2, opacity=0.8)
p.add_title("Conformal", font_size=12)

p.view_xy()  # type: ignore
p.camera.parallel_projection = True
p.camera.parallel_scale = 6400.0
p.screenshot("quadtree.png", transparent_background=True, return_img=False)
p.close()

# Example 4: topographic terrain
terrain = pv.examples.download_crater_topo().extract_subset(
    (500, 900, 400, 800, 0, 0), (10, 10, 1)
)
terrain = pv.wrap(terrain)
terrain_delaunay = pvg.Polygon(terrain, celltype="triangle")
terrain = terrain.cast_to_structured_grid().warp_by_scalar("scalar1of1")

mesh = (
    pvg.MeshStack3D(
        pvg.VoronoiMesh2D(terrain_delaunay, preference="point").generate_mesh()
    )
    .add(0.0)
    .add(
        terrain.translate((0.0, 0.0, -1000.0)), 5, method="log_r", group="Bottom layer"
    )
    .add(500.0, 5, group="Middle layer")
    .add(terrain, 5, method="log", group="Top Layer")
    .generate_mesh()
)

p = pv.Plotter(off_screen=True)
p.theme.font.color = "#808080"
p.add_mesh(
    mesh,
    scalars=pvg.get_cell_group(mesh),
    show_edges=True,
)
p.camera_position = [
    (1829341.9573620637, 5657798.379338102, 8412.358806162943),
    (1821117.4671851501, 5649573.889161188, 187.86862924924867),
    (0.0, 0.0, 1.0),
]
p.add_axes()  # type: ignore
p.screenshot("topographic_terrain.png", transparent_background=True, return_img=False)
