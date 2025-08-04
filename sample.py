#%%
import numpy as np
import pvgridder as pvg
import pyvista as pv

from scipy.spatial import KDTree

%reload_ext autoreload
%autoreload 2

# mesh = pvg.examples.load_well_2d()
# polyline = pv.Line([-15.0, 0.0, 0.0], [15.0, 0.0, 0.0], resolution=100)


mesh = pvg.examples.load_well_3d(voronoi=False)
# # polyline = pv.Line([0.0, 0.0, 15.0], [0.0, 0.0, -50.0], resolution=100)
# polyline = pv.Line([0.1, 0.1, 10.0], [0.1, 0.1, -50.0], resolution=10)
# # polyline = pv.Line([-5.2, -5.3, 10.0], [5.412, 5.32, -50.0], resolution=100)
polyline = pv.Line([-14.0, -9.0, 16.0], [0.0, 0.0, -36.0], resolution=42)
# polyline = pv.Line([0.0, 0.0, 16.0], [0.0, 0.0, -32.0], resolution=31)
# polyline = pv.Line([0.0, 0.0, 16.0], [0.0, 0.0, -30.0], resolution=100)
# # polyline = pv.Line([-5.0, -5.0, 10.0], [-7.0, -7.0, -50.0], resolution=1)

# x = np.linspace(-10.0, 10.0, 20)
# mesh = pv.RectilinearGrid(x, x, x)

intersection_polyline = pvg.intersect_polyline(
    mesh,
    polyline,
    ignore_points_before_entry=True,
    ignore_points_after_exit=True,
)
cell_ids = intersection_polyline.cell_data["IntersectedCellIds"]
cell_ids = cell_ids[cell_ids > -1]

#%%
count = 0

for line, cell_id in zip(
    pvg.split_lines(intersection_polyline, as_lines=True),
    intersection_polyline.cell_data["IntersectedCellIds"],
):
    count += 1
    if cell_id == -1:
        continue

    cell = mesh.extract_cells(cell_id)
    # assert cell.find_containing_cell(line.cell_centers().points[0]) == 0

    try:
        assert cell.find_containing_cell(line.cell_centers().points[0]) == 0
    except AssertionError:
        p = pv.Plotter(notebook=False)
        p.add_mesh(cell, color="blue", show_edges=True, opacity=0.5)
        p.add_mesh(line, color="red", line_width=3)
        p.add_axes()
        p.show()

#%%
p = pv.Plotter(notebook=False)
p.add_mesh(pvg.extract_cell_geometry(mesh), opacity=0.1, show_edges=True, style="wireframe", color="black")
p.add_mesh(mesh.extract_cells(intersection_polyline.cell_data["IntersectedCellIds"][intersection_polyline.cell_data["IntersectedCellIds"] > -1]), opacity=0.1, show_edges=True, color="orange")
p.add_mesh(intersection_polyline, color="red", line_width=3)
p.add_mesh(pv.PolyData(intersection_polyline.points), color="green", point_size=10, render_points_as_spheres=True)
p.add_axes()
# p.view_xz()
p.view_xy()
# p.view_yz()
p.show_bounds()
p.enable_parallel_projection()
p.show()

# %%
