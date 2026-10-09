from xml.dom import minidom

import numpy as np
import pyvista as pv
from shapely import Polygon
from svg.path import parse_path

import pvgridder as pvg


# Parameters
shift = 14.37

# Extract and interpolate coordinates of the first snake
with minidom.parse("python_logo.svg") as doc:
    paths = [path.getAttribute("d") for path in doc.getElementsByTagName("path")]

snake_coordinates = []
t_values = np.linspace(0.0, 1.0, 8)

for segment in parse_path(paths[0]):
    if hasattr(segment, "start") and hasattr(segment, "end"):
        for t in t_values:
            point = segment.point(t)
            snake_coordinates.append((point.real, point.imag))

# Extract snake
snake = snake_coordinates[:-48]
snake = pv.MultipleLines(np.insert(snake, 2, 0.0, axis=-1))
snake = pvg.decimate_rdp(snake)

# Extract eye
eye = snake_coordinates[-48:]
eye = pv.MultipleLines(np.insert(eye, 2, 0.0, axis=-1))
eye = pvg.decimate_rdp(eye)

# Generate Voronoi tesselation from Delaunay triangulation
snake1 = pvg.VoronoiMesh2D(
    pvg.Polygon(
        snake,
        [eye],
        celltype="triangle",
        cellsize=5.0,
        algorithm=8,
    ),
    preference="point",
).generate_mesh()
snake1 = snake1.translate(list(map(lambda x: -x, snake1.center)))
snake1 = snake1.translate((-shift, -shift, 0.0)).rotate_z(180.0)

# Generate QuadTree mesh
x = np.linspace(snake.bounds.x_min, snake.bounds.x_max, 11)
y = np.linspace(snake.bounds.y_min, snake.bounds.y_max, 11)
bmesh = pv.RectilinearGrid(x, y, [0.0])

snake2 = (
    pvg.QuadTree(bmesh, max_depth=4)
    .add_boundary_polygon(Polygon(snake.points[:, :2], [eye.points[:, :2]]), depth=4)
    .generate_mesh(balance=True)
)
snake2 = snake2.translate(list(map(lambda x: -x, snake2.center)))
snake2 = snake2.translate((-shift, -shift, 0.0))

# Plot
p = pv.Plotter(
    window_size=[800, 800],
    off_screen=True,
    image_scale=2,
)
p.add_mesh(snake1, color="#306998", line_width=3, show_edges=True)
p.add_mesh(snake2, color="#FFD43B", line_width=3, show_edges=True)
# p.view_xy(negative=True)
p.camera_position = [
    (0.0, 0.0, -210.0),
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
]
p.screenshot("logo.png", transparent_background=True, return_img=False)
