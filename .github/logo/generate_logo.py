import numpy as np
import pyvista as pv
import pvgridder as pvg
from svg.path import parse_path
from xml.dom import minidom


# Extract and interpolate coordinates of the first snake
with minidom.parse("python_logo.svg") as doc:
    path_strings = [path.getAttribute("d") for path in doc.getElementsByTagName("path")]

snake_coordinates = []
t_values = np.linspace(0.0, 1.0, 8)

for segment in parse_path(path_strings[0]):
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
    pvg.Polygon(snake, [eye], celltype="triangle", cellsize=5.0), preference="point"
).generate_mesh()
snake1 = snake1.translate(list(map(lambda x: -x, snake1.center)))

# Shift and rotate the second snake
shift = 14.37
snake1 = snake1.translate((-shift, -shift, 0.0))
snake2 = snake1.rotate_z(180.0)

# Plot
p = pv.Plotter(
    window_size=(800, 800),
    off_screen=True,
    image_scale=2,
)
p.add_mesh(snake1, color="#FFD43B", line_width=3, show_edges=True)
p.add_mesh(snake2, color="#306998", line_width=3, show_edges=True)
# p.view_xy(negative=True)
p.camera_position = [
    (0.0, 0.0, -210.0),
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
]
p.screenshot("logo.png", transparent_background=True, return_img=False)
