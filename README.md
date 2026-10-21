<h1 align="center">
   <img src=".github/logo/logo.png" width=160px><br>
   PyVista Gridder
</h1>

[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](https://github.com/INTERA-Inc/pyvista-gridder/blob/master/LICENSE)
[![Stars](https://img.shields.io/github/stars/INTERA-Inc/pyvista-gridder?style=flat&logo=github)](https://github.com/INTERA-Inc/pyvista-gridder)
[![Pyversions](https://img.shields.io/pypi/pyversions/pyvista-gridder.svg?style=flat)](https://pypi.org/pypi/pyvista-gridder/)
[![Version](https://img.shields.io/pypi/v/pyvista-gridder.svg?style=flat)](https://pypi.org/project/pyvista-gridder)
[![Downloads](https://pepy.tech/badge/pyvista-gridder)](https://pepy.tech/project/pyvista-gridder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black)

Structured and unstructured mesh generation using PyVista for the Finite-Element (FEM), Finite-Difference (FDM), and Finite-Volume Methods (FVM).

## Features

- **Pre-Meshed Geometric Primitives**: Create basic geometric objects with pre-configured meshes, using structured grids whenever possible.
- **Line/Polyline Extrusion**: Extrude lines or polylines into 2D structured quadrilateral grids.
- **Surface Extrusion**: Extrude surface meshes into 3D volumetric meshes while preserving cell types.
- **1.5D/2.5D Mesh Creation**: Stack polylines or surfaces to build layered meshes, ideal for geological modeling and similar applications.
- **2D QuadTree Mesh Generation**: Generate adaptive QuadTree meshes with fast 2:1 balancing, point/polyline refinement, and optional conformal/orthogonal polygonal export.
- **2D Delaunay Mesh Generation**: Create constrained triangular or quad-recombined meshes via Gmsh, supporting embedded features, physical group tagging, and an API consistent with the QuadTree interface.
- **2D Voronoi Mesh Generation**: Create 2D Voronoi meshes from a background mesh, with support for adding constraint points to define custom shapes.
- **Mesh Merging**: Combine multiple PyVista meshes into a single dataset and assign cell groups, leaving conformity checks to the user.
- **Additional Grid Utility Functions**: Includes tools to manipulate structured and unstructured grids.

## Installation

The recommended way to install **pyvista-gridder** and all its dependencies is through the Python Package Index:

```bash
pip install pyvista-gridder
```

Otherwise, clone and extract the package, then run from the package location:

```bash
pip install .
```

To test the integrity of the installed package, check out this repository and run:

```bash
pytest
```

## Examples

### 2D Structured Grid

```python
import numpy as np
import pyvista as pv
import pvgridder as pvg

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
mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))
```

<p align="center">
  <img src="https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/anticline.png?raw=true" alt="anticline" width="80%">
</p>

### 2D Voronoi Mesh

```python
import numpy as np
import pyvista as pv
import pvgridder as pvg

smile_radius = 0.64
smile_points = [
    (smile_radius * np.cos(theta), smile_radius * np.sin(theta), 0.0)
    for theta in np.deg2rad(np.linspace(200.0, 340.0, 32))
]
mesh = (
    pvg.VoronoiMesh2D(pvg.Annulus(0.0, 1.0, 16, 32), default_group="Face")
    .add_circle(0.16, plain=False, resolution=16, center=(-0.32, 0.32, 0.0), group="Eye")
    .add_circle(0.16, plain=True, resolution=16, center=(0.32, 0.32, 0.0), group="Eye")
    .add_polyline(smile_points, width=0.05, group="Mouth")
    .generate_mesh()
)
mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))
```

<p align="center">
  <img src="https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/nightmare_fuel.png?raw=true" alt="nightmare-fuel" width="80%">
</p>

### Freyberg Model

```python
import pvgridder as pvg

outcrop = pvg.examples.freyberg.load_outcrop()  # shapely.Polygon
river = pvg.examples.freyberg.load_river()  # shapely.LineString
wells = pvg.examples.freyberg.load_wells()  # 2D array
tri = (
    pvg.DelaunayMesh2D(outcrop, cellsize=1000.0)
    .add_polyline(river, cellsize=150.0, growth=1.2, group="River")
    .add_points(wells, cellsize=150.0, constraint=6, group="Well")
    .generate_mesh()
)
quad = (
    pvg.QuadTree(pvg.Rectangle(5000.0, 10000.0, 20, 40), max_depth=2)
    .add_boundary_polygon(outcrop)
    .add_polyline(river, group="River")
    .add_points(wells, group="Well")
    .generate_mesh(balance=True, conformal=True)
)
tri_voro = pvg.VoronoiMesh2D(tri, preference="point").generate_mesh()
quad_voro = pvg.VoronoiMesh2D(quad, preference="cell").generate_mesh()
```

<p align="center">
  <img src="https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/freyberg.png?raw=true" alt="freyberg" width="60%">
</p>

### 2.5D Geological Model

```python
import pyvista as pv
import pvgridder as pvg

terrain = pv.examples.download_crater_topo().extract_subset(
    (500, 900, 400, 800, 0, 0), (10, 10, 1)
)
terrain_delaunay = pvg.Polygon(terrain, celltype="triangle")
terrain = terrain.cast_to_structured_grid().warp_by_scalar("scalar1of1")

mesh = (
    pvg.MeshStack3D(
        pvg.VoronoiMesh2D(terrain_delaunay, preference="point").generate_mesh()
    )
    .add(0.0)
    .add(terrain.translate((0.0, 0.0, -1000.0)), 5, method="log_r", group="Bottom layer")
    .add(500.0, 5, group="Middle layer")
    .add(terrain, 5, method="log", group="Top Layer")
    .generate_mesh()
)
mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))
```

<p align="center">
  <img src="https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/topographic_terrain.png?raw=true" alt="topographic-terrain" width="80%">
</p>

## Acknowledgements

This project is supported by Nagra (National Cooperative for the Disposal of Radioactive Waste), Switzerland.