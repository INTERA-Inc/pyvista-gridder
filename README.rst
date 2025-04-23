.. raw:: html

   <p align="center">
      <img src="https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/logo/logo.png?raw=true" width=25% height=25%>
      <h1 align="center"><b>PyVista Gridder</b></h1>
   </p>

Structured and unstructured mesh generation using PyVista for the Finite-Element (FEM), Finite-Difference (FDM) and Finite-Volume Methods (FVM).

Features
--------

- **Pre-Meshed Geometric Objects**: Easily create basic geometric objects with pre-defined meshes, using structured grids whenever possible.
- **Line/Polyline Extrusion**: Extrude lines or polylines into 2D structured grids.
- **Surface Extrusion**: Extrude surface meshes into volumetric meshes while preserving their original type.
- **1.5D/2.5D Mesh Creation**: Generate meshes by stacking polylines or surfaces, ideal for geological modeling and similar applications.
- **2D Voronoi Mesh Generation**: Create 2D Voronoi meshes from a background mesh, with support for adding constraint points to define custom shapes.
- **Mesh Merging**: Combine multiple PyVista meshes into a single mesh and assign cell groups, leaving conformity checks to the user.
- **Additional Utility Functions**: Includes tools to manipulate structured and unstructured grids.

Installation
------------

The recommended way to install **pvgridder** and all its dependencies is through the Python Package Index:

.. code:: bash

   pip install pyvista-gridder --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install .[full] --user

To test the integrity of the installed package, check out this repository and run:

.. code:: bash

   pytest

Examples
--------

2D structured grid
******************

.. code:: python

   import numpy as np
   import pyvista as pv
   import pvgridder as pvg

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
   mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))

.. figure:: https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/anticline.png?raw=true

   :alt: anticline
   :width: 100%
   :align: center

2D Voronoi mesh
***************

.. code:: python

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
      .add_circle(0.16, resolution=16, center=(-0.32, 0.32, 0.0), group="Eye")
      .add_circle(0.16, resolution=16, center=(0.32, 0.32, 0.0), group="Eye")
      .add_polyline(smile_points, width=0.05, group="Mouth")
      .generate_mesh()
   )
   mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))
   
.. figure:: https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/nightmare_fuel.png?raw=true

   :alt: nightmare-fuel
   :width: 100%
   :align: center

2.5D geological model
*********************

.. code:: python

   import pyvista as pv
   import pvgridder as pv

   terrain = pv.examples.download_crater_topo().extract_subset(
      (500, 900, 400, 800, 0, 0), (10, 10, 1)
   )
   bgmesh = pvg.VoronoiMesh2D(pvg.Polygon(terrain, celltype="triangle"), preference="point").generate_mesh()
   terrain_delaunay = pvg.Polygon(terrain, celltype="triangle")
   terrain = terrain.cast_to_structured_grid().warp_by_scalar("scalar1of1")

   mesh = (
      pvg.MeshStack3D(pvg.VoronoiMesh2D(terrain_delaunay, preference="point").generate_mesh())
      .add(0.0)
      .add(terrain.translate((0.0, 0.0, -1000.0)), 5, group="Bottom layer")
      .add(terrain.translate((0.0, 0.0, -500.0)), 5, group="Middle layer")
      .add(terrain, 5, group="Top Layer")
      .generate_mesh()
   )
   mesh.plot(show_edges=True, scalars=pvg.get_cell_group(mesh))

.. figure:: https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/topographic_terrain.png?raw=true

   :alt: topographic-terrain
   :width: 100%
   :align: center

Acknowledgements
----------------

This project is supported by Nagra (National Cooperative for the Disposal of Radioactive Waste), Switzerland.
