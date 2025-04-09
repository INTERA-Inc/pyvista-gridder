PyVista Gridder
===============

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

   pip install pvgridder --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install .[full] --user

To test the integrity of the installed package, check out this repository and run:

.. code:: bash

   pytest

Examples
--------

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
   mesh.plot(show_edges=True)

.. figure:: https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/anticline.png?raw=true

   :alt: anticline
   :width: 100%
   :align: center

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

   group_map = {v: k for k, v in mesh.user_dict["CellGroup"].items()}
   
.. figure:: https://github.com/INTERA-Inc/pyvista-gridder/blob/main/.github/nightmare_fuel.png?raw=true

   :alt: nightmare-fuel
   :width: 100%
   :align: center
