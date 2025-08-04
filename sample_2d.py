#%%
import numpy as np
import pvgridder as pvg
import pyvista as pv

%reload_ext autoreload
%autoreload 2

mesh = pv.RectilinearGrid([0, 1, 2], [0, 1, 2], [0, 1, 2])
mesh.cell_data["vtkGhostType"] = np.zeros(mesh.n_cells, dtype=np.uint8)
mesh.cell_data["vtkGhostType"][-1] = 32
centers = pvg.get_cell_centers(mesh)

# mesh = pvg.examples.load_anticline_2d().cast_to_unstructured_grid()
# mesh_empty = pv.UnstructuredGrid([0], [pv.CellType.EMPTY_CELL], [])
# mesh_empty.cell_data["vtkGhostType"] = np.array([0], dtype=np.uint8)
# mesh2 = mesh + mesh_empty
# centers = pvg.get_cell_centers(mesh2)
# %%
