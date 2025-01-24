from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pyvista as pv


def get_neighborhood(
    mesh: pv.UnstructuredGrid,
    remove_ghost_cells: bool = True,
) -> Sequence[ArrayLike]:
    """
    Get mesh neighborhood.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        Input mesh.
    remove_ghost_cells : bool, optional
        If True, remove ghost cells.

    Returns
    -------
    Sequence[ArrayLike]
        List of neighbor cell IDs for all cells.

    """
    from .. import extract_cell_geometry

    neighbors = [[] for _ in range(mesh.n_cells)]
    mesh = extract_cell_geometry(mesh, remove_ghost_cells)

    for i1, i2 in mesh["vtkOriginalCellIds"]:
        if i1 == -1 or i2 == -1:
            continue

        neighbors[i1].append(i2)
        neighbors[i2].append(i1)

    return neighbors


def get_connectivity(
    mesh: pv.UnstructuredGrid,
    cell_centers: Optional[ArrayLike] = None,
    remove_ghost_cells: bool = True,
) -> pv.PolyData:
    """
    Get mesh connectivity.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        Input mesh.
    cell_centers : ArrayLike, optional
        Cell centers used for connectivity lines.
    remove_ghost_cells : bool, optional
        If True, remove ghost cells.

    Returns
    -------
    pyvista.PolyData
        Mesh connectivity.

    """
    from .. import extract_cell_geometry

    if cell_centers is None:
        # Remove ghost cells before calculating cell centers
        # See <https://github.com/pyvista/pyvista/issues/7113>
        mesh_copy = mesh.copy(deep=False)
        mesh_copy.clear_data()
        cell_centers = mesh_copy.cell_centers(vertex=False).points

    if np.shape(cell_centers) != (mesh.n_cells, 3):
        raise ValueError(f"invalid cell centers (expected 2D array of shape ({mesh.n_cells}, 3)")

    mesh = extract_cell_geometry(mesh, remove_ghost_cells)
    lines = [(i1, i2) for i1, i2 in mesh["vtkOriginalCellIds"] if i1 != -1 and i2 != -1]
    lines = np.column_stack((np.full(len(lines), 2), lines)).ravel()

    return pv.PolyData(cell_centers, lines=lines)
