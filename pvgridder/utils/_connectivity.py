from __future__ import annotations

from typing import Optional

import numpy as np
import pyvista as pv


def get_neighborhood(mesh: pv.UnstructuredGrid) -> list[ArrayLike]:
    """
    Get mesh neighborhood.

    Parameters
    ----------
    mesh : :class:`pyvista.UnstructuredGrid`
        Input mesh.

    Returns
    -------
    sequence of ArrayLike
        List of neighbor cell IDs for all cells.

    """
    from .. import extract_cell_geometry

    mesh = extract_cell_geometry(mesh)
    neighbors = [[] for _ in range(mesh.n_cells)]

    for i1, i2 in mesh["vtkOriginalCellIds"]:
        if i1 == -1 or i2 == -1:
            continue

        neighbors[i1].append(i2)
        neighbors[i2].append(i1)

    return neighbors


def get_connectivity(
    mesh: pv.UnstructuredGrid,
    cell_centers: Optional[ArrayLike] = None,
) -> pv.PolyData:
    """
    Get mesh connectivity.

    Parameters
    ----------
    mesh : :class:`pyvista.UnstructuredGrid`
        Input mesh.
    cell_centers : ArrayLike, optional
        Cell centers used for connectivity lines.

    Returns
    -------
    :class:`pyvista.PolyData`
        Mesh connectivity.

    """
    from .. import extract_cell_geometry

    cell_centers = cell_centers if cell_centers is not None else mesh.cell_centers().points

    mesh = extract_cell_geometry(mesh)
    lines = [(i1, i2) for i1, i2 in mesh["vtkOriginalCellIds"] if i1 != -1 and i2 != -1]
    lines = np.column_stack((np.full(len(lines), 2), lines)).ravel()

    return pv.PolyData(cell_centers, lines=lines)
