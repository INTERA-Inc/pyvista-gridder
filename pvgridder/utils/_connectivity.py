from __future__ import annotations

from typing import Literal, Optional

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
    from .. import cast_to_polydata

    mesh = cast_to_polydata(mesh)
    neighbors = [[] for _ in range(mesh.n_cells)]

    for i1, i2 in mesh["vtkOriginalCellIds"]:
        if i1 == -1 or i2 == -1:
            continue

        neighbors[i1].append(i2)
        neighbors[i2].append(i1)

    return neighbors


def get_connectivity(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    """
    Get mesh connectivity.

    Parameters
    ----------
    mesh : :class:`pyvista.UnstructuredGrid`
        Input mesh.

    Returns
    -------
    :class:`pyvista.PolyData`
        Mesh connectivity.

    """
    from .. import cast_to_polydata

    centers = mesh.cell_centers().points
    mesh = cast_to_polydata(mesh)
    connectivity = [x for x in mesh["vtkOriginalCellIds"] if x[1] != -1]

    poly = pv.PolyData()

    for i, j in connectivity:
        poly += pv.Line(centers[i], centers[j])

    return poly
