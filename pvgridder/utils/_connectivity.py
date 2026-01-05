from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv


if TYPE_CHECKING:
    from typing import Optional  # pragma: no cover

    from numpy.typing import ArrayLike, NDArray  # pragma: no cover


def get_neighborhood(
    mesh: pv.DataSet,
    remove_ghost_cells: bool = True,
) -> tuple[NDArray, ...]:
    """
    Get mesh neighborhood.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    remove_ghost_cells : bool, optional
        If True, remove ghost cells.

    Returns
    -------
    tuple[NDArray, ...]
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

    return tuple([np.asanyarray(neighbor) for neighbor in neighbors])


def get_connectivity(
    mesh: pv.DataSet,
    cell_centers: Optional[ArrayLike] = None,
    remove_ghost_cells: bool = True,
) -> pv.PolyData:
    """
    Get mesh connectivity.

    Parameters
    ----------
    mesh : pyvista.DataSet
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
    from .. import extract_cell_geometry, get_cell_centers

    cell_centers = (
        get_cell_centers(mesh) if cell_centers is None else np.asanyarray(cell_centers)
    )

    if np.shape(cell_centers) != (mesh.n_cells, 3):
        raise ValueError(
            f"invalid cell centers (expected 2D array of shape ({mesh.n_cells}, 3)"
        )

    mesh = extract_cell_geometry(mesh, remove_ghost_cells)
    lines = [(i1, i2) for i1, i2 in mesh["vtkOriginalCellIds"] if i1 != -1 and i2 != -1]
    lines = np.column_stack((np.full(len(lines), 2), lines)).ravel()

    return pv.PolyData(cell_centers, lines=lines)
