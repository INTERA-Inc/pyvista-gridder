from __future__ import annotations
from numpy.typing import ArrayLike

import numpy as np
import pyvista as pv

from ..core._helpers import generate_volume_from_two_surfaces


def extrude(
    mesh: pv.StructuredGrid | pv.UnstructuredGrid,
    vector: ArrayLike,
    nsub: Optional[int | list[float]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    vector = np.asarray(vector)

    if vector.shape != (3,):
        raise ValueError("invalid extrusion vector")

    mesh_a = mesh
    mesh_b = mesh.copy()
    mesh_b.points += vector
    mesh = generate_volume_from_two_surfaces(mesh_a, mesh_b, nsub)

    return mesh


def quadraticize(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    n_points = mesh.n_points

    cells = []
    celltypes = []
    quad_points = []

    for cell in mesh.cell:
        if cell.type.name not in {"TRIANGLE", "QUAD"}:
            raise NotImplementedError()

        celltype = f"QUADRATIC_{cell.type.name}"
        new_points = 0.5 * (cell.points + np.roll(cell.points, -1, axis=0))
        n_new_points = len(new_points)
        new_points_ids = np.arange(n_new_points) + n_points

        celltypes.append(int(pv.CellType[celltype]))
        cell_ = cell.point_ids + new_points_ids.tolist()
        cells += [len(cell_), *cell_]
        quad_points.append(new_points)
        n_points += n_new_points

    quad_points = np.concatenate(quad_points)
    points = np.row_stack((mesh.points, quad_points))

    return pv.UnstructuredGrid(cells, celltypes, points)
