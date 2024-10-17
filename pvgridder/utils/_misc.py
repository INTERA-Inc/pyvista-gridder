from __future__ import annotations
from typing import Optional
from numpy.typing import ArrayLike

import numpy as np
import pyvista as pv


def merge(
    mesh_a: pv.StructuredGrid | pv.UnstructuredGrid,
    mesh_b: pv.StructuredGrid | pv.UnstructuredGrid,
    axis: Optional[int] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    if isinstance(mesh_a, pv.StructuredGrid) and isinstance(mesh_b, pv.StructuredGrid):
        if axis is None:
            raise ValueError("could not merge structured grids with None axis")

        if axis == 0:
            if not (
                np.allclose(mesh_a.x[-1], mesh_b.x[0])
                and np.allclose(mesh_a.y[-1], mesh_b.y[0])
                and np.allclose(mesh_a.z[-1], mesh_b.z[0])
            ):
                raise ValueError("could not merge structured grids with non-matching east and west surfaces")

            slice_ = (slice(1, None),)

        elif axis == 1:
            if not (
                np.allclose(mesh_a.x[:, -1], mesh_b.x[:, 0])
                and np.allclose(mesh_a.y[:, -1], mesh_b.y[:, 0])
                and np.allclose(mesh_a.z[:, -1], mesh_b.z[:, 0])
            ):
                raise ValueError("could not merge structured grids with non-matching north and south surfaces")

            slice_ = (slice(None), slice(1, None))

        else:
            if not (
                np.allclose(mesh_a.x[..., -1], mesh_b.x[..., 0])
                and np.allclose(mesh_a.y[..., -1], mesh_b.y[..., 0])
                and np.allclose(mesh_a.z[..., -1], mesh_b.z[..., 0])
            ):
                raise ValueError("could not merge structured grids with non-matching top and bottom surfaces")

            slice_ = (slice(None), slice(None), slice(1, None))

        X = np.concatenate((mesh_a.x, mesh_b.x[slice_]), axis=axis)
        Y = np.concatenate((mesh_a.y, mesh_b.y[slice_]), axis=axis)
        Z = np.concatenate((mesh_a.z, mesh_b.z[slice_]), axis=axis)
        mesh = pv.StructuredGrid(X, Y, Z)

        if mesh_a.cell_data:
            shape_a = [max(1, n - 1) for n in mesh_a.dimensions]
            shape_b = [max(1, n - 1) for n in mesh_b.dimensions]
            mesh.cell_data.update(
                {
                    k: np.concatenate(
                        (
                            v.reshape(shape_a, order="F"),
                            mesh_b.cell_data[k].reshape(shape_b, order="F"),
                        ),
                        axis=axis,
                    ).ravel(order="F")
                    for k, v in mesh_a.cell_data.items()
                    if k in mesh_b.cell_data
                }
            )

    else:
        mesh = mesh_a + mesh_b

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
