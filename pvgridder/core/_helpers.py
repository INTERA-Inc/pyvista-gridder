from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

import numpy as np
import pyvista as pv


def generate_arc(
    radius: float,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    nsub: Optional[int | list[float]] = None,
) -> pv.PolyData:
    perc = nsub_to_perc(nsub)
    angles = theta_min + perc * (theta_max - theta_min)
    angles = np.deg2rad(angles)
    points = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))

    return pv.MultipleLines(points)


def generate_line_from_two_points(
    point_a: ArrayLike,
    point_b: ArrayLike,
    nsub: Optional[int | list[float]] = None,
) -> pv.PolyData:
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    if point_a.shape != point_b.shape:
        raise ValueError("could not generate a line from two inhomogeneous points")

    perc = nsub_to_perc(nsub)[:, np.newaxis]
    points = point_a + perc * (point_b - point_a)
    points = points if points.shape[1] == 3 else np.column_stack((points, np.zeros(len(points))))

    return pv.MultipleLines(points)


def generate_plane_surface_from_two_lines(
    line_a: pv.PolyData | ArrayLike,
    line_b: pv.PolyData | ArrayLike,
    nsub: Optional[int | list[float]] = None,
) -> pv.StructuredGrid:
    line_a = line_a.points if isinstance(line_a, pv.PolyData) else np.asarray(line_a)
    line_b = line_b.points if isinstance(line_b, pv.PolyData) else np.asarray(line_b)
    
    if line_a.shape != line_b.shape:
        raise ValueError("could not generate plane surface from two inhomogeneous lines")

    perc = nsub_to_perc(nsub)[:, np.newaxis, np.newaxis]
    X, Y, Z = (line_a + perc * (line_b - line_a)).transpose((2, 1, 0))

    return pv.StructuredGrid(X, Y, Z)


def generate_volume_from_two_surfaces(
    surface_a: pv.StructuredGrid | pv.UnstructuredGrid,
    surface_b: pv.StructuredGrid | pv.UnstructuredGrid,
    nsub: Optional[int | list[float]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    if surface_a.points.shape != surface_b.points.shape or not isinstance(surface_a, type(surface_b)):
        raise ValueError("could not generate volume from two inhomogeneous surfaces")

    if isinstance(surface_a, pv.StructuredGrid):
        if surface_a.dimensions != surface_b.dimensions:
            raise ValueError("could not generate volume from two inhomogeneous structured surfaces")

        if sum(n == 1 for n in surface_a.dimensions) != 1:
            raise ValueError("could not generate volume from non 2D structured grid")

        idx = surface_a.dimensions.index(1)

        if idx == 0:
            slice_ = (0,)
            axis = (0, 1, 2)

        elif idx == 1:
            slice_ = (slice(None), 0)
            axis = (2, 0, 1)

        else:
            slice_ = (slice(None), slice(None), 0)
            axis = (1, 2, 0)

        xa, ya, za = surface_a.x[slice_], surface_a.y[slice_], surface_a.z[slice_]
        xb, yb, zb = surface_b.x[slice_], surface_b.y[slice_], surface_b.z[slice_]

        perc = nsub_to_perc(nsub)[:, np.newaxis, np.newaxis]
        X = (xa + perc * (xb - xa)).transpose(axis)
        Y = (ya + perc * (yb - ya)).transpose(axis)
        Z = (za + perc * (zb - za)).transpose(axis)
        mesh = pv.StructuredGrid(X, Y, Z)

    elif isinstance(surface_a, pv.UnstructuredGrid):
        celltypes = _celltype_map[surface_a.celltypes]

        if (celltypes == -1).any():
            raise ValueError("could not generate volume from surfaces with unsupported cell types")

        if not np.allclose(surface_a.celltypes, surface_b.celltypes):
            raise ValueError("could not generate volume from two inhomogeneous unstructured surfaces")

        points_a = surface_a.points
        points_b = surface_b.points

        perc = nsub_to_perc(nsub)[:, np.newaxis, np.newaxis]
        points = points_a + perc * (points_b - points_a)

        n = perc.size - 1
        n_points = surface_a.n_points
        offset = surface_a.offset
        cell_connectivity = surface_a.cell_connectivity
        cells = [[] for _ in range(n)]

        for i1, i2, celltype in zip(offset[:-1], offset[1:], celltypes):
            cell = cell_connectivity[i1 : i2]

            if celltype == 42:  # POLYHEDRON
                raise NotImplementedError()

            else:
                cell = np.concatenate((cell, cell + n_points))

            for i, cells_ in enumerate(cells):
                cells_ += [cell.size, *(cell + (i * n_points)).tolist()]

        cells = np.concatenate(cells)
        celltypes = np.tile(celltypes, n)
        points = points.reshape((n_points * (n + 1), 3))
        mesh = pv.UnstructuredGrid(cells, celltypes, points)

    else:
        raise ValueError(f"could not generate volume from {type(surface_a)}")

    for k, v in surface_a.point_data.items():
        mesh.point_data[k] = np.tile(v, perc.size)

    for k, v in surface_a.cell_data.items():
        mesh.cell_data[k] = np.tile(v, perc.size - 1)

    return mesh


def stack_two_structured_grids(
    mesh_a: pv.StructuredGrid,
    mesh_b: pv.StructuredGrid,
    axis: int,
) -> pv.StructuredGrid:
    if sum(n == 1 for n in mesh_a.dimensions) == 1:
        axis = min(axis, 1)

    if axis == 0:
        if not (
            np.allclose(mesh_a.x[-1], mesh_b.x[0])
            and np.allclose(mesh_a.y[-1], mesh_b.y[0])
            and np.allclose(mesh_a.z[-1], mesh_b.z[0])
        ):
            raise ValueError("could not stack structured grids with non-matching east and west surfaces")

        slice_ = (slice(1, None),)

    elif axis == 1:
        if not (
            np.allclose(mesh_a.x[:, -1], mesh_b.x[:, 0])
            and np.allclose(mesh_a.y[:, -1], mesh_b.y[:, 0])
            and np.allclose(mesh_a.z[:, -1], mesh_b.z[:, 0])
        ):
            raise ValueError("could not stack structured grids with non-matching north and south surfaces")

        slice_ = (slice(None), slice(1, None))

    else:
        if not (
            np.allclose(mesh_a.x[..., -1], mesh_b.x[..., 0])
            and np.allclose(mesh_a.y[..., -1], mesh_b.y[..., 0])
            and np.allclose(mesh_a.z[..., -1], mesh_b.z[..., 0])
        ):
            raise ValueError("could not stack structured grids with non-matching top and bottom surfaces")

        slice_ = (slice(None), slice(None), slice(1, None))

    X = np.concatenate((mesh_a.x, mesh_b.x[slice_]), axis=axis)
    Y = np.concatenate((mesh_a.y, mesh_b.y[slice_]), axis=axis)
    Z = np.concatenate((mesh_a.z, mesh_b.z[slice_]), axis=axis)

    return pv.StructuredGrid(X, Y, Z)


def nsub_to_perc(nsub: int | list[float]) -> list[float]:
    if np.ndim(nsub) == 0:
        nsub = nsub if nsub else 1
        perc = np.linspace(0.0, 1.0, nsub + 1)

    elif np.ndim(nsub) == 1:
        if not all(0.0 <= n <= 1 for n in nsub):
            raise ValueError("invalid subdivisions")

        perc = np.sort(nsub)

    else:
        raise ValueError("invalid subdivisions")

    return perc


_celltype_map = np.full(int(max(pv.CellType)), -1)
_celltype_map[int(pv.CellType["TRIANGLE"])] = int(pv.CellType["WEDGE"])
_celltype_map[int(pv.CellType["QUAD"])] = int(pv.CellType["HEXAHEDRON"])
# _celltype_map[int(pv.CellType["POLYGON"])] = int(pv.CellType["POLYHEDRON"])
