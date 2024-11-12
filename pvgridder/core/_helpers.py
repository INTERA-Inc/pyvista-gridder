from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional

import numpy as np
import pyvista as pv


def generate_arc(
    radius: float,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.PolyData:
    perc = resolution_to_perc(resolution, method)
    angles = theta_min + perc * (theta_max - theta_min)
    angles = np.deg2rad(angles)
    points = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))

    return pv.MultipleLines(points)


def generate_line_from_two_points(
    point_a: ArrayLike,
    point_b: ArrayLike,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.PolyData:
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    if point_a.shape != point_b.shape:
        raise ValueError("could not generate a line from two inhomogeneous points")

    perc = resolution_to_perc(resolution, method)[:, np.newaxis]
    points = point_a + perc * (point_b - point_a)
    points = points if points.shape[1] == 3 else np.column_stack((points, np.zeros(len(points))))

    return pv.MultipleLines(points)


def generate_plane_surface_from_two_lines(
    line_a: pv.PolyData | ArrayLike,
    line_b: pv.PolyData | ArrayLike,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
    axis: int = 2,
) -> pv.StructuredGrid:
    line_points_a = line_a.points if isinstance(line_a, pv.PolyData) else np.asarray(line_a)
    line_points_b = line_b.points if isinstance(line_b, pv.PolyData) else np.asarray(line_b)
    
    if line_points_a.shape != line_points_b.shape:
        raise ValueError("could not generate plane surface from two inhomogeneous lines")

    perc = resolution_to_perc(resolution, method)[:, np.newaxis, np.newaxis]
    X, Y, Z = (line_points_a + perc * (line_points_b - line_points_a)).transpose((2, 1, 0))

    if axis == 0:
        X = np.expand_dims(X.transpose(), 2)
        Y = np.expand_dims(Y.transpose(), 2)
        Z = np.expand_dims(Z.transpose(), 2)

    elif axis == 1:
        X = np.expand_dims(X, 2)
        Y = np.expand_dims(Y, 2)
        Z = np.expand_dims(Z, 2)

    else:
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
        Z = np.expand_dims(Z, 1)

    mesh = pv.StructuredGrid(X, Y, Z)

    if isinstance(line_a, pv.PolyData):
        reps = (perc.size, 1)
        for k, v in line_a.point_data.items():
            mesh.point_data[k] = np.tile(v, reps[:v.ndim])

        reps = (perc.size - 1, 1)
        for k, v in line_a.cell_data.items():
            mesh.cell_data[k] = np.tile(v, reps[:v.ndim])

    return mesh


def generate_volume_from_two_surfaces(
    surface_a: pv.StructuredGrid | pv.UnstructuredGrid,
    surface_b: pv.StructuredGrid | pv.UnstructuredGrid,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    if surface_a.points.shape != surface_b.points.shape or not isinstance(surface_a, type(surface_b)):
        raise ValueError("could not generate volume from two inhomogeneous surfaces")

    if isinstance(surface_a, pv.StructuredGrid):
        if surface_a.dimensions != surface_b.dimensions:
            raise ValueError("could not generate volume from two inhomogeneous structured surfaces")

        if not is2d(surface_a):
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

        perc = resolution_to_perc(resolution, method)[:, np.newaxis, np.newaxis]
        X = (xa + perc * (xb - xa)).transpose(axis)
        Y = (ya + perc * (yb - ya)).transpose(axis)
        Z = (za + perc * (zb - za)).transpose(axis)
        mesh = pv.StructuredGrid(X, Y, Z)

    elif isinstance(surface_a, pv.UnstructuredGrid):
        if not is2d(surface_a):
            raise ValueError("could not generate volume from surfaces with unsupported cell types")

        if not np.allclose(surface_a.celltypes, surface_b.celltypes):
            raise ValueError("could not generate volume from two inhomogeneous unstructured surfaces")

        points_a = surface_a.points
        points_b = surface_b.points

        perc = resolution_to_perc(resolution, method)[:, np.newaxis, np.newaxis]
        points = points_a + perc * (points_b - points_a)

        n = perc.size - 1
        n_points = surface_a.n_points
        offset = surface_a.offset
        celltypes = _celltype_map[surface_a.celltypes]
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

    reps = (perc.size, 1)
    for k, v in surface_a.point_data.items():
        mesh.point_data[k] = np.tile(v, reps[:v.ndim])

    reps = (perc.size - 1, 1)
    for k, v in surface_a.cell_data.items():
        mesh.cell_data[k] = np.tile(v, reps[:v.ndim])

    return mesh


def resolution_to_perc(
    resolution: int | ArrayLike,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> ArrayLike:
    if np.ndim(resolution) == 0:
        resolution = resolution if resolution else 1
        method = method if method else "constant"

        if method == "constant":
            perc = np.linspace(0.0, 1.0, resolution + 1)

        elif method in {"log", "log_r"}:
            perc = np.log10(np.linspace(1.0, 10.0, resolution + 1))

        else:
            raise ValueError(f"invalid subdivision method '{method}'")

        if method.endswith("_r"):
            perc = 1.0 - perc

    elif np.ndim(resolution) == 1:
        if not all(0.0 <= n <= 1 for n in resolution):
            raise ValueError(f"invalid subdivision value '{resolution}'")

        perc = np.sort(resolution)

    else:
        raise ValueError(f"invalid subdivision value '{resolution}'")

    return perc


def is2d(mesh: pv.StructuredGrid | pv.UnstructuredGrid) -> bool:
    if isinstance(mesh, pv.StructuredGrid):
        return sum(n == 1 for n in mesh.dimensions) == 1

    else:
        return (_celltype_map[mesh.celltypes] != -1).all()


def translate(
    mesh: pv.StructuredGrid | pv.UnstructuredGrid,
    center: ArrayLike | None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    if center is not None:
        center = np.ravel(center)
        
        if center.size != 3:
            if center.size == 2:
                center = np.append(center, 0.0)

            else:
                raise ValueError("invalid translation vector")

        mesh = mesh.translate(center)

    return mesh


_celltype_map = np.full(int(max(pv.CellType)), -1)
_celltype_map[int(pv.CellType["TRIANGLE"])] = int(pv.CellType["WEDGE"])
_celltype_map[int(pv.CellType["QUAD"])] = int(pv.CellType["HEXAHEDRON"])
# _celltype_map[int(pv.CellType["POLYGON"])] = int(pv.CellType["POLYHEDRON"])
