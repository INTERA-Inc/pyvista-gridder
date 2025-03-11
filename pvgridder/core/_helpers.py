from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike


def generate_arc(
    radius: float,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.PolyData:
    """
    Generate an arc polyline.

    Parameters
    ----------
    radius : scalar
        Arc radius.
    theta_min : scalar, default 0.0
        Starting angle (in degree).
    theta_max : scalar, default 90.0
        Ending angle (in degree).
    resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.PolyData
        Arc polyline mesh.

    """
    perc = resolution_to_perc(resolution, method)
    angles = theta_min + perc * (theta_max - theta_min)
    angles = np.deg2rad(angles)
    points = radius * np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(len(angles)))
    )

    return pv.MultipleLines(points)


def generate_line_from_two_points(
    point_a: ArrayLike,
    point_b: ArrayLike,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.PolyData:
    """
    Generate a polyline from two points.

    Parameters
    ----------
    point_a : ArrayLike
        Starting point coordinates.
    point_b : ArrayLike
        Ending point coordinates.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the line or relative position of subdivisions
        (in percentage) with respect to the starting point.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.PolyData
        Polyline mesh.

    """
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    if point_a.shape != point_b.shape:
        raise ValueError("could not generate a line from two inhomogeneous points")

    perc = resolution_to_perc(resolution, method)[:, np.newaxis]
    points = point_a + perc * (point_b - point_a)
    points = (
        points
        if points.shape[1] == 3
        else np.column_stack((points, np.zeros(len(points))))
    )

    return pv.MultipleLines(points)


def generate_surface_from_two_lines(
    line_a: pv.PolyData | ArrayLike,
    line_b: pv.PolyData | ArrayLike,
    plane: Literal["xy", "yx", "xz", "zx", "yz", "zy"] = "xy",
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid:
    """
    Generate a surface from two polylines.

    Parameters
    ----------
    line_a : pyvista.PolyData | ArrayLike
        Starting polyline mesh or coordinates.
    line_b : pyvista.PolyData | ArrayLike
        Ending polyline mesh or coordinates.
    plane : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}, default 'xy'
        Surface plane.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the plane or relative position of subdivisions
        (in percentage) with respect to the starting line.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.StructuredGrid
        Surface mesh.

    """
    line_points_a = (
        line_a.points if isinstance(line_a, pv.PolyData) else np.asarray(line_a)
    )
    line_points_b = (
        line_b.points if isinstance(line_b, pv.PolyData) else np.asarray(line_b)
    )

    if line_points_a.shape != line_points_b.shape:
        raise ValueError(
            "could not generate plane surface from two inhomogeneous lines"
        )

    perc = resolution_to_perc(resolution, method)[:, np.newaxis, np.newaxis]
    X, Y, Z = (line_points_a + perc * (line_points_b - line_points_a)).transpose(
        (2, 1, 0)
    )

    if plane == "xy":
        X = np.expand_dims(X, 2)
        Y = np.expand_dims(Y, 2)
        Z = np.expand_dims(Z, 2)

    elif plane == "yx":
        X = np.expand_dims(X.T, 2)
        Y = np.expand_dims(Y.T, 2)
        Z = np.expand_dims(Z.T, 2)

    elif plane == "xz":
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
        Z = np.expand_dims(Z, 1)

    elif plane == "zx":
        X = np.expand_dims(X.T, 1)
        Y = np.expand_dims(Y.T, 1)
        Z = np.expand_dims(Z.T, 1)

    elif plane == "yz":
        X = np.expand_dims(X, 0)
        Y = np.expand_dims(Y, 0)
        Z = np.expand_dims(Z, 0)

    elif plane == "zy":
        X = np.expand_dims(X.T, 0)
        Y = np.expand_dims(Y.T, 0)
        Z = np.expand_dims(Z.T, 0)

    else:
        raise ValueError(f"invalid plane '{plane}'")

    mesh = pv.StructuredGrid(X, Y, Z)

    if isinstance(line_a, pv.PolyData):
        reps = (perc.size, 1)
        for k, v in line_a.point_data.items():
            mesh.point_data[k] = np.tile(v, reps[: v.ndim])

        reps = (perc.size - 1, 1)
        for k, v in line_a.cell_data.items():
            mesh.cell_data[k] = np.tile(v, reps[: v.ndim])

    # Handle collapsed cells
    if "vtkGhostType" not in mesh.cell_data:
        mesh.cell_data["vtkGhostType"] = np.zeros(mesh.n_cells, dtype=np.uint8)
        
    areas = mesh.compute_cell_sizes(
        length=False, area=True, volume=False
    ).cell_data["Area"]
    mesh.cell_data["vtkGhostType"][np.abs(areas) == 0.0] = 32

    return mesh


def generate_volume_from_two_surfaces(
    surface_a: pv.StructuredGrid | pv.UnstructuredGrid,
    surface_b: pv.StructuredGrid | pv.UnstructuredGrid,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Generate a volume from two surface meshes.

    Parameters
    ----------
    surface_a : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Starting surface mesh.
    surface_b : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Ending surface mesh.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the extrusion axis or relative position of
        subdivisions (in percentage) with respect to the starting surface.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Volume mesh.

    """
    from .. import get_dimension

    if surface_a.points.shape != surface_b.points.shape or not isinstance(
        surface_a, type(surface_b)
    ):
        raise ValueError("could not generate volume from two inhomogeneous surfaces")

    if isinstance(surface_a, pv.StructuredGrid):
        if surface_a.dimensions != surface_b.dimensions:
            raise ValueError(
                "could not generate volume from two inhomogeneous structured surfaces"
            )

        if get_dimension(surface_a) != 2:
            raise ValueError("could not generate volume from non 2D structured grid")

        nx, ny, nz = surface_a.dimensions
        perc = resolution_to_perc(resolution, method)

        if nx == 1:
            axis = 0
            slice_ = (0,)
            perc = perc[:, np.newaxis, np.newaxis]

        elif ny == 1:
            axis = 1
            slice_ = (slice(None), 0)
            perc = perc[:, np.newaxis]

        elif nz == 1:
            axis = 2
            slice_ = (slice(None), slice(None), 0)

        xa, ya, za = surface_a.x[slice_], surface_a.y[slice_], surface_a.z[slice_]
        xb, yb, zb = surface_b.x[slice_], surface_b.y[slice_], surface_b.z[slice_]
        xa = np.expand_dims(xa, axis)
        ya = np.expand_dims(ya, axis)
        za = np.expand_dims(za, axis)
        xb = np.expand_dims(xb, axis)
        yb = np.expand_dims(yb, axis)
        zb = np.expand_dims(zb, axis)

        X = xa + perc * (xb - xa)
        Y = ya + perc * (yb - ya)
        Z = za + perc * (zb - za)
        mesh = pv.StructuredGrid(X, Y, Z)

        # Handle collapsed cells
        volumes = mesh.compute_cell_sizes(
            length=False, area=False, volume=True
        ).cell_data["Volume"]
        inactive = np.abs(volumes) == 0.0

        # Repeat data
        shape = surface_a.dimensions
        for k, v in surface_a.point_data.items():
            mesh.point_data[k] = repeat_structured_data(shape, v, perc.size, axis)

        shape = [max(1, n - 1) for n in surface_a.dimensions]
        for k, v in surface_a.cell_data.items():
            mesh.cell_data[k] = repeat_structured_data(shape, v, perc.size - 1, axis)

    elif isinstance(surface_a, pv.UnstructuredGrid):
        if not (_celltype_map[surface_a.celltypes] != -1).all():
            raise ValueError(
                "could not generate volume from surfaces with unsupported cell types"
            )

        if not np.allclose(surface_a.celltypes, surface_b.celltypes):
            raise ValueError(
                "could not generate volume from two inhomogeneous unstructured surfaces"
            )

        points_a = surface_a.points
        points_b = surface_b.points
        n_points = surface_a.n_points

        perc = resolution_to_perc(resolution, method)[:, np.newaxis, np.newaxis]
        points = points_a + perc * (points_b - points_a)
        points = points.reshape((n_points * perc.size, 3))

        n = perc.size - 1
        offset = surface_a.offset
        celltypes = _celltype_map[surface_a.celltypes]
        cell_connectivity = surface_a.cell_connectivity
        cells = [[] for _ in range(n)]
        inactive = [[] for _ in range(n)]

        for i1, i2, celltype in zip(offset[:-1], offset[1:], celltypes):
            cell = cell_connectivity[i1:i2]
            faces = [cell, cell + n_points]

            # Handle collapsed cells
            is_collapsed = np.allclose(*points[faces])

            if celltype == 42:
                faces += [
                    np.array([p0, p1, p2, p3])
                    for p0, p1, p2, p3 in zip(
                        faces[0], np.roll(faces[0], -1), np.roll(faces[1], -1), faces[1]
                    )
                ]
                n_faces = len(faces)

                for i, cells_ in enumerate(cells):
                    cell = np.concatenate(
                        [[face.size, *(face + (i * n_points))] for face in faces]
                    )
                    cells_ += [cell.size + 1, n_faces, *cell]
                    inactive[i].append(is_collapsed)

            else:
                cell = np.concatenate(faces)

                for i, cells_ in enumerate(cells):
                    cells_ += [cell.size, *(cell + (i * n_points))]
                    inactive[i].append(is_collapsed)

        cells = np.concatenate(cells)
        inactive = np.concatenate(inactive)
        celltypes = np.tile(celltypes, n)
        mesh = pv.UnstructuredGrid(cells, celltypes, points)

        # Repeat data
        reps = (perc.size, 1)
        for k, v in surface_a.point_data.items():
            mesh.point_data[k] = np.tile(v, reps[: v.ndim])

        reps = (perc.size - 1, 1)
        for k, v in surface_a.cell_data.items():
            mesh.cell_data[k] = np.tile(v, reps[: v.ndim])

    else:
        raise ValueError(f"could not generate volume from {type(surface_a)}")

    # Handle collapsed cells
    if "vtkGhostType" not in mesh.cell_data:
        mesh.cell_data["vtkGhostType"] = np.zeros(mesh.n_cells, dtype=np.uint8)

    mesh.cell_data["vtkGhostType"][inactive] = 32

    return mesh


def resolution_to_perc(
    resolution: int | ArrayLike,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> ArrayLike:
    """
    Convert resolution to relative position.

    Parameters
    ----------
    resolution : int | ArrayLike
        Number of subdivisions or relative position of subdivisions (in percentage).
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    ArrayLike
        Relative position of subdivisions (in percentage).

    """
    if np.ndim(resolution) == 0:
        resolution = resolution if resolution else 1
        method = method if method else "constant"

        if method == "constant":
            perc = np.linspace(0.0, 1.0, resolution + 1)

        elif method in {"log", "log_r"}:
            perc = np.log10(np.linspace(1.0, 10.0, resolution + 1))

        else:
            raise ValueError(f"invalid subdivision method '{method}'")

        if not (method == "constant" or method.endswith("_r")):
            perc = 1.0 - perc

    elif np.ndim(resolution) == 1:
        perc = np.sort(resolution)

    else:
        raise ValueError(f"invalid subdivision value '{resolution}'")

    return perc


def repeat_structured_data(shape: ArrayLike, data: ArrayLike, repeats: int, axis: int) -> ArrayLike:
    """
    Repeat structured data array.

    Parameters
    ----------
    shape : ArrayLike
        Structured grid shape.
    data : ArrayLike
        Data array to repeat.
    repeats : int
        The number of repetitions.
    axis : int
        The axis along which to repeat values.

    Returns
    -------
    ArrayLike
        Data array with repeated values.

    """
    data = data if data.ndim == 2 else data[:, np.newaxis]

    return np.column_stack(
        [
            np.repeat(
                v.reshape(shape, order="F"),
                repeats,
                axis,
            ).ravel(order="F")
            for v in data.T
        ]
    ).squeeze()


def translate(
    mesh: pv.StructuredGrid | pv.UnstructuredGrid,
    vector: ArrayLike | None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Translate a mesh.

    Parameters
    ----------
    mesh : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Mesh to translate.
    vector : ArrayLike | None
        Translation vector. If None, no translation is performed.

    Returns
    -------
    pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Translated mesh.

    """
    if vector is not None:
        vector = np.ravel(vector)

        if vector.size != 3:
            if vector.size == 2:
                vector = np.append(vector, 0.0)

            else:
                raise ValueError("invalid translation vector")

        mesh = mesh.translate(vector)

    return mesh


_celltype_map = -np.ones(int(max(pv.CellType)) + 1, dtype=int)
_celltype_map[pv.CellType.TRIANGLE] = int(pv.CellType.WEDGE)
_celltype_map[pv.CellType.QUAD] = int(pv.CellType.HEXAHEDRON)
_celltype_map[pv.CellType.POLYGON] = int(pv.CellType.POLYHEDRON)
