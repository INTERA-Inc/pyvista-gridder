from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

from scipy.spatial import KDTree


def average_points(mesh: pv.PolyData, tolerance: float = 0.0) -> pv.PolyData:
    """
    Average duplicate points in this mesh.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh to average points from.
    tolerance : float, default 0.0
        Specify a tolerance to use when comparing points. Points within this tolerance
        will be averaged.

    Returns
    -------
    pyvista.PolyData
        Mesh with averaged points.

    """
    def decimate(cell: ArrayLike, close: bool) -> ArrayLike:
        cell = cell[np.insert(np.diff(cell), 0, 1) != 0]

        return cell[:-1] if close and cell[0] == cell[-1] else cell

    points = mesh.points
    groups, group_map = [], {}

    for i, j in KDTree(points).query_pairs(tolerance):
        igrp = group_map[i] if i in group_map else -1
        jgrp = group_map[j] if j in group_map else -1

        if igrp >= 0 and jgrp < 0:
            groups[igrp].append(j)
            group_map[j] = igrp

        elif igrp < 0 and jgrp >= 0:
            groups[jgrp].append(i)
            group_map[i] = jgrp

        elif igrp >= 0 and jgrp >= 0:
            if igrp != jgrp:
                group_map.update({k: igrp for k in groups[jgrp]})
                groups[igrp] += groups[jgrp]
                groups[jgrp] = []

        else:
            gid = len(groups)
            groups.append([i, j])
            group_map[i] = gid
            group_map[j] = gid
    
    point_map = np.arange(mesh.n_points)
    new_points = points.copy()

    for group in groups:
        if not group:
            continue

        point_map[group] = group[0]
        new_points[group[0]] = points[group].mean(axis=0)

    if mesh.n_faces_strict:
        irregular_faces = [decimate(point_map[face], close=True) for face in mesh.irregular_faces]
        faces = [face for face in irregular_faces if face.size > 2]
        new_mesh = pv.PolyData().from_irregular_faces(new_points, faces)
        new_mesh.cell_data["vtkOriginalCellIds"] = [i for i, face in enumerate(irregular_faces) if face.size > 2]

    else:
        new_mesh = pv.PolyData(new_points)

    if mesh.n_lines:
        raise NotImplementedError()

    return new_mesh.clean()


def decimate_rdp(mesh: pv.PolyData, tolerance: float = 1.0e-8) -> pv.PolyData:
    """
    Decimate polylines and/or polygons in a polydata.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Polydata to decimate.
    tolerance : scalar, default 1.0e-8
        Tolerance for the Ramer-Douglas-Peucker algorithm.

    Returns
    -------
    pyvista.PolyData
        Decimated polydata.

    """

    def decimate(points: ArrayLike) -> ArrayLike:
        """Ramer-Douglas-Packer algorithm."""
        u = points[-1] - points[0]
        un = np.linalg.norm(u)
        dist = (
            np.linalg.norm(np.cross(u, points[0] - points), axis=1) / un
            if un > 0.0
            else np.linalg.norm(points - points[0], axis=1)
        )
        imax = dist.argmax()

        if dist[imax] > tolerance:
            res1 = decimate(points[: imax + 1])
            res2 = decimate(points[imax:])

            return np.row_stack((res1[:-1], res2))

        else:
            return np.row_stack((points[0], points[-1]))

    lines = []
    points = []

    for cell in mesh.cell:
        if cell.type.name in {"LINE", "POLY_LINE"}:
            points_ = mesh.points[cell.point_ids]

            if cell.type.name == "POLY_LINE":
                points_ = decimate(points_)

            lines += [len(points_), *(np.arange(len(points_)) + len(points))]
            points += points_.tolist()

    return pv.PolyData(points, lines=lines).clean()


def extract_boundary_polygons(
    mesh: pv.DataSet,
    fill: bool = False,
) -> Sequence[pv.PolyData]:
    """
    Extract boundary edges of a mesh as continuous polylines or polygons.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to extract boundary edges from.
    fill : bool, default False
        If False, only return boundary edges as polylines.

    Returns
    -------
    Sequence[pyvista.PolyData]
        Extracted boundary polylines or polygons.

    """
    poly = mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
        clear_data=True,
    )
    lines = poly.lines.reshape((poly.n_cells, 3))[:, 1:]
    lines = np.sort(lines, axis=1).tolist()
    polygon, polygons = [], []

    while lines:
        if not polygon:
            polygon += lines.pop(0)

        cond = np.array(lines) == polygon[-1]

        if cond[:, 0].any():
            j1, j2 = 0, 1

        elif cond[:, 1].any():
            j1, j2 = 1, 0

        else:
            raise ValueError("could not match end point with another start point")

        i = np.flatnonzero(cond[:, j1])[0]
        polygon.append(lines[i][j2])
        _ = lines.pop(i)

        if polygon[-1] == polygon[0]:
            polygons.append(polygon)
            polygon = []

    return [
        pv.PolyData(
            poly.points[polygon[:-1]],
            lines=[len(polygon), *list(range(len(polygon) - 1)), 0],
            faces=[len(polygon) - 1, *list(range(len(polygon) - 1))],
        )
        if fill
        else pv.PolyData(
            poly.points[polygon[:-1]],
            lines=[len(polygon), *list(range(len(polygon) - 1)), 0],
        )
        for polygon in polygons
    ]


def extract_cell_geometry(
    mesh: pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid,
    remove_empty_cells: bool = True,
) -> pv.PolyData:
    """
    Extract the geometry of individual cells.

    Parameters
    ----------
    mesh : pyvista.ExplicitStructuredGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Mesh to extract cell geometry from.
    remove_empty_cells : bool, default True
        If True, remove empty cells.

    Returns
    -------
    pyvista.PolyData
        Extracted cell geometry.

    """

    def get_polydata_from_points_cells(
        points: ArrayLike,
        cells: ArrayLike,
        key: str,
    ) -> pv.PolyData:
        cell_ids, cells_, lines_or_faces = [], [], []
        cell_map = {}

        for i, cell in enumerate(cells):
            if len(cell) == 0:
                continue

            for c in cell:
                cell_set = tuple(sorted(set(c)))

                try:
                    idx = cell_map[cell_set]
                    cell_ids[idx].append(i)

                except KeyError:
                    idx = len(cell_map)
                    cell_map[cell_set] = idx
                    cell_ids.append([i])

                    cells_.append(c)
                    lines_or_faces += [len(c), *cells_[idx]]

        tmp = -np.ones((len(cell_ids), 2), dtype=int)
        for i, ids in enumerate(cell_ids):
            tmp[i, : len(ids)] = ids

        poly = pv.PolyData(points, **{key: lines_or_faces})
        poly.cell_data["vtkOriginalCellIds"] = tmp

        return poly

    from .. import get_dimension

    if not remove_empty_cells and "vtkGhostType" in mesh.cell_data:
        mesh = mesh.copy(deep=False)
        mesh.clear_data()

    ndim = get_dimension(mesh)
    mesh = mesh.cast_to_unstructured_grid()

    offset = mesh.offset
    celltypes = mesh.celltypes
    connectivity = mesh.cell_connectivity

    if ndim in {1, 2}:
        supported_celltypes = {
            pv.CellType.EMPTY_CELL,
            pv.CellType.LINE,
            pv.CellType.POLYGON,
            pv.CellType.POLY_LINE,
            pv.CellType.QUAD,
            pv.CellType.TRIANGLE,
        }
        unsupported_celltypes = set(celltypes).difference(supported_celltypes)

        if unsupported_celltypes:
            raise NotImplementedError(
                f"cells of type '{pv.CellType(list(unsupported_celltypes)[0]).name}' are not supported yet"
            )

        # Generate edge data
        cell_edges = [
            np.column_stack((connectivity[i1:i2 - 1], connectivity[i1 + 1:i2]))
            if celltype in {pv.CellType.LINE, pv.CellType.POLY_LINE}
            else np.column_stack((connectivity[i1:i2], np.roll(connectivity[i1:i2], -1)))
            if not remove_empty_cells or celltype != pv.CellType.EMPTY_CELL
            else []
            for i, (i1, i2, celltype) in enumerate(
                zip(offset[:-1], offset[1:], celltypes)
            )
        ]

        poly = get_polydata_from_points_cells(mesh.points, cell_edges, "lines")

        # Handle collapsed cells
        if remove_empty_cells:
            lengths = poly.compute_cell_sizes(length=True, area=False, volume=False)[
                "Length"
            ]
            mask = np.abs(lengths) > 0.0

            if not mask.all():
                lines = poly.lines.reshape((poly.n_lines, 3))[mask]
                tmp = poly.cell_data["vtkOriginalCellIds"][mask]

                poly = pv.PolyData(poly.points, lines=lines)
                poly.cell_data["vtkOriginalCellIds"] = tmp

    else:
        # Generate polyhedral cell faces if any
        polyhedral_cells = pv.convert_array(mesh.GetFaces())

        if polyhedral_cells is not None:
            locations = pv.convert_array(mesh.GetFaceLocations())
            polyhedral_cell_faces = []

            for location in locations:
                if location == -1:
                    continue

                n_faces = polyhedral_cells[location]
                i, cell = location + 1, []

                while len(cell) < n_faces:
                    n_vertices = polyhedral_cells[i]
                    cell.append(polyhedral_cells[i + 1 : i + 1 + n_vertices])
                    i += n_vertices + 1

                polyhedral_cell_faces.append(cell)

        # Generate face data
        if celltypes.min() == celltypes.max():
            celltype = pv.CellType(celltypes[0]).name

            if celltype == "POLYHEDRON":
                cell_faces = polyhedral_cell_faces

            else:
                n_vertices = _celltype_to_n_vertices[celltype]
                cells = connectivity.reshape(
                    (connectivity.size // n_vertices, n_vertices)
                )
                cell_faces = [
                    [
                        face
                        for v in _celltype_to_faces[celltype].values()
                        for face in cell[v]
                    ]
                    for cell in cells
                ]

        else:
            polyhedron_count, cell_faces = 0, []

            for i, (i1, i2, celltype) in enumerate(
                zip(offset[:-1], offset[1:], celltypes)
            ):
                celltype = pv.CellType(celltype).name

                if celltype == "POLYHEDRON":
                    cell_face = polyhedral_cell_faces[polyhedron_count]
                    polyhedron_count += 1

                elif celltype in _celltype_to_faces:
                    cell = connectivity[i1:i2]
                    cell_face = [
                        face
                        for v in _celltype_to_faces[celltype].values()
                        for face in cell[v]
                    ]

                elif celltype == "EMPTY_CELL":
                    cell_face = []

                else:
                    raise NotImplementedError(
                        f"cells of type '{celltype}' are not supported yet"
                    )

                cell_faces.append(cell_face)

        poly = get_polydata_from_points_cells(mesh.points, cell_faces, "faces")

        # Handle collapsed cells
        if remove_empty_cells:
            areas = poly.compute_cell_sizes(length=False, area=True, volume=False)[
                "Area"
            ]
            mask = np.abs(areas) > 0.0

            if not mask.all():
                faces = [
                    face for face, mask_ in zip(poly.irregular_faces, mask) if mask_
                ]
                tmp = poly.cell_data["vtkOriginalCellIds"][mask]

                poly = pv.PolyData().from_irregular_faces(poly.points, faces)
                poly.cell_data["vtkOriginalCellIds"] = tmp

    return poly


def extract_cells_by_dimension(
    mesh: pv.UnstructuredGrid,
    ndim: Optional[int] = None,
    method: Literal["lower", "upper"] = "upper",
) -> pv.UnstructuredGrid:
    """
    Extract cells by a specified dimension.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        Mesh to extract cells from.
    ndim : int, optional
        Dimension to be used for extraction. If None, the dimension of *mesh* is used.
    method : {'lower', 'upper'}, default 'upper'
        Set the extraction method. 'lower' will extract cells of dimension lower than
        *ndim*. 'upper' will extract cells of dimension larger than *ndim*.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh with extracted cells.

    """
    from ._properties import _dimension_map, get_dimension

    ndim = ndim if ndim is not None else get_dimension(mesh)

    if method == "upper":
        mask = _dimension_map[mesh.celltypes] < ndim

    elif method == "lower":
        mask = _dimension_map[mesh.celltypes] > ndim

    else:
        raise ValueError(f"invalid method '{method}' (expected 'lower' or 'upper')")

    if mask.any():
        mesh = mesh.extract_cells(~mask)

    return mesh


def merge(
    mesh_a: pv.StructuredGrid | pv.UnstructuredGrid,
    mesh_b: pv.StructuredGrid | pv.UnstructuredGrid,
    axis: Optional[int] = None,
    merge_points: bool = True,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Merge two meshes.

    Parameters
    ----------
    mesh_a, mesh_b : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        The meshes to merge together.
    axis : int, optional
        The axis along which two structured grids are merged (if *mesh_a* and *mesh_b*
        are structured grids).
    merge_points : bool, default True
        If True, merge equivalent points for two unstructured grids.

    Returns
    -------
    pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Merged mesh.

    """
    if isinstance(mesh_a, pv.StructuredGrid) and isinstance(mesh_b, pv.StructuredGrid):
        if axis is None:
            raise ValueError("could not merge structured grids with None axis")

        if axis == 0:
            if not (
                np.allclose(mesh_a.x[-1], mesh_b.x[0])
                and np.allclose(mesh_a.y[-1], mesh_b.y[0])
                and np.allclose(mesh_a.z[-1], mesh_b.z[0])
            ):
                raise ValueError(
                    "could not merge structured grids with non-matching east and west interfaces"
                )

            slice_ = (slice(1, None),)

        elif axis == 1:
            if not (
                np.allclose(mesh_a.x[:, -1], mesh_b.x[:, 0])
                and np.allclose(mesh_a.y[:, -1], mesh_b.y[:, 0])
                and np.allclose(mesh_a.z[:, -1], mesh_b.z[:, 0])
            ):
                raise ValueError(
                    "could not merge structured grids with non-matching north and south interfaces"
                )

            slice_ = (slice(None), slice(1, None))

        else:
            if not (
                np.allclose(mesh_a.x[..., -1], mesh_b.x[..., 0])
                and np.allclose(mesh_a.y[..., -1], mesh_b.y[..., 0])
                and np.allclose(mesh_a.z[..., -1], mesh_b.z[..., 0])
            ):
                raise ValueError(
                    "could not merge structured grids with non-matching top and bottom interfaces"
                )

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
        mesh = pv.merge((mesh_a, mesh_b), merge_points=merge_points, main_has_priority=True)

    return mesh


def merge_lines(
    lines: Sequence[pv.PolyData],
    as_lines: bool = True,
) -> pv.PolyData:
    """
    Merge line(s) or polyline(s) into a polydata.

    Parameters
    ----------
    lines : Sequence[pyvista.PolyData]
        List of line(s) or polyline(s) to merge.
    as_lines : bool, default True
        If True, return merged line(s) or polyline(s) as line(s).

    Returns
    -------
    pyvista.PolyData
        Polydata with merged line(s) or polyline(s).

    Note
    ----
    Preserve ordering compared to pyvista.merge().

    """
    points, cells, offset = [], [], 0

    for lines_ in lines:
        for line in split_lines(lines_, as_lines=False):
            points.append(line.points)
            ids = np.arange(line.n_points) + offset
            cells += (
                np.insert(np.column_stack((ids[:-1], ids[1:])), 0, 2, axis=-1).ravel().tolist()
                if as_lines
                else [line.n_points, *ids]
            )
            offset += line.n_points

    return pv.PolyData(np.concatenate(points), lines=cells).merge_points()


def reconstruct_line(
    mesh: pv.DataSet,
    start: int = 0,
    close: bool = False,
    tolerance: float = 1.0e-8,
) -> pv.PolyData:
    """
    Reconstruct a line from the points in this dataset.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh from which points to reconstruct a line.
    start : int, default 0
        Index of point to use as starting point for 2-opt algorithm.
    close : bool, default False
        If True, the ending point is the starting point.
    tolerance : scalar, default 1.0e-8
        Tolerance for the 2-opt algorithm.

    Returns
    -------
    pyvista.PolyData
        Reconstructed line.

    """
    points = mesh.points

    if not (points.ndim == 2 and points.shape[1] in {2, 3}):
        raise ValueError(
            f"could not reconstruct polyline from {points.shape[1]}D points"
        )

    def path_length(path):
        if close:
            path = np.append(path, path[0])

        return np.linalg.norm(np.diff(points[path], axis=0), axis=1).sum()

    n = len(points)
    shortest_path = np.roll(np.arange(n), -start)
    shortest_length = path_length(shortest_path)
    path = shortest_path.copy()

    while True:
        best_length = shortest_length

        for first in range(1, n - 2):
            for last in range(first + 2, n + 1):
                path[first:last] = np.flip(path[first:last])
                length = path_length(path)

                if length < shortest_length:
                    shortest_path[:] = path
                    shortest_length = length

                else:
                    path[first:last] = np.flip(
                        path[first:last]
                    )  # reset path to current shortest path

        if shortest_length > (1.0 - tolerance) * best_length:
            break

    points = points[shortest_path]
    points = (
        points
        if points.shape[1] == 3
        else np.column_stack((points, np.zeros(len(points))))
    )

    return pv.lines_from_points(points, close=close)


def remap_categorical_data(
    mesh: pv.DataSet,
    key: str,
    mapping: dict[int, int],
    preference: Literal["cell", "point"] = "cell",
    inplace: bool = False,
) -> pv.DataSet | None:
    """
    Remap categorical cell or point data.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh with categorical data to remap.
    key : str
        Name of the categorical data to remap.
    mapping : dict
        Mapping of old to new values.
    preference : {'cell', 'point'}, default 'cell'
        Determine whether to remap cell or point data.
    inplace : bool, default False
        If True, overwrite the original mesh.

    Returns
    -------
    pyvista.DataSet | None
        Mesh with remapped categorical data.

    """
    if preference == "cell":
        data = mesh.cell_data[key]

    elif preference == "point":
        data = mesh.point_data[key]

    else:
        raise ValueError(f"invalid preference '{preference}'")

    if data.dtype.kind != "i":
        raise ValueError(f"could not remap non-categorical '{preference}' data '{key}'")

    try:
        data_labels = dict(mesh.user_dict[key])

    except KeyError:
        data_labels = {}

    if not inplace:
        mesh = mesh.copy()

    remapped_data = data.copy()
    data_labels_map = {v: k for k, v in data_labels.items()}
    unused_labels = set(list(data_labels))

    for k, v in mapping.items():
        if isinstance(k, str):
            try:
                vid = data_labels[k]

            except KeyError:
                raise ValueError(f"could not map unknown key '{k}'")

        else:
            vid = k

        mask = data == vid

        if mask.any():
            remapped_data[mask] = v

            try:
                key_ = k if isinstance(k, str) else data_labels_map[vid]
                data_labels[key_] = v
                unused_labels.remove(key_)

            except KeyError:
                pass

    if preference == "cell":
        mesh.cell_data[key] = remapped_data

    else:
        mesh.point_data[key] = remapped_data

    if data_labels:
        for k in unused_labels:
            data_labels.pop(k, None)

        mesh.user_dict[key] = dict(sorted(data_labels.items(), key=lambda x: x[1]))

    if not inplace:
        return mesh


def split_lines(mesh: pv.PolyData, as_lines: bool = True) -> Sequence[pv.PolyData]:
    """
    Split line(s) or polyline(s) into multiple line(s) or polyline(s).

    Parameters
    ----------
    mesh : pyvista.PolyData
        Mesh with line(s) or polyline(s) to split.
    as_lines : bool, default True
        If True, return split line(s) or polyline(s) as line(s).

    Returns
    -------
    Sequence[pyvista.PolyData]
        Split line(s) or polyline(s).

    """
    from pyvista.core.cell import _get_irregular_cells

    return [
        pv.PolyData(
            mesh.points[line],
            lines=(
                np.insert(
                    np.column_stack((np.arange(line.size - 1), np.arange(1, line.size))), 0, 2,
                    axis=-1
                ).ravel()
                if as_lines
                else [line.size, *np.arange(line.size)]
            )
        )
        for line in _get_irregular_cells(mesh.GetLines())
    ]


def quadraticize(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """
    Convert linear mesh to quadratic mesh.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        Mesh with linear cells.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh with quadratic cells.

    """
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


_celltype_to_faces = {
    "TETRA": {
        "TRIANGLE": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
    },
    "PYRAMID": {
        "QUAD": np.array([[0, 3, 2, 1]]),
        "TRIANGLE": np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    },
    "WEDGE": {
        "TRIANGLE": np.array([[0, 2, 1], [3, 4, 5]]),
        "QUAD": np.array([[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]),
    },
    "HEXAHEDRON": {
        "QUAD": np.array(
            [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [0, 4, 7, 3],
            ]
        ),
    },
}

_celltype_to_n_vertices = {
    "TETRA": 4,
    "PYRAMID": 5,
    "WEDGE": 6,
    "HEXAHEDRON": 8,
}
