from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal, Optional

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
        irregular_faces = [
            decimate(point_map[face], close=True) for face in mesh.irregular_faces
        ]
        faces = [face for face in irregular_faces if face.size > 2]
        new_mesh = pv.PolyData().from_irregular_faces(new_points, faces)
        new_mesh.cell_data["vtkOriginalCellIds"] = [
            i for i, face in enumerate(irregular_faces) if face.size > 2
        ]

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

            return np.vstack((res1[:-1], res2))

        else:
            return np.vstack((points[0], points[-1]))

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
    with_holes: bool = False,
) -> (
    Sequence[pv.PolyData | pv.UnstructuredGrid]
    | Sequence[Sequence[pv.PolyData | pv.UnstructuredGrid]]
    | None
):
    """
    Extract boundary edges of a mesh as continuous polylines or polygons.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to extract boundary edges from.
    fill : bool, default False
        If True, return boundary edges as polygons.
    with_holes : bool, default False
        If True, group holes with their corresponding boundary edges.

    Returns
    -------
    Sequence[pyvista.PolyData | pyvista.UnstructuredGrid] | Sequence[Sequence[pyvista.PolyData | pyvista.UnstructuredGrid]] | None
        Extracted boundary polylines or polygons.

    """
    import shapely

    from .. import Polygon

    if isinstance(mesh, pv.PolyData) and mesh.n_faces_strict == 0 and mesh.n_lines > 0:
        mesh = edges

    else:
        edges = (
            mesh.cast_to_unstructured_grid()
            .clean()
            .extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=False,
                feature_edges=False,
                manifold_edges=False,
                clear_data=True,
            )
        )

    edges = edges.strip(max_length=mesh.n_points)

    if edges.n_lines == 0:
        return None

    edges = [edge.merge_points() for edge in split_lines(edges, as_lines=False)]

    # Identify holes
    if with_holes:
        holes = {}
        polygons = [shapely.Polygon(edge.points) for edge in edges]

        for i, j in itertools.permutations(range(len(polygons)), 2):
            if i in holes or j in holes:
                continue

            if polygons[i].area > polygons[j].area and polygons[i].contains(
                polygons[j]
            ):
                holes[j] = i

        # Group boundary edges and holes
        polygons = [[edge] if i not in holes else [] for i, edge in enumerate(edges)]

        for k, v in holes.items():
            polygons[v].append(edges[k])

        polygons = [polygon for polygon in polygons if polygon]
        polygons = (
            [Polygon(polygon[0], polygon[1:]) for polygon in polygons]
            if fill
            else polygons
        )

    else:
        polygons = (
            [
                polygon
                + pv.PolyData().from_regular_faces(
                    polygon.points, [np.arange(polygon.n_points)]
                )
                for polygon in edges
            ]
            if fill
            else edges
        )

    return tuple(polygons)


def extract_cell_geometry(
    mesh: pv.DataSet,
    remove_ghost_cells: bool = True,
) -> pv.PolyData:
    """
    Extract the geometry of individual cells.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to extract cell geometry from.
    remove_ghost_cells : bool, default True
        If True, remove ghost cells.

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

        tmp = -np.ones((len(cell_ids), len(max(cell_ids, key=len))), dtype=int)
        for i, ids in enumerate(cell_ids):
            tmp[i, : len(ids)] = ids

        poly = pv.PolyData(points, **{key: lines_or_faces})
        poly.cell_data["vtkOriginalCellIds"] = tmp

        return poly

    from .. import get_cell_connectivity, get_dimension

    if not remove_ghost_cells and "vtkGhostType" in mesh.cell_data:
        mesh = mesh.copy(deep=False)
        mesh.clear_data()

    ndim = get_dimension(mesh)
    mesh = mesh.cast_to_unstructured_grid()
    celltypes = mesh.celltypes
    connectivity = get_cell_connectivity(mesh)

    if ndim in {1, 2}:
        supported_celltypes = {
            pv.CellType.EMPTY_CELL,
            pv.CellType.LINE,
            pv.CellType.PIXEL,
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
            np.column_stack((cell[:-1], cell[1:]))
            if celltype in {pv.CellType.LINE, pv.CellType.POLY_LINE}
            else np.column_stack(
                (
                    cell[[0, 1, 3, 2]],
                    np.roll(cell[[0, 1, 3, 2]], -1),
                )
            )
            if celltype == pv.CellType.PIXEL
            else np.column_stack((cell, np.roll(cell, -1)))
            if not remove_ghost_cells or celltype != pv.CellType.EMPTY_CELL
            else []
            for cell, celltype in zip(connectivity, celltypes)
        ]

        poly = get_polydata_from_points_cells(mesh.points, cell_edges, "lines")

        # Handle collapsed cells
        if remove_ghost_cells:
            lengths = poly.compute_cell_sizes(length=True, area=False, volume=False)[
                "Length"
            ]
            mask = np.abs(lengths) > 0.0

            if not mask.all():
                lines = poly.lines.reshape((poly.n_lines, 3))[mask]
                tmp = poly.cell_data["vtkOriginalCellIds"][mask]
                tmp = tmp[:, np.ptp(tmp, axis=0) > 0]

                poly = pv.PolyData(poly.points, lines=lines)
                poly.cell_data["vtkOriginalCellIds"] = tmp

    else:
        # Generate face data
        if np.ptp(celltypes) == 0:
            celltype = celltypes[0]
            cell_faces = (
                [
                    [
                        face
                        for v in _celltype_to_faces[celltype].values()
                        for face in cell[v]
                    ]
                    for cell in connectivity
                ]
                if celltype != pv.CellType.POLYHEDRON
                else connectivity
            )

        else:
            cell_faces = []

            for cell, celltype in zip(connectivity, celltypes):
                if celltype == pv.CellType.POLYHEDRON:
                    cell_face = cell

                elif celltype in _celltype_to_faces:
                    cell_face = [
                        face
                        for v in _celltype_to_faces[celltype].values()
                        for face in cell[v]
                    ]

                elif celltype == pv.CellType.EMPTY_CELL:
                    cell_face = []

                else:
                    raise NotImplementedError(
                        f"cells of type '{celltype.name}' are not supported yet"
                    )

                cell_faces.append(cell_face)

        poly = get_polydata_from_points_cells(mesh.points, cell_faces, "faces")

        # Handle collapsed cells
        if remove_ghost_cells:
            areas = poly.compute_cell_sizes(length=False, area=True, volume=False)[
                "Area"
            ]
            mask = np.abs(areas) > 0.0

            if not mask.all():
                faces = [
                    face for face, mask_ in zip(poly.irregular_faces, mask) if mask_
                ]
                tmp = poly.cell_data["vtkOriginalCellIds"][mask]
                tmp = tmp[:, np.ptp(tmp, axis=0) > 0]

                poly = pv.PolyData().from_irregular_faces(poly.points, faces)
                poly.cell_data["vtkOriginalCellIds"] = tmp

    return poly


def extract_cells(
    mesh: pv.DataSet,
    ind: ArrayLike,
    invert: bool = False,
    progress_bar: bool = False,
) -> pv.UnstructuredGrid:
    """
    Return a subset of the grid.

    Parameters
    ----------
    ind : ArrayLike
        Indices of cells to extract.
    invert : bool, default False
        If True, invert the selection.
    progress_bar : bool, default False
        If True, display a progress bar.

    Returns
    -------
    pyvista.UnstructuredGrid
        Extracted cells.

    Note
    ----
    This function wraps `pyvista.DataSet.extract_cells()` with consistent handling of
    ghost cells across different versions of VTK.

    """
    ghost_cells = (
        mesh.cell_data.pop("vtkGhostType") if "vtkGhostType" in mesh.cell_data else None
    )

    try:
        cells = mesh.extract_cells(ind, invert=invert, progress_bar=progress_bar)

        if ghost_cells is not None:
            cells.cell_data["vtkGhostType"] = ghost_cells[
                cells.cell_data["vtkOriginalCellIds"]
            ]

    finally:
        if ghost_cells is not None:
            mesh.cell_data["vtkGhostType"] = ghost_cells

    return cells


def extract_cells_by_dimension(
    mesh: pv.DataSet,
    ndim: Optional[int] = None,
    method: Literal["lower", "upper"] = "upper",
    keep_empty_cells: bool = False,
) -> pv.UnstructuredGrid:
    """
    Extract cells by a specified dimension.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to extract cells from.
    ndim : int, optional
        Dimension to be used for extraction. If None, the dimension of *mesh* is used.
    method : {'lower', 'upper'}, default 'upper'
        Set the extraction method. 'lower' will extract cells of dimension lower than
        *ndim*. 'upper' will extract cells of dimension larger than *ndim*.
    keep_empty_cells : bool, default False
        If True, keep empty cells in the output mesh.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh with extracted cells.

    """
    from ._properties import _dimension_map
    from .. import get_dimension

    mesh = mesh.cast_to_unstructured_grid()
    ndim = ndim if ndim is not None else get_dimension(mesh)

    if method == "upper":
        mask = _dimension_map[mesh.celltypes] >= ndim

    elif method == "lower":
        mask = _dimension_map[mesh.celltypes] <= ndim

    else:
        raise ValueError(f"invalid method '{method}' (expected 'lower' or 'upper')")

    if keep_empty_cells:
        mask |= mesh.celltypes == pv.CellType.EMPTY_CELL

    if not mask.all():
        mesh = extract_cells(mesh, mask)

    return mesh


def fuse_cells(
    mesh: pv.DataSet, ind: ArrayLike | Sequence[ArrayLike]
) -> pv.UnstructuredGrid:
    """
    Fuse connected cells into a single cell.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to fuse cells from.
    ind : ArrayLike | Sequence[ArrayLike]
        Indices or sequence of indices of cells to fuse.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh with fused cells.

    """
    from .. import extract_boundary_polygons, get_cell_connectivity, get_dimension

    indices = [ind] if np.ndim(ind[0]) == 0 else ind
    mesh = mesh.cast_to_unstructured_grid()
    connectivity = list(get_cell_connectivity(mesh))
    celltypes = mesh.celltypes.copy()
    mask = np.ones(mesh.n_cells, dtype=bool)

    for ind in indices:
        ind = np.asanyarray(ind)
        ind = np.flatnonzero(ind) if ind.dtype.kind == "b" else ind
        mesh_ = extract_cells(mesh, ind)
        mask[ind[1:]] = False

        if get_dimension(mesh_) == 2:
            poly = extract_boundary_polygons(mesh_, fill=False)

            if len(poly) > 1:
                raise ValueError("could not fuse not fully connected cells together")

            # Find original point IDs of polygon
            # Select first instance found for each point
            cell = poly[0].cast_to_unstructured_grid()
            ids = np.array(
                [
                    np.flatnonzero(mask)[0]
                    for mask in (
                        cell.points[:, None] == mesh.points.astype(cell.points.dtype)
                    ).all(axis=-1)
                ]
            )
            mesh_points = mesh.points[ids]
            sorted_ids = ids[
                np.ravel(
                    [
                        np.flatnonzero(
                            (mesh_points.astype(point.dtype) == point).all(axis=1)
                        )
                        for point in cell.points
                    ]
                )
            ]

            # Update connectivity and cell type
            connectivity[ind[0]] = sorted_ids
            celltypes[ind[0]] = pv.CellType.POLYGON

            for cell_id in ind[1:]:
                connectivity[cell_id] = []
                celltypes[cell_id] = pv.CellType.EMPTY_CELL

            # Generate new mesh with fused cells
            cells = [item for cell in connectivity for item in [len(cell), *cell]]

        else:
            raise NotImplementedError("could not fuse cells for non 2D mesh")

    fused_mesh = pv.UnstructuredGrid(cells, celltypes, mesh.points)
    fused_mesh.point_data.update(mesh.point_data)
    fused_mesh.cell_data.update(mesh.cell_data)
    fused_mesh.user_dict.update(mesh.user_dict)

    # Tidy up
    fused_mesh = extract_cells(fused_mesh, mask).clean()

    return fused_mesh


def intersect_polyline(
    mesh: pv.DataSet,
    line: pv.PolyData,
    min_length: float = 1.0e-4,
    tolerance: float = 1.0e-8,
    pass_cell_data: bool = True,
    ignore_points_before_entry: bool = False,
    ignore_points_after_exit: bool = False,
) -> pv.PolyData:
    """
    Intersect a polyline with a mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to intersect with.
    line : pyvista.PolyData
        Polyline to intersect with the mesh.
    min_length : scalar, default 1.0e-4
        Set the minimum length of a line.
    tolerance : float, default 1.0e-8
        The absolute tolerance to use to find cells along the line.
    pass_cell_data : bool, default True
        If True, pass cell data from the line to the output.
    ignore_points_before_entry : bool, default False
        If True, ignore points before the first entry point into the mesh.
    ignore_points_after_exit : bool, default False
        If True, ignore points after the last exit point from the mesh.

    Returns
    -------
    pyvista.PolyData
        Polydata containing the intersection points and cell IDs.

    """
    from .. import get_cell_centers

    lines = split_lines(line.strip(), as_lines=True)[0]

    line_ids, cell_ids, cell = [], [], None
    mesh_entered, mesh_exited = False, False
    points = [lines.points[0]]
    count = 0

    def add_point(point: ArrayLike, line_id: int, cell_id: int) -> None:
        """Add a point to the intersection results."""
        if not np.allclose(points[-1], point, atol=tolerance):
            if (
                not mesh_entered
                or mesh_exited
                or np.linalg.norm(point - points[-1]) >= min_length
            ):
                points.append(point)
                line_ids.append(line_id)
                cell_ids.append(cell_id)

    cell_geometry = extract_cell_geometry(mesh, remove_ghost_cells=True)
    tree = KDTree(get_cell_centers(cell_geometry))

    for lid, (pointa, pointb) in enumerate(zip(lines.points[:-1], lines.points[1:])):
        # Find the first cell intersected by the line
        if cell is None:
            ids = mesh.find_cells_intersecting_line(pointa, pointb, tolerance=tolerance)

            if ids.size == 0:
                add_point(pointb, lid, -1)
                continue

            ids = np.sort(ids)
            id_ = np.linalg.norm(
                get_cell_centers(extract_cells(mesh, ids)) - pointa,
                axis=-1,
            ).argmin()
            cid = ids[id_]
            cell = extract_cells(mesh, cid)

        if not mesh_exited:
            while True:
                # Find cell edges or faces intersected by the line
                intersections = ray_cast(
                    cell,
                    pointa,
                    pointb,
                    tolerance=tolerance,
                )

                # Find the exit, if any
                # Line is fully contained in the cell
                if intersections is None:
                    break

                # Either an entrance or an exit
                elif intersections.n_cells == 1:
                    # It's an entrance if intersection point matches previous point
                    if not mesh_entered or np.allclose(
                        points[-1],
                        intersections.cell_data["IntersectionPoints"][0],
                        atol=tolerance,
                    ):
                        if not mesh_entered:
                            count = len(points)
                            mesh_entered = True
                            add_point(
                                intersections.cell_data["IntersectionPoints"][0],
                                lid,
                                -1,
                            )

                        else:
                            add_point(
                                intersections.cell_data["IntersectionPoints"][0],
                                lid,
                                cid,
                            )
                            break

                    fid = 0

                # The exit is the farthest from last point
                elif intersections.n_cells == 2:
                    dist = np.linalg.norm(
                        intersections.cell_data["IntersectionPoints"] - points[-1],
                        axis=-1,
                    )

                    if not mesh_entered:
                        count = len(points)
                        add_point(
                            intersections.cell_data["IntersectionPoints"][
                                dist.argmin()
                            ],
                            lid,
                            -1,
                        )
                        mesh_entered = True

                    fid = dist.argmax()

                # The line is hitting an edge or a corner
                else:
                    raise ValueError("could not find the exit face")

                add_point(intersections.cell_data["IntersectionPoints"][fid], lid, cid)
                exit_face = intersections.get_cell(fid)

                # Check distance to last point
                if (
                    np.linalg.norm(
                        intersections.cell_data["IntersectionPoints"][fid]
                        - lines.points[-1]
                    )
                    < min_length
                ):
                    break

                # Determine the exit cell
                _, id_ = tree.query(exit_face.center)
                cells = cell_geometry.cell_data["vtkOriginalCellIds"][id_]

                if cid not in cells:
                    raise ValueError("could not determine exit cell")

                if -1 in cells:
                    if not ignore_points_after_exit:
                        add_point(pointb, lid, -1)

                    mesh_exited = True
                    break

                cid = [id_ for id_ in cells if id_ != cid][0]
                cell = extract_cells(mesh, cid)

        else:
            if ignore_points_after_exit:
                break

            add_point(pointb, lid, -1)

    if ignore_points_before_entry:
        points = points[count:]
        line_ids = line_ids[count:]
        cell_ids = cell_ids[count:]

    polyline = split_lines(pv.MultipleLines(points))[0].compute_cell_sizes(
        length=True, area=False, volume=False
    )
    polyline.cell_data["vtkOriginalCellIds"] = line_ids
    polyline.cell_data["IntersectedCellIds"] = cell_ids

    if pass_cell_data:
        for k, v in line.cell_data.items():
            if k in polyline.cell_data:
                continue

            polyline.cell_data[k] = v[line_ids]

    return polyline


def merge(
    dataset: Sequence[pv.StructuredGrid | pv.UnstructuredGrid],
    axis: Optional[int] = None,
    merge_points: bool = True,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Merge several meshes.

    Parameters
    ----------
    dataset : Sequence[pyvista.StructuredGrid | pyvista.UnstructuredGrid]
        Meshes to merge together. At least two meshes are required.
    axis : int, optional
        The axis along which two structured grids are merged (if all meshes in *dataset*
        are structured grids).
    merge_points : bool, default True
        If True, merge equivalent points for two unstructured grids.

    Returns
    -------
    pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Merged mesh.

    """
    if len(dataset) < 2:
        return dataset[0]

    if all(isinstance(mesh, pv.StructuredGrid) for mesh in dataset):
        if axis is None:
            raise ValueError("could not merge structured grids with None axis")

        mesh_a = dataset[0]

        for mesh_b in dataset[1:]:
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

            mesh_a = mesh

    else:
        mesh = pv.merge(dataset, merge_points=merge_points)

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
    # Find common point and cell data keys
    point_data_keys = set(lines[0].point_data)
    cell_data_keys = set(lines[0].cell_data)

    for lines_ in lines[1:]:
        point_data_keys = point_data_keys.intersection(lines_.point_data)
        cell_data_keys = cell_data_keys.intersection(lines_.cell_data)

    point_data = {k: [] for k in point_data_keys}
    cell_data = {k: [] for k in cell_data_keys}

    # Loop over lines
    points, cells, offset = [], [], 0

    for lines_ in lines:
        for line in split_lines(lines_, as_lines=False):
            points.append(line.points)
            ids = np.arange(line.n_points) + offset
            cells += (
                np.insert(np.column_stack((ids[:-1], ids[1:])), 0, 2, axis=-1)
                .ravel()
                .tolist()
                if as_lines
                else [line.n_points, *ids]
            )
            offset += line.n_points

            for k, v in point_data.items():
                v.append(line.point_data[k])

            for k, v in cell_data.items():
                v.append(
                    (
                        np.full(ids.size - 1, line.cell_data[k])
                        if np.ndim(line.cell_data[k]) == 0
                        or line.cell_data[k].dtype.kind == "U"
                        else np.tile(line.cell_data[k], (ids.size - 1, 1))
                    )
                    if as_lines
                    else line.cell_data[k]
                )

    mesh = pv.PolyData(np.concatenate(points), lines=cells)

    for k, v in point_data.items():
        mesh.point_data[k] = np.concatenate(v) if v[0].ndim == 1 else np.vstack(v)

    for k, v in cell_data.items():
        mesh.cell_data[k] = np.concatenate(v) if v[0].ndim == 1 else np.vstack(v)

    return mesh.merge_points()


def offset_polygon(
    mesh_or_points: pv.PolyData | ArrayLike,
    distance: float,
) -> pv.PolyData:
    """
    Offset a polygon by a specified distance.

    Parameters
    ----------
    mesh_or_points : pyvista.PolyData | ArrayLike
        Polygon to offset.
    distance : scalar
        Distance to offset the polygon.

    Returns
    -------
    pyvista.PolyData
        Offset polygon.

    """
    if not isinstance(mesh_or_points, pv.PolyData):
        n_points = len(mesh_or_points)
        mesh = pv.PolyData(mesh_or_points, faces=[n_points, *np.arange(n_points)])

    else:
        mesh = mesh_or_points

    if not mesh.n_faces_strict:
        raise ValueError("could not offset polygon with zero polygon")

    if distance > 0.0:
        fac = 1.0

    elif distance < 0.0:
        fac = -1.0
        distance *= -1.0

    else:
        return mesh.copy()

    # Loop over faces
    faces = []
    points_ = []

    for face in mesh.irregular_faces:
        # Extract points
        points = mesh.points[face]
        mask = np.ptp(points, axis=0) == 0.0

        if mask.sum() != 1:
            raise ValueError("could not offset non-planar polygon")

        else:
            axis = np.flatnonzero(mask)[0]

        # Simple polygon offset algorithm
        # Vectorized version of C# code
        # <https://stackoverflow.com/a/73061541/9729313>
        points = points[:, ~mask]
        points = np.vstack(
            (
                points[-1],
                points,
                points[0],
            ),
        )

        x, y = points.T
        signed_area = (x[:-1] * y[1:] - x[1:] * y[:-1]).sum()

        vn = points[2:] - points[1:-1]
        vn /= np.linalg.norm(vn, axis=1)[:, np.newaxis]

        vp = points[1:-1] - points[:-2]
        vp /= np.linalg.norm(vp, axis=1)[:, np.newaxis]

        vb = (vn + vp) * fac * np.sign(signed_area)
        vb[:, 0] *= -1.0
        vb /= np.linalg.norm(vb, axis=1)[:, np.newaxis]

        dist = distance / (0.5 * (1.0 + (vn * vp).sum(axis=1))) ** 0.5
        points = points[1:-1] + dist[:, np.newaxis] * vb[:, [1, 0]]

        faces += [len(points), *(np.arange(len(points)) + len(points_))]
        points_ += np.insert(points, axis, 0.0, axis=1).tolist()

    return pv.PolyData(points_, faces=faces)


def ray_cast(
    mesh: pv.DataSet,
    pointa: ArrayLike,
    pointb: ArrayLike,
    tolerance: float = 1.0e-8,
    max_angle: Optional[float] = None,
) -> pv.PolyData | None:
    """
    Perform a single ray casting calculation.

    Parameters
    ----------
    mesh : pyvista.DataSet
        The input mesh to perform ray casting on.
    pointa : ArrayLike
        Length 3 coordinate of the start of the ray.
    pointb : ArrayLike
        Length 3 coordinate of the end of the ray.
    tolerance : float, default 1.0e-8
        The absolute tolerance to use to find cells along the ray.
    max_angle : float, optional
        The maximum angle between face normals and the ray direction. Ignored if *mesh* is 2-dimensional.

    Returns
    -------
    pyvista.PolyData | None
        A polydata with the intersected cells and points. None if no intersections.

    """
    from .. import get_cell_centers

    if isinstance(mesh, pv.PolyData):
        if mesh.n_faces_strict and mesh.n_lines:
            raise ValueError(
                "could not ray cast on a polydata with both faces and lines"
            )

        ids_ = None

    else:
        mesh = extract_cell_geometry(mesh, remove_ghost_cells=False)
        ids_ = mesh.cell_data["vtkOriginalCellIds"]

    pointa = np.asanyarray(pointa)
    pointb = np.asanyarray(pointb)

    # Find cells intersected by the line
    ids = mesh.find_cells_intersecting_line(pointa, pointb, tolerance=tolerance)

    if ids.size == 0:
        return None

    ids = np.sort(ids)
    dvec = (pointa - pointb) / np.linalg.norm(pointa - pointb)

    # Filter faces based on angle with line direction
    if mesh.n_faces_strict:
        max_angle = max_angle if max_angle is not None else 90.0 - tolerance
        normals = mesh.compute_normals(
            cell_normals=True, point_normals=False
        ).cell_data["Normals"]
        angles = np.rad2deg(np.arccos(np.abs(normals[ids] @ dvec)))
        ids = ids[angles < max_angle]

        if ids.size == 0:
            return None

    # Calculate intersection points
    cells = extract_cells(mesh, ids).extract_geometry()

    if mesh.n_faces_strict:
        centers = get_cell_centers(cells)
        intersection = pointa + dvec * np.expand_dims(
            ((centers - pointa) * normals[ids]).sum(axis=1)
            / (dvec * normals[ids]).sum(axis=1),
            axis=1,
        )

    else:
        points = np.array([edge.points for edge in split_lines(cells, as_lines=True)])
        vecs = np.diff(points, axis=1).squeeze()
        vecs = np.atleast_2d(vecs)
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
        cross = np.cross(vecs, dvec)
        denom = np.linalg.norm(cross, axis=1)
        t = (np.cross((points[:, 0] - pointa), dvec) * cross).sum(axis=1) / denom**2
        intersection = points[:, 0] - t[:, None] * vecs

    # Add data
    cells.clear_data()
    cells.cell_data["IntersectionPoints"] = intersection
    cells.cell_data["vtkOriginalCellIds"] = ids_[ids] if ids_ is not None else ids

    return cells


def reconstruct_line(
    mesh_or_points: pv.DataSet | ArrayLike,
    start: int = 0,
    close: bool = False,
    tolerance: float = 1.0e-8,
) -> pv.PolyData:
    """
    Reconstruct a line from the points in this dataset.

    Parameters
    ----------
    mesh_or_points : pyvista.DataSet | ArrayLike
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
    if isinstance(mesh_or_points, pv.PolyData):
        points = mesh_or_points.points

    else:
        points = np.asanyarray(mesh_or_points)

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
    mapping: dict[str | int, int],
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

    mesh = mesh.extract_cells_by_type((pv.CellType.LINE, pv.CellType.POLY_LINE))
    lines = []

    for i, line_ids in enumerate(_get_irregular_cells(mesh.GetLines())):
        line = pv.PolyData(
            mesh.points[line_ids],
            lines=(
                np.insert(
                    np.column_stack(
                        (np.arange(line_ids.size - 1), np.arange(1, line_ids.size))
                    ),
                    0,
                    2,
                    axis=-1,
                ).ravel()
                if as_lines
                else [line_ids.size, *np.arange(line_ids.size)]
            ),
        )

        for k, v in mesh.point_data.items():
            line.point_data[k] = v[line_ids]

        for k, v in mesh.cell_data.items():
            line.cell_data[k] = (
                np.full(line.n_cells, v[i])
                if np.ndim(v[i]) == 0
                else np.tile(v[i], (line.n_cells, 1)).copy()
            )

        lines.append(line)

    return lines


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
    points = np.vstack((mesh.points, quad_points))

    return pv.UnstructuredGrid(cells, celltypes, points)


_celltype_to_faces = {
    pv.CellType.TETRA: {
        "TRIANGLE": np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]),
    },
    pv.CellType.PYRAMID: {
        "QUAD": np.array([[0, 3, 2, 1]]),
        "TRIANGLE": np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
    },
    pv.CellType.WEDGE: {
        "TRIANGLE": np.array([[0, 2, 1], [3, 4, 5]]),
        "QUAD": np.array([[0, 1, 4, 3], [1, 2, 5, 4], [0, 3, 5, 2]]),
    },
    pv.CellType.HEXAHEDRON: {
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
    pv.CellType.VOXEL: {
        "QUAD": np.array(
            [
                [0, 2, 3, 1],
                [4, 5, 7, 6],
                [0, 1, 5, 4],
                [1, 3, 7, 5],
                [3, 2, 6, 7],
                [0, 4, 6, 2],
            ]
        ),
    },
}
