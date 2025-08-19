from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike


def get_cell_connectivity(
    mesh: pv.DataSet,
    flatten: bool = False,
) -> Sequence[Sequence[int | Sequence[int]]] | Sequence[int]:
    """
    Get the cell connectivity of a mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    flatten : bool, default False
        If True, flatten the cell connectivity array (e.g., as input of
        :class:`pyvista.UnstructuredGrid`).

    Returns
    -------
    Sequence[Sequence[int | Sequence[int]]] | Sequence[int]
        Cell connectivity.

    """
    from itertools import chain

    from pyvista.core.cell import _get_irregular_cells

    mesh = mesh.cast_to_unstructured_grid()

    # Generate cells
    cells = list(_get_irregular_cells(mesh.GetCells()))

    # Generate polyhedral cell faces if any
    if (mesh.celltypes == pv.CellType.POLYHEDRON).any():
        faces = _get_irregular_cells(mesh.GetPolyhedronFaces())
        locations = _get_irregular_cells(mesh.GetPolyhedronFaceLocations())

        for cid, location in enumerate(locations):
            if location.size == 0:
                continue

            cells[cid] = [faces[face] for face in location]

    if flatten:
        cells_ = []

        for cell, celltype in zip(cells, mesh.celltypes):
            if celltype == pv.CellType.POLYHEDRON:
                cell = [len(cell), *chain.from_iterable([[len(c), *c] for c in cell])]

            cells_ += [len(cell), *cell]

        return np.array(cells_)

    else:
        return tuple(cells)


def get_cell_centers(
    mesh: pv.DataSet,
    polyhedron_method: Optional[Literal["box", "geometric", "tetra"]] = None,
) -> ArrayLike:
    """
    Get the cell centers of a mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    polyhedron_method : {'box', 'geometric', 'tetra'} | None, optional
        Calculation method for centers of polyhedral cells:

         - 'box': bounding box (as VTK < 9.5.0)
         - 'geometric': geometric average of points (as VTK 9.5.0)
         - 'tetra': weighted tetrahedral decomposition (as > VTK 9.5.0)

        If None, use current VTK's implementation.

    Returns
    -------
    ArrayLike
        Cell centers.

    """
    from pyvista.core.cell import _get_irregular_cells
    from vtk import __version__ as vtk_version

    centers = np.full((mesh.n_cells, 3), np.nan)
    ghost_cells = (
        mesh.cell_data.pop("vtkGhostType") if "vtkGhostType" in mesh.cell_data else None
    )

    if not isinstance(mesh, pv.UnstructuredGrid):
        centers[:] = mesh.cell_centers().points

    else:
        celltypes = mesh.celltypes
        centers[celltypes != pv.CellType.EMPTY_CELL] = mesh.cell_centers().points

        # Polyhedral cells' centers
        mask = celltypes == pv.CellType.POLYHEDRON

        if polyhedron_method is not None and mask.any():
            offset = mesh.offset
            connectivity = mesh.cell_connectivity

            if polyhedron_method == "box" and vtk_version >= "9.5.0":
                polyhedron_points = [
                    mesh.points[connectivity[i1:i2]]
                    for i1, i2, mask_ in zip(offset[:-1], offset[1:], mask)
                    if mask_
                ]
                centers[mask] = 0.5 * np.array(
                    [
                        points.min(axis=0) + points.max(axis=0)
                        for points in polyhedron_points
                    ]
                )

            elif polyhedron_method == "geometric" and vtk_version != "9.5.0":
                centers[mask] = np.array(
                    [
                        mesh.points[connectivity[i1:i2]].mean(axis=0)
                        for i1, i2, mask_ in zip(offset[:-1], offset[1:], mask)
                        if mask_
                    ]
                )

            elif polyhedron_method == "tetra" and vtk_version <= "9.5.0":
                polyhedron_faces = _get_irregular_cells(mesh.GetPolyhedronFaces())
                polyhedron_face_locations = _get_irregular_cells(
                    mesh.GetPolyhedronFaceLocations()
                )

                for center, locations, i0, mask_ in zip(
                    centers,
                    polyhedron_face_locations,
                    offset[:-1],
                    mask,
                ):
                    if not mask_:
                        continue

                    # Triangulate polyhedron's faces
                    triangles = np.array(
                        [
                            (face[0], v1, v2)
                            for face in [polyhedron_faces[loc] for loc in locations]
                            for v1, v2 in zip(face[1:], face[2:])
                        ]
                    )

                    # Use polyhedron's first vertex as apex
                    apex = mesh.points[connectivity[i0]]
                    apex = np.broadcast_to(apex, (len(triangles), 3))

                    # Tetrahedralize polyhedron
                    v0 = mesh.points[triangles[:, 0]]
                    v1 = mesh.points[triangles[:, 1]]
                    v2 = mesh.points[triangles[:, 2]]
                    tetras = np.stack((v0, v1, v2, apex), axis=1)

                    # Compute tetrahedral volumes
                    volumes = (
                        np.einsum("ij,ij->i", np.cross(v0 - apex, v1 - apex), v2 - apex)
                        / 6.0
                    )

                    # Compute centroid
                    center[:] = np.average(tetras.mean(axis=1), axis=0, weights=volumes)

    if ghost_cells is not None:
        mesh.cell_data["vtkGhostType"] = ghost_cells

    return centers


def get_cell_group(mesh: pv.DataSet, key: str = "CellGroup") -> ArrayLike | None:
    """
    Get the cell group of a mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    key : str, default "CellGroup"
        Key to use to get the cell group.

    Returns
    -------
    ArrayLike | None
        Cell group.

    """
    if key in mesh.cell_data:
        cell_groups = mesh.cell_data[key]

        if key in mesh.user_dict:
            groups = {v: k for k, v in mesh.user_dict[key].items()}
            cell_groups = [groups[i] for i in cell_groups]

        return np.array(cell_groups)

    else:
        return None


def get_dimension(mesh: pv.DataSet) -> int:
    """
    Get the dimension of a mesh.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.

    Returns
    -------
    int
        Dimension of the mesh.

    """
    if isinstance(
        mesh,
        (
            pv.ExplicitStructuredGrid,
            pv.ImageData,
            pv.RectilinearGrid,
            pv.StructuredGrid,
        ),
    ):
        return 3 - sum(n == 1 for n in mesh.dimensions)

    elif isinstance(mesh, (pv.PolyData, pv.UnstructuredGrid)):
        mesh = mesh.cast_to_unstructured_grid()

        return _dimension_map[mesh.celltypes].max()

    else:
        raise TypeError(f"could not get dimension of mesh of type '{type(mesh)}'")


_dimension_map = np.array(
    [
        -1,  # EMPTY_CELL
        0,  # VERTEX
        0,  # POLY_VERTEX
        1,  # LINE
        1,  # POLY_LINE
        2,  # TRIANGLE
        2,  # TRIANGLE_STRIP
        2,  # POLYGON
        2,  # PIXEL
        2,  # QUAD
        3,  # TETRA
        3,  # VOXEL
        3,  # HEXAHEDRON
        3,  # WEDGE
        3,  # PYRAMID
        3,  # PENTAGONAL_PRISM
        3,  # HEXAGONAL_PRISM
        -1,
        -1,
        -1,
        -1,
        1,  # QUADRATIC_EDGE
        2,  # QUADRATIC_TRIANGLE
        2,  # QUADRATIC_QUAD
        3,  # QUADRATIC_TETRA
        3,  # QUADRATIC_HEXAHEDRON
        3,  # QUADRATIC_WEDGE
        3,  # QUADRATIC_PYRAMID
        2,  # BIQUADRATIC_QUAD
        3,  # TRIQUADRATIC_HEXAHEDRON
        2,  # QUADRATIC_LINEAR_QUAD
        3,  # QUADRATIC_LINEAR_WEDGE
        3,  # BIQUADRATIC_QUADRATIC_WEDGE
        3,  # BIQUADRATIC_QUADRATIC_HEXAHEDRON
        2,  # BIQUADRATIC_TRIANGLE
        1,  # CUBIC_LINE
        2,  # QUADRATIC_POLYGON
        3,  # TRIQUADRATIC_PYRAMID
        -1,
        -1,
        -1,
        0,  # CONVEX_POINT_SET
        3,  # POLYHEDRON
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,  # PARAMETRIC_CURVE
        2,  # PARAMETRIC_SURFACE
        2,  # PARAMETRIC_TRI_SURFACE
        2,  # PARAMETRIC_QUAD_SURFACE
        3,  # PARAMETRIC_TETRA_REGION
        3,  # PARAMETRIC_HEX_REGION
        -1,
        -1,
        -1,
        1,  # HIGHER_ORDER_EDGE
        2,  # HIGHER_ORDER_TRIANGLE
        2,  # HIGHER_ORDER_QUAD
        2,  # HIGHER_ORDER_POLYGON
        3,  # HIGHER_ORDER_TETRAHEDRON
        3,  # HIGHER_ORDER_WEDGE
        3,  # HIGHER_ORDER_PYRAMID
        3,  # HIGHER_ORDER_HEXAHEDRON
        1,  # LAGRANGE_CURVE
        2,  # LAGRANGE_TRIANGLE
        2,  # LAGRANGE_QUADRILATERAL
        3,  # LAGRANGE_TETRAHEDRON
        3,  # LAGRANGE_HEXAHEDRON
        3,  # LAGRANGE_WEDGE
        3,  # LAGRANGE_PYRAMID
        1,  # BEZIER_CURVE
        2,  # BEZIER_TRIANGLE
        2,  # BEZIER_QUADRILATERAL
        3,  # BEZIER_TETRAHEDRON
        3,  # BEZIER_HEXAHEDRON
        3,  # BEZIER_WEDGE
        3,  # BEZIER_PYRAMID
    ]
)
