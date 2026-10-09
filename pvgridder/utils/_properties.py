from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import pyvista as pv


if TYPE_CHECKING:
    from typing import Literal, Optional

    from numpy.typing import NDArray

    from .._typing import DataSetLike


@overload
def get_cell_connectivity(
    mesh: DataSetLike,
    flatten: Literal[False],
) -> tuple[NDArray | list[NDArray], ...]: ...


@overload
def get_cell_connectivity(
    mesh: DataSetLike,
    flatten: Literal[True],
) -> NDArray: ...


@overload
def get_cell_connectivity(
    mesh: DataSetLike,
) -> tuple[NDArray | list[NDArray], ...]: ...


def get_cell_connectivity(
    mesh: DataSetLike,
    flatten: bool = False,
) -> NDArray | tuple[NDArray | list[NDArray], ...]:
    """
    Get the cell connectivity of a mesh.

    Parameters
    ----------
    mesh : DataSetLike
        Input mesh.
    flatten : bool, default False
        If True, flatten the cell connectivity array (e.g., as input of
        :class:`pyvista.UnstructuredGrid`).

    Returns
    -------
    NDArray | tuple[NDArray | list[NDArray], ...]
        Cell connectivity.

    """
    from itertools import chain

    from pyvista.core.cell import _get_irregular_cells

    mesh = mesh.cast_to_unstructured_grid()

    # Generate cells
    cells: list[NDArray | list[NDArray]] = list(_get_irregular_cells(mesh.GetCells()))

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
    mesh: DataSetLike,
    polyhedron_method: Optional[Literal["box", "geometric", "tetra"]] = None,
) -> NDArray:
    """
    Get the cell centers of a mesh.

    Parameters
    ----------
    mesh : DataSetLike
        Input mesh.
    polyhedron_method : {'box', 'geometric', 'tetra'} | None, optional
        Calculation method for centers of polyhedral cells:

         - 'box': bounding box (as VTK < 9.5.0)
         - 'geometric': geometric average of points (as VTK 9.5.0)
         - 'tetra': weighted tetrahedral decomposition (as > VTK 9.5.0)

        If None, use current VTK's implementation.

    Returns
    -------
    NDArray
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
            offset = mesh.cell_offsets
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
                all_v0, all_v1, all_v2, all_apex, cell_ids = [], [], [], [], []

                for cell_id, (locations, i0, mask_) in enumerate(
                    zip(polyhedron_face_locations, offset[:-1], mask)
                ):
                    if not mask_:
                        continue

                    apex_idx = connectivity[i0]

                    for loc in locations:
                        face = polyhedron_faces[loc]
                        f0 = face[0]

                        for v1, v2 in zip(face[1:-1], face[2:]):
                            all_v0.append(f0)
                            all_v1.append(v1)
                            all_v2.append(v2)
                            all_apex.append(apex_idx)
                            cell_ids.append(cell_id)

                cell_ids = np.array(cell_ids)

                # Extract 3D points in unified NumPy arrays
                v0 = mesh.points[all_v0]
                v1 = mesh.points[all_v1]
                v2 = mesh.points[all_v2]
                apex = mesh.points[all_apex]

                # Compute all tetrahedral volumes and centroids
                v0_a, v1_a, v2_a = v0 - apex, v1 - apex, v2 - apex
                volumes = np.einsum("ij,ij->i", np.cross(v0_a, v1_a), v2_a) / 6.0
                tetra_centroids = (v0 + v1 + v2 + apex) / 4.0

                # Aggregate globally
                n_cells = len(mask)
                total_volumes = np.zeros(n_cells)
                weighted_centers = np.zeros((n_cells, 3))

                np.add.at(total_volumes, cell_ids, volumes)
                np.add.at(weighted_centers, cell_ids, tetra_centroids * volumes[:, None])

                # Compute centroids
                mask = total_volumes != 0.0
                centers[mask] = weighted_centers[mask] / total_volumes[mask, None]

    if ghost_cells is not None:
        mesh.cell_data["vtkGhostType"] = ghost_cells

    return centers


def get_cell_group(mesh: DataSetLike, key: str = "CellGroup") -> NDArray | None:
    """
    Get the cell group of a mesh.

    Parameters
    ----------
    mesh : DataSetLike
        Input mesh.
    key : str, default "CellGroup"
        Key to use to get the cell group.

    Returns
    -------
    NDArray | None
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


def get_dimension(mesh: DataSetLike) -> int:
    """
    Get the dimension of a mesh.

    Parameters
    ----------
    mesh : DataSetLike
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
