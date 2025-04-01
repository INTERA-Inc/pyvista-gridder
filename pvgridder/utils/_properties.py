from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyvista as pv


def get_cell_connectivity(
    mesh: pv.UnstructuredGrid,
    flatten: bool = False,
) -> Sequence[Sequence[int | Sequence[int]]] | Sequence[int]:
    """
    Get the original cell connectivity of an unstructured mesh.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
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

    # Generate cells
    cells = list(_get_irregular_cells(mesh.GetCells()))

    # Generate polyhedral cell faces if any
    polyhedral_cells = pv.convert_array(mesh.GetFaces())

    if polyhedral_cells is not None:
        locations = pv.convert_array(mesh.GetFaceLocations())

        for cid, location in enumerate(locations):
            if location == -1:
                continue

            n_faces = polyhedral_cells[location]
            i, cell = location + 1, []

            while len(cell) < n_faces:
                n_vertices = polyhedral_cells[i]
                cell.append(polyhedral_cells[i + 1 : i + 1 + n_vertices])
                i += n_vertices + 1

            cells[cid] = cell

    if flatten:
        cells_ = []

        for cell, celltype in zip(cells, mesh.celltypes):
            if celltype == pv.CellType.POLYHEDRON:
                cell = [len(cell), *chain.from_iterable([[len(c), *c] for c in cell])]

            cells_ += [len(cell), *cell]

        return np.array(cells_)

    else:
        return tuple(cells)


def get_dimension(
    mesh: pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid,
) -> int:
    """
    Get the dimension of a mesh.

    Parameters
    ----------
    mesh : pyvista.ExplicitStructuredGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Input mesh.

    Returns
    -------
    int
        Dimension of the mesh.

    """
    if isinstance(mesh, (pv.ExplicitStructuredGrid, pv.StructuredGrid)):
        return 3 - sum(n == 1 for n in mesh.dimensions)

    elif isinstance(mesh, pv.UnstructuredGrid):
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
