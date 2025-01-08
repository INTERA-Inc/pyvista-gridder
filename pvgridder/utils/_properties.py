from __future__ import annotations

import numpy as np
import pyvista as pv


def get_dimension(
    mesh: pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid,
) -> int:
    """
    Get the dimension of a mesh.

    Parameters
    ----------
    mesh : pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid
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
