from __future__ import annotations

import numpy as np
import pyvista as pv


def get_dimension(mesh: pv.ExplicitStructuredGrid | pv.StructuredGrid | pv.UnstructuredGrid) -> int:
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
        -1,  0,  0,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  1,  3,  3,  3,
        -1, -1, -1, -1,  1,  2,  2,  2,  2,  1,  2,  2,  2,  2,  1,  1,  2,
        2,  1,  2,  2, -1, -1, -1,  0,  3, -1, -1, -1, -1, -1, -1, -1, -1,
        1,  2,  2,  2,  3,  3, -1, -1, -1,  1,  2,  2,  2,  3,  1,  3,  3,
        1,  2,  2,  3,  3,  1,  3,  1,  2,  2,  3,  3,  1,  3,
    ]
)
