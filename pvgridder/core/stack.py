from __future__ import annotations
from typing import Optional

import numpy as np
import pyvista as pv

from ._base import MeshStackBase
from ._helpers import (
    generate_plane_surface_from_two_lines,
    generate_volume_from_two_surfaces,
    is2d,
)


class MeshStack2D(MeshStackBase):
    def __init__(
        self,
        mesh: pv.PolyData,
        axis: int = 2,
    ) -> None:
        if not isinstance(mesh, pv.PolyData) and not mesh.n_lines:
            raise ValueError("invalid mesh, input mesh should be a line or a polyline")

        lines = mesh.lines
        n_cells = lines[0] - 1
        points = mesh.points[lines[1 : lines[0] + 1]]
        cells = np.column_stack(
            (
                np.full(n_cells, 2),
                np.arange(n_cells),
                np.roll(np.arange(n_cells), -1),
            )
        ).ravel()
        super().__init__(pv.PolyData(points, lines=lines), axis)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid:
        return generate_plane_surface_from_two_lines(*args, **kwargs)


class MeshStack3D(MeshStackBase):
    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
    ) -> None:
        if isinstance(mesh, (pv.StructuredGrid, pv.UnstructuredGrid)):
            if not is2d(mesh):
                raise ValueError("invalid mesh, input mesh should be 2D")

        else:
            raise ValueError("invalid mesh, input mesh should be a 2D structured grid or an unstructured grid")

        super().__init__(mesh, axis)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return generate_volume_from_two_surfaces(*args, **kwargs)

