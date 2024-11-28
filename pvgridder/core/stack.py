from __future__ import annotations
from typing import Optional

import numpy as np
import pyvista as pv

from ._base import MeshStackBase
from ._helpers import (
    generate_surface_from_two_lines,
    generate_volume_from_two_surfaces,
    is2d,
)


class MeshStack2D(MeshStackBase):
    __name__: str = "MeshStack2D"
    __qualname__: str = "pvgridder.MeshStack2D"

    def __init__(
        self,
        mesh: pv.PolyData,
        axis: int = 2,
        group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        from .. import split_lines

        if not isinstance(mesh, pv.PolyData) and not mesh.n_lines:
            raise ValueError("invalid mesh, input mesh should be a line or a polyline")

        lines = split_lines(mesh)
        super().__init__(lines[0], axis, group, ignore_groups)

    def _extrude(self, *args) -> pv.StructuredGrid:
        line_a, line_b, resolution, method = args
        plane = "yx" if self.axis == 0 else "xy" if self.axis == 1 else "xz"

        return generate_surface_from_two_lines(line_a, line_b, plane, resolution, method)

    def _set_active(self, mesh: pv.StructuredGrid) -> None:
        areas = mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"]
        mesh.cell_data["Active"] = (np.abs(areas) > 0.0).astype(int)


class MeshStack3D(MeshStackBase):
    __name__: str = "MeshStack3D"
    __qualname__: str = "pvgridder.MeshStack3D"

    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        if isinstance(mesh, (pv.StructuredGrid, pv.UnstructuredGrid)):
            if not is2d(mesh):
                raise ValueError("invalid mesh, input mesh should be 2D")

        else:
            raise ValueError("invalid mesh, input mesh should be a 2D structured grid or an unstructured grid")

        super().__init__(mesh, axis, group, ignore_groups)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return generate_volume_from_two_surfaces(*args, **kwargs)

    def _set_active(self, mesh: pv.StructuredGrid | pv.UnstructuredGrid) -> None:
        volumes = mesh.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]
        mesh.cell_data["Active"] = (np.abs(volumes) > 0.0).astype(int)
