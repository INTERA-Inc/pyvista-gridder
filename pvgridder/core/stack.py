from __future__ import annotations
from typing import Optional

import numpy as np
import pyvista as pv

from ._base import MeshStackBase
from ._helpers import (
    generate_plane_surface_from_two_lines,
    generate_volume_from_two_surfaces,
)


class MeshStack2D(MeshStackBase):
    def __init__(
        self,
        mesh: pv.PolyData,
        axis: int = 2,
    ) -> None:
        super().__init__(mesh, axis)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid:
        return generate_plane_surface_from_two_lines(*args, **kwargs)


class MeshStack3D(MeshStackBase):
    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
    ) -> None:
        super().__init__(mesh, axis)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return generate_volume_from_two_surfaces(*args, **kwargs)

