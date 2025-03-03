from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
from typing_extensions import Self

import numpy as np
import pyvista as pv

from ._base import MeshStackBase
from ._helpers import (
    generate_surface_from_two_lines,
    generate_volume_from_two_surfaces,
)


class MeshStack2D(MeshStackBase):
    """
    2D mesh stack class.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Base mesh.
    axis : int, default 2
        Stacking axis.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshStack2D"
    __qualname__: str = "pvgridder.MeshStack2D"

    def __init__(
        self,
        mesh: pv.PolyData,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new 2D mesh stack."""
        from .. import split_lines

        if not isinstance(mesh, pv.PolyData) and not mesh.n_lines:
            raise ValueError("invalid mesh, input mesh should be a line or a polyline")

        lines = split_lines(mesh)
        super().__init__(lines[0], axis, default_group, ignore_groups)

    def _extrude(self, *args) -> pv.StructuredGrid:
        """Extrude a line."""
        line_a, line_b, resolution, method = args
        plane = "yx" if self.axis == 0 else "xy" if self.axis == 1 else "xz"

        return generate_surface_from_two_lines(
            line_a, line_b, plane, resolution, method
        )

    def _transition(self, mesh_a: pv.PolyData, mesh_b: pv.PolyData) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        from .. import Polygon

        points = np.row_stack((mesh_a.points, mesh_b.points[::-1]))
        mesh = Polygon(points, celltype="triangle")

        return mesh

    def set_transition(self, mesh_or_resolution: pv.PolyData | int) -> Self:
        """
        Set next item as a transition item.

        Parameters
        ----------
        mesh_or_resolution : pyvista.PolyData | int
            New base mesh for subsequent items.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import RegularLine, split_lines

        if isinstance(mesh_or_resolution, int):
            mesh_or_resolution = RegularLine(self.mesh.points, resolution=mesh_or_resolution)

        self._mesh = split_lines(mesh_or_resolution)[0]
        self._transition_flag = True

        return self


class MeshStack3D(MeshStackBase):
    """
    3D mesh stack class.

    Parameters
    ----------
    mesh : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Base mesh.
    axis : int, default 2
        Stacking axis.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshStack3D"
    __qualname__: str = "pvgridder.MeshStack3D"

    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        from .. import get_dimension

        if isinstance(mesh, (pv.StructuredGrid, pv.UnstructuredGrid)):
            if get_dimension(mesh) != 2:
                raise ValueError("invalid mesh, input mesh should be 2D")

        else:
            raise ValueError(
                "invalid mesh, input mesh should be a 2D structured grid or an unstructured grid"
            )

        super().__init__(mesh, axis, default_group, ignore_groups)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Extrude a line."""
        return generate_volume_from_two_surfaces(*args, **kwargs)

    def _transition(self, *args) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        raise NotImplementedError()
