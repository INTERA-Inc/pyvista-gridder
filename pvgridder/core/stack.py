from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from typing_extensions import Self

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
    mesh : pyvista.PolyData | ArrayLike
        Base mesh. If ArrayLike, assume straight line depending on *axis*.

         - 0: along Z axis
         - 1: along Y axis
         - 2: along X axis

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
        mesh: pv.PolyData | ArrayLike,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new 2D mesh stack."""
        from .. import split_lines

        if isinstance(mesh, (list, tuple, np.ndarray)) and np.asarray(mesh).ndim == 1:
            points = np.zeros((len(mesh), 3))
            points[:, (axis + 1) % 3] = mesh
            lines = pv.lines_from_points(points)

        elif not isinstance(mesh, pv.PolyData) and not mesh.n_lines:
            raise ValueError("invalid mesh, input mesh should be a line or a polyline")

        else:
            lines = split_lines(mesh)[0]

        super().__init__(lines, axis, default_group, ignore_groups)

    def _extrude(self, *args) -> pv.StructuredGrid:
        """Extrude a line."""
        line_a, line_b, resolution, method = args
        plane = "yx" if self.axis == 0 else "xy" if self.axis == 1 else "xz"

        return generate_surface_from_two_lines(
            line_a, line_b, plane, resolution, method
        )

    def _transition(self, *args) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        from .. import Polygon

        mesh_a, mesh_b, groups, group = args
        points = np.vstack((mesh_a.points, mesh_b.points[::-1]))
        mesh = Polygon(points, celltype="triangle")
        mesh.cell_data["CellGroup"] = self._initialize_group_array(mesh, groups, group)

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
            mesh_or_resolution = RegularLine(
                self.mesh.points, resolution=mesh_or_resolution
            )

        self._mesh = split_lines(mesh_or_resolution)[0]
        self._transition_flag = True

        return self


class MeshStack3D(MeshStackBase):
    """
    3D mesh stack class.

    Parameters
    ----------
    mesh : pyvista.ImageData | pyvista.RectilinearGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
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
        mesh: pv.ImageData
        | pv.RectilinearGrid
        | pv.StructuredGrid
        | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        from .. import get_dimension

        if isinstance(
            mesh,
            (pv.ImageData, pv.RectilinearGrid, pv.StructuredGrid, pv.UnstructuredGrid),
        ):
            if get_dimension(mesh) != 2:
                raise ValueError("invalid mesh, input mesh should be 2D")

        else:
            raise ValueError(
                "invalid mesh, input mesh should be a 2D structured grid or an unstructured grid"
            )

        super().__init__(mesh, axis, default_group, ignore_groups)

    def add_plane(
        self,
        angles: Sequence[float, float, float],
        point: Sequence[float, float, float],
        *args,
        **kwargs,
    ) -> Self:
        """
        Add a plane to the stack.

        Parameters
        ----------
        angles : Sequence[float, float, float]
            Rotation angles in degrees around the X, Y, and Z axes.
        point : Sequence[float, float, float]
            Coordinates of one point on the plane.
        *args, **kwargs
            Additional arguments. See ``pvgridder.MeshStack3D.add`` for details.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from scipy.spatial.transform import Rotation

        angles = np.array(angles)
        point = np.asanyarray(point)

        angles[self.axis] = 0.0
        rot = Rotation.from_rotvec(angles, degrees=True)
        idx = np.delete(np.arange(3), self.axis)

        def func(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> ArrayLike:
            points = np.column_stack((x, y, z))
            points[:, idx] -= point[idx]
            points[:, self.axis] = 0.0

            return rot.apply(points, inverse=True)[:, self.axis] + point[self.axis]

        return self.add(func, *args, **kwargs)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Extrude a line."""
        return generate_volume_from_two_surfaces(*args, **kwargs)

    def _transition(self, *args) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        raise NotImplementedError()
