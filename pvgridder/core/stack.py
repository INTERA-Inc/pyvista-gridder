from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pyvista as pv

from ._base import MeshStackBase
from ._helpers import (
    generate_surface_from_two_lines,
    generate_volume_from_two_surfaces,
)


if TYPE_CHECKING:
    from collections.abc import Sequence  # pragma: no cover
    from typing import Optional  # pragma: no cover

    from numpy.typing import ArrayLike, NDArray  # pragma: no cover
    from typing_extensions import Self  # pragma: no cover


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
    bottom_up : bool, default True
        If True, assume items are stacked from bottom to top.
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
        bottom_up: bool = True,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new 2D mesh stack."""
        from .. import split_lines

        if isinstance(mesh, (list, tuple, np.ndarray)) and np.asarray(mesh).ndim == 1:
            points = np.zeros((len(mesh), 3))
            points[:, (axis + 1) % 3] = mesh
            lines = pv.lines_from_points(points)

        elif not isinstance(mesh, pv.PolyData) or (
            isinstance(mesh, pv.PolyData) and not mesh.n_lines
        ):
            raise ValueError("invalid mesh, input mesh should be a line or a polyline")

        else:
            lines = split_lines(mesh)[0]

        super().__init__(lines, axis, bottom_up, default_group, ignore_groups)

    def generate_mesh(self) -> pv.StructuredGrid:
        """
        Generate mesh by stacking all items.

        Returns
        -------
        pyvista.StructuredGrid
            Stacked mesh.

        """
        plane = "yx" if self.axis == 0 else "xy" if self.axis == 1 else "xz"
        extrude = (
            lambda mesh_a, mesh_b, resolution, method: generate_surface_from_two_lines(
                mesh_a, mesh_b, plane, resolution, method
            )
        )

        return cast(pv.StructuredGrid, self._generate_mesh(extrude))

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
    bottom_up : bool, default True
        If True, assume items are stacked from bottom to top.
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
        bottom_up: bool = True,
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

        super().__init__(mesh, axis, bottom_up, default_group, ignore_groups)

    def add_plane(
        self,
        angles: ArrayLike,
        point: ArrayLike,
        *args,
        **kwargs,
    ) -> Self:
        """
        Add a plane to the stack.

        Parameters
        ----------
        angles : ArrayLike
            Rotation angles in degrees around the X, Y, and Z axes.
        point : ArrayLike
            Coordinates of one point on the plane.
        *args, **kwargs
            Additional arguments. See ``pvgridder.MeshStack3D.add`` for details.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from scipy.spatial.transform import Rotation

        angles = np.asanyarray(angles)
        point = np.asanyarray(point)

        angles[self.axis] = 0.0
        rot = Rotation.from_rotvec(angles, degrees=True)
        idx = np.delete(np.arange(3), self.axis)

        def func(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> NDArray:
            points = np.column_stack((x, y, z))
            points[:, idx] -= point[idx]
            points[:, self.axis] = 0.0

            return rot.apply(points, inverse=True)[:, self.axis] + point[self.axis]

        return self.add(func, *args, **kwargs)

    def generate_mesh(
        self,
        tolerance: float = 1.0e-8,
        invert_polyhedron_faces: bool = True,
    ) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """
        Generate mesh by stacking all items.

        Parameters
        ----------
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points (for unstructured grids).
        invert_polyhedron_faces : bool, default True
            If True, ensure that normal vectors of polyhedron faces point outwards by
            inverting the faces of polyhedral cells if necessary.

        Returns
        -------
        pyvista.StructuredGrid | pyvista.UnstructuredGrid
            Stacked mesh.

        """
        extrude = (
            lambda mesh_a,
            mesh_b,
            resolution,
            method: generate_volume_from_two_surfaces(
                mesh_a,
                mesh_b,
                resolution,
                method,
                invert_polyhedron_faces=invert_polyhedron_faces,
            )
        )

        return self._generate_mesh(extrude, tolerance)

    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Extrude a line."""
        return generate_volume_from_two_surfaces(*args, **kwargs)

    def _transition(self, *args) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        raise NotImplementedError()
