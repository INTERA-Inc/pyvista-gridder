from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from typing_extensions import Self

from ._base import MeshBase, MeshItem
from ._helpers import (
    generate_volume_from_two_surfaces,
)


class MeshExtrude(MeshBase):
    """
    Mesh extrusion class.

    Parameters
    ----------
    mesh : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Base mesh.
    scale : scalar, optional
        Default scaling factor.
    angle : scalar, optional
        Default rotation angle (in degree).
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshExtrude"
    __qualname__: str = "pvgridder.MeshExtrude"

    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        scale: Optional[float] = None,
        angle: Optional[float] = None,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new mesh extrusion."""
        from .. import get_dimension

        if get_dimension(mesh) != 2:
            raise ValueError(
                "invalid mesh, input mesh should be a 2D structured grid or an unstructured grid"
            )

        super().__init__(default_group, ignore_groups, items=[MeshItem(mesh)])
        self._mesh = mesh
        self._angle = angle
        self._scale = scale

    def add(
        self,
        vector: ArrayLike,
        resolution: Optional[int | ArrayLike] = None,
        method: Optional[Literal["constant", "log", "log_r"]] = None,
        scale: Optional[float] = None,
        angle: Optional[float] = None,
        group: Optional[str | dict] = None,
    ) -> Self:
        """
        Add a new item to extrusion.

        Parameters
        ----------
        vector : ArrayLike
            Translation vector.
        resolution : int | ArrayLike, optional
            Number of subdivisions along the extrusion axis or relative position of
            subdivisions (in percentage) with respect to the previous item.
        method : {'constant', 'log', 'log_r'}, optional
            Subdivision method if *resolution* is an integer:

             - if 'constant', subdivisions are equally spaced.
             - if 'log', subdivisions are logarithmically spaced (from small to large).
             - if 'log_r', subdivisions are logarithmically spaced (from large to small).

        scale : scalar, optional
            Scaling factor applied to the previous item before extrusion.
        angle : scalar, optional
            Rotation angle (in degree) applied to the previous item before extrusion.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        vector = np.asarray(vector)

        if vector.shape != (3,):
            raise ValueError("invalid extrusion vector")

        scale = scale if scale is not None else self.scale
        angle = angle if angle is not None else self.angle

        mesh = self.items[-1].mesh.copy()
        mesh = mesh.translate(vector)

        if scale is not None:
            mesh.points = (mesh.points - mesh.center) * scale + mesh.center

        if angle is not None:
            mesh = mesh.rotate_vector(vector, angle, mesh.center)

        item = MeshItem(mesh, resolution=resolution, method=method, group=group)
        self.items.append(item)

        return self

    def generate_mesh(
        self, tolerance: float = 1.0e-8
    ) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """
        Generate mesh by extruding all items.

        Parameters
        ----------
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points (for unstructured grids).

        Returns
        -------
        pyvista.StructuredGrid | pyvista.UnstructuredGrid
            Extruded mesh.

        """
        from .. import merge

        if len(self.items) <= 1:
            raise ValueError("not enough items to extrude")

        groups = {}

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1.mesh.copy()
            tmp = self._initialize_group_array(mesh_a, groups)

            group = item2.group if item2.group else {}

            if isinstance(group, str):
                group = {group: lambda x: np.ones(x.n_cells, dtype=bool)}

            for k, v in group.items():
                tmp[v(mesh_a)] = self._get_group_number(k, groups)

            mesh_a.cell_data["Group"] = tmp
            mesh_b = generate_volume_from_two_surfaces(
                mesh_a, item2.mesh, item2.resolution, item2.method
            )

            if i > 0:
                axis = (
                    self.mesh.dimensions.index(1)
                    if isinstance(mesh, pv.StructuredGrid)
                    else None
                )
                mesh = merge(mesh, mesh_b, axis)

            else:
                mesh = mesh_b

        mesh.user_dict["Group"] = groups

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        return mesh

    @property
    def mesh(self) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Return base mesh."""
        return self._mesh

    @property
    def scale(self) -> float | None:
        """Return default scaling factor."""
        return self._scale

    @property
    def angle(self) -> float | None:
        """Return default rotation angle."""
        return self._angle
