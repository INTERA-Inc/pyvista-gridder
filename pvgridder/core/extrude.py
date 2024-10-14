from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

import numpy as np
import pyvista as pv

from ._base import MeshBase
from ._helpers import (
    generate_volume_from_two_surfaces,
    stack_two_structured_grids,
    is2d,
)


class MeshExtrude(MeshBase):
    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        vector: ArrayLike,
        angle: float = 0.0,
        scale: float = 1.0,
        group: Optional[str] = None,
        ignore_group: bool = False,
    ) -> None:
        if not is2d(mesh):
            raise ValueError()

        self._mesh = mesh
        self._vector = self._check_vector(vector)
        self._angle = angle
        self._scale = scale
        self._items = [{"mesh": mesh}]
        self._ignore_group = ignore_group
        super().__init__(group)

    def add(
        self,
        vector: Optional[ArrayLike] = None,
        angle: Optional[float] = None,
        scale: Optional[float] = None,
        nsub: Optional[int | ArrayLike] = None,
        group: Optional[str | dict] = None,
        return_mesh: bool = False,
    ) -> pv.StructuredGrid | pv.UnstructuredGrid | None:
        vector = vector if vector is not None else self.vector
        vector = self._check_vector(vector)
        angle = angle if angle is not None else self.angle
        scale = scale if scale is not None else self.scale

        mesh = self.items[-1]["mesh"].copy()
        mesh = mesh.translate(vector)
        
        if angle:
            mesh = mesh.rotate_vector(vector, angle, mesh.center)

        if scale:
            mesh.points = (mesh.points - mesh.center) * scale + mesh.center

        item = {
            "mesh": mesh,
            "nsub": nsub,
            "group": group if group else {},
        }
        self.items.append(item)

        if return_mesh:
            return mesh

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.StructuredGrid | pv.UnstructuredGrid:
        if len(self.items) <= 1:
            raise ValueError("not enough items to extrude")

        if self.ignore_group or "group" not in self.mesh.user_dict:
            groups = {}

        else:
            groups = dict(self.mesh.user_dict["group"])

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1["mesh"].copy()
            group = item2["group"]
            tmp = (
                np.full(mesh_a.n_cells, -1, dtype=int)
                if self.ignore_group or "group" not in mesh_a.cell_data
                else mesh_a.cell_data["group"]
            )

            for k, v in group.items():
                if k not in groups:
                    groups[k] = len(groups)

                tmp[v(mesh_a)] = groups[k]

            if (tmp == -1).any():
                if self.group not in groups:
                    groups[self.group] = len(groups)

                tmp = np.where(tmp >= 0, tmp, groups[self.group])

            mesh_a.cell_data["group"] = tmp
            mesh_b = generate_volume_from_two_surfaces(mesh_a, item2["mesh"], item2["nsub"])

            if i > 0:
                if isinstance(mesh, pv.StructuredGrid):
                    mesh = stack_two_structured_grids(mesh, mesh_b, self.axis)

                else:
                    mesh += mesh_b

            else:
                mesh = mesh_b

        mesh.user_dict["group"] = groups

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        return mesh

    @staticmethod
    def _check_vector(vector: ArrayLike) -> ArrayLike:
        vector = np.asarray(vector)

        if vector.shape != (3,):
            raise ValueError("invalid extrusion vector")

        return vector

    @property
    def mesh(self) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return self._mesh

    @property
    def vector(self) -> ArrayLike:
        return self._vector

    @property
    def angle(self) -> float:
        return self._angle

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def items(self) -> list:
        return self._items

    @property
    def ignore_group(self) -> bool:
        return self._ignore_group
