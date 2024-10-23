from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional

import numpy as np
import pyvista as pv

from ._base import MeshBase, MeshItem
from ._helpers import (
    generate_volume_from_two_surfaces,
    is2d,
)


class MeshExtrude(MeshBase):
    def __init__(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        scale: Optional[float] = None,
        angle: Optional[float] = None,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        if not is2d(mesh):
            raise ValueError("invalid mesh, input mesh should be a 2D structured grid or an unstructured grid")

        super().__init__(default_group, ignore_groups, items=[MeshItem(mesh)])
        self._mesh = mesh
        self._angle = angle
        self._scale = scale

    def add(
        self,
        vector: ArrayLike,
        nsub: Optional[int | ArrayLike] = None,
        method: Optional[Literal["constant", "log", "log_r"]] = None,
        scale: Optional[float] = None,
        angle: Optional[float] = None,
        group: Optional[str | dict] = None,
    ) -> None:
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

        item = MeshItem(mesh, nsub=nsub, method=method, group=group)
        self.items.append(item)

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.StructuredGrid | pv.UnstructuredGrid:
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

            if (tmp == -1).any():
                tmp[tmp == -1] = self._get_group_number(self.default_group, groups)

            mesh_a.cell_data["group"] = tmp
            mesh_b = generate_volume_from_two_surfaces(mesh_a, item2.mesh, item2.nsub, item2.method)

            if i > 0:
                axis = self.mesh.dimensions.index(1) if isinstance(mesh, pv.StructuredGrid) else None
                mesh = merge(mesh, mesh_b, axis)
                
            else:
                mesh = mesh_b

        mesh.user_dict["group"] = groups

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        return mesh

    @property
    def mesh(self) -> pv.StructuredGrid | pv.UnstructuredGrid:
        return self._mesh

    @property
    def scale(self) -> float | None:
        return self._scale

    @property
    def angle(self) -> float | None:
        return self._angle
