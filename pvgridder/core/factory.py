from __future__ import annotations
from typing import Optional

import pyvista as pv

from ._base import MeshBase


class MeshFactory(MeshBase):
    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        super().__init__(default_group, ignore_groups)

    def add(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        angle: Optional[float] = None,
        group: Optional[str] = None,
        return_mesh: bool = False,
    ) -> pv.UnstructuredGrid | None:
        mesh = mesh.cast_to_unstructured_grid()
        
        # Rotate
        if angle is not None:
            mesh = mesh.rotate_z(angle)

        # Add group
        item = {
            "mesh": mesh,
            "group": group,
        }
        self.items.append(item)

        if return_mesh:
            return mesh

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.UnstructuredGrid:
        if len(self.items) == 0:
            raise ValueError("not enough items to merge")

        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = item["mesh"]
            tmp = self._initialize_group_array(mesh_b, groups)

            if (tmp == -1).any():
                group = item["group"] if item["group"] else self.default_group
                tmp[tmp == -1] = self._get_group_number(group, groups)

            mesh_b.cell_data["group"] = tmp

            if i > 0:
                mesh += mesh_b

            else:
                mesh = mesh_b.cast_to_unstructured_grid()

        mesh.user_dict["group"] = groups

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)
