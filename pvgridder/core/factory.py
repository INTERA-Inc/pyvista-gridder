from __future__ import annotations
from typing import Optional
from typing_extensions import Self

import pyvista as pv

from ._base import MeshBase, MeshItem


class MeshFactory(MeshBase):
    __name__: str = "MeshFactory"
    __qualname__: str = "pvgridder.MeshFactory"

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
    ) -> Self:
        mesh = mesh.cast_to_unstructured_grid()
        
        # Rotate
        if angle is not None:
            mesh = mesh.rotate_z(angle)

        # Add group
        item = MeshItem(mesh, group=group)
        self.items.append(item)

        return self

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.UnstructuredGrid:
        if len(self.items) == 0:
            raise ValueError("not enough items to merge")

        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = item.mesh
            mesh_b.cell_data["Group"] = self._initialize_group_array(mesh_b, groups, item.group)

            if i > 0:
                mesh += mesh_b

            else:
                mesh = mesh_b.cast_to_unstructured_grid()

        mesh.user_dict["Group"] = groups

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)
