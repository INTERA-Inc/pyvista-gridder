from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import pyvista as pv
from typing_extensions import Self

from ._base import MeshBase, MeshItem


class MeshMerge(MeshBase):
    """
    Mesh merging class.

    Parameters
    ----------
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshMerge"
    __qualname__: str = "pvgridder.MeshMerge"

    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new mesh factory."""
        super().__init__(default_group, ignore_groups)

    def add(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add a new item to factory.

        Parameters
        ----------
        mesh : pv.StructuredGrid | pv.UnstructuredGrid
            Mesh to merge.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        mesh = mesh.cast_to_unstructured_grid()

        # Add group
        item = MeshItem(mesh, group=group)
        self.items.append(item)

        return self

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.UnstructuredGrid:
        """
        Generate mesh by merging all items.

        Parameters
        ----------
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points.

        Returns
        -------
        pyvista.StructuredGrid | pyvista.UnstructuredGrid
            Merged mesh.

        """
        if len(self.items) == 0:
            raise ValueError("not enough items to merge")

        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = item.mesh
            mesh_b.cell_data["CellGroup"] = self._initialize_group_array(
                mesh_b, groups, default_group=item.group
            )

            if i > 0:
                mesh += mesh_b

            else:
                mesh = mesh_b.cast_to_unstructured_grid()

        mesh.user_dict["CellGroup"] = groups
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)
