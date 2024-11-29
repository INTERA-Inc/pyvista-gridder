from __future__ import annotations

from typing import Optional

import pyvista as pv
from typing_extensions import Self

from ._base import MeshBase, MeshItem


class MeshFactory(MeshBase):
    """
    Mesh factory class.

    Parameters
    ----------
    default_group : str, optional
        Default group name.
    ignore_groups : sequence of str, optional
        List of groups to ignore.

    """

    __name__: str = "MeshFactory"
    __qualname__: str = "pvgridder.MeshFactory"

    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
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
        :class:`pyvista.StructuredGrid` | :class:`pyvista.UnstructuredGrid`
            Merged mesh.

        """
        if len(self.items) == 0:
            raise ValueError("not enough items to merge")

        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = item.mesh
            mesh_b.cell_data["Group"] = self._initialize_group_array(
                mesh_b, groups, item.group
            )

            if i > 0:
                mesh += mesh_b

            else:
                mesh = mesh_b.cast_to_unstructured_grid()

        mesh.user_dict["Group"] = groups

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)
