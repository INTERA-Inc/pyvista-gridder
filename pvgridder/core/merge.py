from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pyvista as pv

from ._base import MeshBase, MeshItem


if TYPE_CHECKING:
    from collections.abc import Sequence  # pragma: no cover
    from typing import Optional  # pragma: no cover

    from typing_extensions import Self  # pragma: no cover


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
        """Initialize a new mesh merger."""
        super().__init__(default_group, ignore_groups)

    def add(
        self,
        mesh: pv.DataSet,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add a new item to merge.

        Parameters
        ----------
        mesh : pyvista.DataSet
            Mesh to merge.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        mesh = mesh.cast_to_unstructured_grid()

        # Check for existing user_dict
        if "_PYVISTA_USER_DICT" not in mesh.field_data:
            mesh.field_data["_PYVISTA_USER_DICT"] = "{}"

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
        pyvista.UnstructuredGrid
            Merged mesh.

        """
        if len(self.items) == 0:
            raise ValueError("not enough items to merge")

        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = cast(pv.UnstructuredGrid, item.mesh)
            mesh_b.cell_data["CellGroup"] = self._initialize_group_array(
                mesh_b, groups, default_group=item.group
            )

            if i > 0:
                mesh = pv.merge((mesh, mesh_b))

            else:
                mesh = mesh_b.cast_to_unstructured_grid()

        mesh.user_dict = None  # make sure to reset user_dict
        mesh.user_dict = {"CellGroup": groups}
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))
