from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional

from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator


class MeshItem:
    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        **kwargs
    ) -> None:
        self._mesh = mesh
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        return self._mesh


class MeshBase(ABC):
    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
        items: Optional[list[MeshItem]] = None,
    ) -> None:
        self._default_group = default_group if default_group else "default"
        self._ignore_groups = list(ignore_groups) if ignore_groups else []
        self._items = items if items else []

    def _initialize_group_array(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        groups: dict,
        default_group: Optional[str] = None,
    ) -> ArrayLike:
        group = np.full(mesh.n_cells, -1, dtype=int)

        if ("group" in mesh.cell_data and "group" in mesh.user_dict):
            for k, v in mesh.user_dict["group"].items():
                if k in self.ignore_groups:
                    continue

                group[mesh.cell_data["group"] == v] = self._get_group_number(k, groups)

        if (group == -1).any():
            default_group = default_group if default_group else self.default_group
            group[group == -1] = self._get_group_number(default_group, groups)

        return group

    @staticmethod
    def _get_group_number(group: str, groups: dict) -> int:
        return groups.setdefault(group, len(groups))

    @abstractmethod
    def generate_mesh(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        pass

    @property
    def default_group(self) -> str:
        return self._default_group

    @property
    def ignore_groups(self) -> bool:
        return self._ignore_groups

    @property
    def items(self) -> list[MeshItem]:
        return self._items


class MeshStackBase(MeshBase):
    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        if axis not in {0, 1, 2}:
            raise ValueError(f"invalid axis {axis} (expected {{0, 1, 2}}, got {axis})")

        if isinstance(mesh, pv.StructuredGrid) and mesh.dimensions[axis] != 1:
            raise ValueError(f"invalid mesh or axis, dimension along axis {axis} should be 1 (got {mesh.dimensions[axis]})")

        super().__init__(group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis

    def add(
        self,
        arg: float | ArrayLike | pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        nsub: Optional[int | ArrayLike] = None,
        method: Optional[Literal["constant", "log", "log_r"]] = None,
        group: Optional[str] = None,
    ) -> None:
        if isinstance(arg, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh = self._interpolate(arg.points)

        else:
            if np.ndim(arg) == 0:
                if not self.items:
                    mesh = self.mesh.copy()
                    mesh.points[:, self.axis] = arg

                else:
                    mesh = self.items[-1].mesh.copy()
                    mesh.points[:, self.axis] += arg

            else:
                arg = np.asarray(arg)

                if arg.ndim == 2:
                    if arg.shape[1] != 3:
                        raise ValueError("invalid 2D array")

                    mesh = self._interpolate(arg)

                else:
                    raise ValueError(f"could not add {arg.ndim}D array to stack")

        item = (
            MeshItem(mesh, nsub=nsub, method=method, group=group)
            if self.items
            else MeshItem(mesh)
        )
        self.items.append(item)

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.StructuredGrid | pv.UnstructuredGrid:
        from .. import merge
        
        if len(self.items) <= 1:
            raise ValueError("not enough items to stack")

        groups = {}

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1.mesh.copy()
            mesh_a.cell_data["group"] = self._initialize_group_array(mesh_a, groups, item2.group)
            mesh_b = self._extrude(mesh_a, item2.mesh, item2.nsub, item2.method)

            if i > 0:
                mesh = merge(mesh, mesh_b, self.axis)

            else:
                mesh = mesh_b

        mesh.user_dict["group"] = groups

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        return mesh

    @abstractmethod
    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        pass

    def _interpolate(self, points: ArrayLike) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        mesh = self.mesh.copy()
        idx = [i for i in range(3) if i != self.axis and np.unique(points[:, i]).size > 1]

        if len(idx) > 1:
            interp = LinearNDInterpolator(points[:, idx], points[:, self.axis])
            tmp = interp(mesh.points[:, idx])

            if np.isnan(tmp).any():
                raise ValueError("could not interpolate from points not fully enclosing base mesh")

            mesh.points[:, self.axis] = tmp

        else:
            idx = idx[0]
            x = mesh.points[:, idx]
            xp = points[:, idx]

            if not (xp[0] <= x[0] <= xp[-1] and xp[0] <= x[-1] <= xp[-1]):
                raise ValueError("could not interpolate from points not fully enclosing base mesh")

            mesh.points[:, self.axis] = np.interp(x, xp, points[:, self.axis])

        return mesh

    @property
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        return self._mesh

    @property
    def axis(self) -> int:
        return self._axis
