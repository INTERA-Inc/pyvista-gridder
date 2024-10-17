from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator


class MeshBase(ABC):
    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        self._default_group = default_group if default_group else "default"
        self._ignore_groups = list(ignore_groups) if ignore_groups else []

    def _initialize_group_array(
        self,
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        groups: dict,
    ) -> ArrayLike:
        group = np.full(mesh.n_cells, -1, dtype=int)

        if ("group" in mesh.cell_data and "group" in mesh.user_dict):
            for k, v in mesh.user_dict["group"].items():
                if k in self.ignore_groups:
                    continue

                group[mesh.cell_data["group"] == v] = self._get_group_number(k, groups)

        return group

    @staticmethod
    def _get_group_number(group: str, groups: dict) -> int:
        if group not in groups:
            groups[group] = len(groups)

        return groups[group]

    @abstractmethod
    def generate_mesh(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        pass

    @property
    def default_group(self) -> str:
        return self._default_group

    @property
    def ignore_groups(self) -> bool:
        return self._ignore_groups


class MeshFactoryBase(MeshBase):
    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        self._items = []
        super().__init__(default_group, ignore_groups)

    def add_mesh(
        self,
        mesh: pv.Grid, 
        origin: Optional[ArrayLike] = None,
        angle: Optional[float] = None,
        group: Optional[str] = None,
        return_mesh: bool = False,
    ) -> pv.UnstructuredGrid | None:
        mesh = mesh.cast_to_unstructured_grid()
        
        # Rotate
        if angle is not None:
            mesh = mesh.rotate_z(angle)

        # Translate
        if origin is not None:
            origin = np.append(origin, 0.0) if len(origin) == 2 else origin[:3]
            mesh = mesh.translate(origin)

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

    @property
    def items(self) -> list:
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

        self._mesh = mesh.copy()
        self._axis = axis
        self._items = []
        super().__init__(group, ignore_groups)

    def add(
        self,
        arg: float | ArrayLike | pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        nsub: Optional[int | ArrayLike] = None,
        group: Optional[str] = None,
        return_mesh: bool = False,
    ) -> pv.StructuredGrid | pv.UnstructuredGrid | None:
        if isinstance(arg, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh = self._interpolate(arg.points)

        else:
            if np.ndim(arg) == 0:
                if not self.items:
                    self.items.append({"mesh": self.mesh})

                mesh = self.items[-1]["mesh"].copy()
                mesh.points[:, self.axis] += arg

            else:
                arg = np.asarray(arg)

                if arg.ndim == 2:
                    if arg.shape[1] != 3:
                        raise ValueError("invalid 2D array")

                    mesh = self._interpolate(arg)

                else:
                    raise ValueError(f"could not add {arg.ndim}D array to stack")

        item = {"mesh": mesh}

        if self.items:
            item["nsub"] = nsub
            item["group"] = group

        self.items.append(item)

        if return_mesh:
            return mesh

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.StructuredGrid | pv.UnstructuredGrid:
        from .. import merge
        
        if len(self.items) <= 1:
            raise ValueError("not enough items to stack")

        groups = {}

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1["mesh"].copy()
            tmp = self._initialize_group_array(mesh_a, groups)

            if (tmp == -1).any():
                group = item2["group"] if item2["group"] else self.default_group
                tmp[tmp == -1] = self._get_group_number(group, groups)

            mesh_a.cell_data["group"] = tmp
            mesh_b = self._extrude(mesh_a, item2["mesh"], item2["nsub"])

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

    @property
    def items(self) -> list:
        return self._items
