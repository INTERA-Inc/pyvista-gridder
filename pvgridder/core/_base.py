from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator

from ._helpers import stack_two_structured_grids


class MeshBase(ABC):
    def __init__(self, group: Optional[str] = None) -> None:
        self._group = group if group else "default"

    @abstractmethod
    def generate_mesh(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        pass

    @property
    def group(self) -> str:
        return self._group


class MeshFactoryBase(MeshBase):
    def __init__(self, group: Optional[str] = None) -> None:
        self._items = []
        super().__init__(group)

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
            "group": group if group else self.group,
        }
        self.items.append(item)

        if return_mesh:
            return mesh

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.UnstructuredGrid:
        mesh = pv.UnstructuredGrid()
        groups = {}

        for i, item in enumerate(self.items):
            mesh_b = item["mesh"]

            if item["group"] not in groups:
                groups[item["group"]] = len(groups)

            mesh_b.cell_data["group"] = [groups[item["group"]]] * mesh_b.n_cells
            mesh += mesh_b

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
    ) -> None:
        if axis not in {0, 1, 2}:
            raise ValueError(f"invalid axis {axis} (expected {{0, 1, 2}}, got {axis})")

        if isinstance(mesh, pv.StructuredGrid) and mesh.dimensions[axis] != 1:
            raise ValueError(f"invalid mesh or axis, dimension along axis {axis} should be 1 (got {mesh.dimensions[axis]})")

        self._mesh = mesh.copy()
        self._axis = axis
        self._items = []
        super().__init__(group)

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
                    raise ValueError("could not add first item with scalar arg")

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
            item["group"] = group if group else self.group

        self.items.append(item)

        if return_mesh:
            return mesh

    def generate_mesh(self, tolerance: float = 1.0e-8) -> pv.StructuredGrid | pv.UnstructuredGrid:
        if len(self.items) <= 1:
            raise ValueError("not enough items to stack")

        groups = {}

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_b = self._extrude(item1["mesh"], item2["mesh"], item2["nsub"])

            if item2["group"] not in groups:
                groups[item2["group"]] = len(groups)

            mesh_b.cell_data["group"] = [groups[item2["group"]]] * mesh_b.n_cells

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

    @abstractmethod
    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        pass

    def _interpolate(self, points: ArrayLike) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        mesh = self.mesh.copy()
        idx = [i for i in range(3) if i != self.axis and np.unique(points[:, i]).size > 1]

        if len(idx) > 1:
            interp = LinearNDInterpolator(points[:, idx], points[:, self.axis])
            mesh.points[:, self.axis] = interp(mesh.points[:, idx])

        else:
            idx = idx[0]
            mesh.points[:, self.axis] = np.interp(mesh.points[:, idx], points[:, idx], points[:, self.axis])

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
