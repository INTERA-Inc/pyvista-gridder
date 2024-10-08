from __future__ import annotations
from typing import Optional

import numpy as np
import pyvista as pv

from abc import ABC


class MeshFactoryBase(ABC):
    def __init__(self):
        self._meshes = {}

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
        group = group if group else "default"

        if group in self.meshes:
            self.meshes[group].append(mesh)

        else:
            self.meshes[group] = [mesh]

        if return_mesh:
            return mesh

    def merge(
        self,
        groups: Optional[list[str]] = None,
        tolerance: float = 1.0e-8,
    ) -> pv.UnstructuredGrid:
        if groups is not None:
            for group in groups:
                if group not in self.meshes:
                    raise ValueError(f"invalid group '{group}'")

        else:
            groups = list(self.meshes)    
        
        mesh = pv.UnstructuredGrid()

        for i, group in enumerate(groups):
            try:
                group_mesh = pv.merge(self.meshes[group])

            except KeyError:
                raise ValueError(f"invalid group '{group}'")

            group_mesh["group"] = np.full(group_mesh.n_cells, i)
            mesh += group_mesh

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)

    @property
    def meshes(self) -> dict:
        return self._meshes
