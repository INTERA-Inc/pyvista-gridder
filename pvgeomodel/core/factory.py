from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional

import numpy as np
import pyvista as pv

from ._base import MeshFactoryBase
from ._helpers import (
    generate_arc,
    generate_line_from_two_points,
    generate_plane_surface_from_two_lines,
)


class MeshFactory2D(MeshFactoryBase):
    def add_plane_surface(
        self,
        line_a: pv.PolyData | ArrayLike,
        line_b: pv.PolyData | ArrayLike,
        nsub: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        mesh = generate_plane_surface_from_two_lines(line_a, line_b, nsub)

        return self.add_mesh(mesh, *args, **kwargs)

    def add_quad(
        self,
        point_0: ArrayLike,
        point_1: ArrayLike,
        point_2: ArrayLike,
        point_3: ArrayLike,
        nsub_x: Optional[int | list[float]] = None,
        nsub_y: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        line_a = generate_line_from_two_points(point_0, point_1, nsub_x)
        line_b = generate_line_from_two_points(point_3, point_2, nsub_x)

        return self.add_plane_surface(line_a, line_b, nsub_y, *args, **kwargs)

    def add_rectangle(
        self,
        dx: float,
        dy: float,
        nsub_x: Optional[int | list[float]] = None,
        nsub_y: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        points = [
            [0.0, 0.0],
            [dx, 0.0],
            [dx, dy],
            [0.0, dy],
        ]

        return self.add_quad(*points, nsub_x, nsub_y, *args, **kwargs)

    def add_sector(
        self,
        r: float,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
        nsub: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        if r <= 0.0:
            raise ValueError("invalid sector radius")

        points = generate_arc(r, theta_min, theta_max, nsub).points
        n_cells = len(points) - 1
        cells = np.column_stack(
            (
                np.zeros(n_cells),
                np.arange(1, n_cells + 1),
                np.arange(2, n_cells + 2),
            )
        ).astype(int)
        points = np.row_stack((np.zeros(3), points))
        mesh = pv.UnstructuredGrid({pv.CellType.TRIANGLE: cells}, points)

        return self.add_mesh(mesh, *args, **kwargs)

    def add_annular_sector(
        self,
        r_in: float,
        r_out: float,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
        nsub_r: Optional[int | list[float]] = None,
        nsub_theta: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        if not 0.0 < r_in < r_out:
            raise ValueError("invalid annular sector radii")

        line_a = generate_arc(r_in, theta_min, theta_max, nsub_theta)
        line_b = generate_arc(r_out, theta_min, theta_max, nsub_theta)
        mesh = generate_plane_surface_from_two_lines(line_a, line_b, nsub_r)

        return self.add_mesh(mesh, *args, **kwargs)

    def add_sector_rectangle(
        self,
        dx: float,
        dy: float,
        r: float,
        nsub_r: Optional[int | list[float]] = None,
        nsub_theta: Optional[int | list[float]] = None,
        *args,
        **kwargs
    ) -> pv.UnstructuredGrid | None:
        if r <= 0.0:
            raise ValueError("invalid sector radius")

        line_x = generate_line_from_two_points([dx, dy], [0.0, dy], nsub_theta)
        line_y = generate_line_from_two_points([dx, 0.0], [dx, dy], nsub_theta)
        line_45 = generate_arc(r, 0.0, 45.0, nsub_theta)
        line_90 = generate_arc(r, 45.0, 90.0, nsub_theta)
        mesh_y45 = generate_plane_surface_from_two_lines(line_45, line_y, nsub_r)
        mesh_x90 = generate_plane_surface_from_two_lines(line_90, line_x, nsub_r)
        mesh = (
            mesh_y45.cast_to_unstructured_grid()
            + mesh_x90.cast_to_unstructured_grid()
        )
        
        return self.add_mesh(mesh, *args, **kwargs)
