from __future__ import annotations
from typing import Literal, Optional
from numpy.typing import ArrayLike

import numpy as np
import pyvista as pv

from ._helpers import (
    generate_arc,
    generate_line_from_two_points,
    generate_plane_surface_from_two_lines,
    generate_volume_from_two_surfaces,
    translate
)


def AnnularSector(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    theta_min: Optional[float] = None,
    theta_max: Optional[float] = None,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError("invalid annular sector radii")

    line_a = generate_arc(inner_radius, theta_min, theta_max, theta_resolution, theta_method)
    line_b = generate_arc(outer_radius, theta_min, theta_max, theta_resolution, theta_method)
    mesh = generate_plane_surface_from_two_lines(line_a, line_b, r_resolution, r_method)
    mesh = translate(mesh, center)

    return mesh


def Surface(
    line_a: Optional[pv.PolyData | ArrayLike] = None,
    line_b: Optional[pv.PolyData | ArrayLike] = None,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid:
    line_a = line_a if line_a is not None else [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    line_b = line_b if line_b is not None else [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    mesh = generate_plane_surface_from_two_lines(line_a, line_b, resolution, method)

    return mesh


def Quadrilateral(
    points: Optional[ArrayLike] = None,
    x_resolution: Optional[int | ArrayLike] = None,
    y_resolution: Optional[int | ArrayLike] = None,
    x_method: Optional[Literal["constant", "log", "log_r"]] = None,
    y_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    if points is None:
        points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

    line_a = generate_line_from_two_points(points[0], points[1], x_resolution, x_method)
    line_b = generate_line_from_two_points(points[3], points[2], x_resolution, x_method)
    mesh = Surface(line_a, line_b, y_resolution, y_method)
    mesh = translate(mesh, center)

    return mesh


def Rectangle(
    dx: float = 1.0,
    dy: float = 1.0,
    x_resolution: Optional[int | ArrayLike] = None,
    y_resolution: Optional[int | ArrayLike] = None,
    x_method: Optional[Literal["constant", "log", "log_r"]] = None,
    y_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    points = [
        (0.0, 0.0),
        (dx, 0.0),
        (dx, dy),
        (0.0, dy),
    ]
    
    return Quadrilateral(points, x_resolution, y_resolution, x_method, y_method, center)


def Sector(
    radius: float = 1.0,
    theta_min: Optional[float] = None,
    theta_max: Optional[float] = None,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    if radius <= 0.0:
        raise ValueError("invalid sector radius")

    points = generate_arc(radius, theta_min, theta_max, resolution, method).points
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
    mesh = translate(mesh, center)

    return mesh


def SectorRectangle(
    radius: float = 0.5,
    dx: float = 1.0,
    dy: float = 1.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    if not 0.0 < radius < min(dx, dy):
        raise ValueError("invalid sector radius")

    line_x = generate_line_from_two_points([dx, dy], [0.0, dy], theta_resolution, theta_method)
    line_y = generate_line_from_two_points([dx, 0.0], [dx, dy], theta_resolution, theta_method)
    line_45 = generate_arc(radius, 0.0, 45.0, theta_resolution)
    line_90 = generate_arc(radius, 45.0, 90.0, theta_resolution)
    mesh_y45 = generate_plane_surface_from_two_lines(line_45, line_y, r_resolution, r_method)
    mesh_x90 = generate_plane_surface_from_two_lines(line_90, line_x, r_resolution, r_method)
    mesh = (
        mesh_y45.cast_to_unstructured_grid()
        + mesh_x90.cast_to_unstructured_grid()
    )
    mesh = translate(mesh, center)

    return mesh


def Volume(
    surface_a: pv.StructuredGrid | pv.UnstructuredGrid,
    surface_b: pv.StructuredGrid | pv.UnstructuredGrid,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    mesh = generate_volume_from_two_surfaces(surface_a, surface_b, resolution, method)

    return mesh
