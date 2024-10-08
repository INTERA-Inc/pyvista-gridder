from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional

import numpy as np
import pyvista as pv


def generate_arc(
    radius: float,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    nsub: Optional[int | list[float]] = None,
) -> pv.PolyData:
    perc = nsub_to_perc(nsub)
    angles = theta_min + perc * (theta_max - theta_min)
    angles = np.deg2rad(angles)
    points = radius * np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))

    return pv.MultipleLines(points)


def generate_line_from_two_points(
    point_a: ArrayLike,
    point_b: ArrayLike,
    nsub: Optional[int | list[float]] = None,
) -> pv.PolyData:
    point_a = np.asarray(point_a)
    point_b = np.asarray(point_b)

    if point_a.shape != point_b.shape:
        raise ValueError("could not generate a line from two inhomogeneous points")

    perc = nsub_to_perc(nsub)[:, np.newaxis]
    points = point_a + perc * (point_b - point_a)
    points = points if points.shape[1] == 3 else np.column_stack((points, np.zeros(len(points))))

    return pv.MultipleLines(points)


def generate_plane_surface_from_two_lines(
    line_a: pv.PolyData | ArrayLike,
    line_b: pv.PolyData | ArrayLike,
    nsub: Optional[int | list[float]] = None,
) -> pv.StructuredGrid:
    line_a = line_to_array(line_a)
    line_b = line_to_array(line_b)
    
    if line_a.shape != line_b.shape:
        raise ValueError("could not generate plane surface from two inhomogeneous lines")

    perc = nsub_to_perc(nsub)[:, np.newaxis, np.newaxis]
    X, Y = np.transpose(line_a + perc * (line_b - line_a), axes=(2, 1, 0))
    Z = np.zeros_like(X)

    return pv.StructuredGrid(X, Y, Z)


def line_to_array(line: pv.PolyData | ArrayLike) -> ArrayLike:
    if isinstance(line, pv.PolyData):
        line = line.points

    return np.asarray(line)[:, :2]


def nsub_to_perc(nsub: int | list[float]) -> list[float]:
    if np.ndim(nsub) == 0:
        nsub = nsub if nsub else 1
        perc = np.linspace(0.0, 1.0, nsub + 1)

    elif np.ndim(nsub) == 1:
        if not all(0.0 <= n <= 1 for n in nsub):
            raise ValueError("invalid subdivisions")

        perc = np.sort(nsub)

    else:
        raise ValueError("invalid subdivisions")

    return perc
