from __future__ import annotations

from typing import Literal, Optional
from collections.abc import Sequence

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike

from ._helpers import (
    generate_arc,
    generate_line_from_two_points,
    generate_surface_from_two_lines,
    generate_volume_from_two_surfaces,
    translate,
)
from .._common import require_package


def AnnularSector(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate an annular sector mesh.

    Parameters
    ----------
    inner_radius : scalar, default 0.5
        Annulus inner radius.
    outer_radius : scalar, optional 1.0
        Annulus outer radius.
    theta_min : scalar, default 0.0
        Starting angle (in degree).
    theta_max : scalar, default 90.0
        Ending angle (in degree).
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the annulus inner radius.
    theta_resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
    r_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *r_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    theta_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *theta_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the annular sector.

    Returns
    -------
    pyvista.StructuredGrid
        Annular sector mesh.

    """
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError("invalid annular sector radii")

    line_a = generate_arc(
        inner_radius, theta_min, theta_max, theta_resolution, theta_method
    )
    line_b = generate_arc(
        outer_radius, theta_min, theta_max, theta_resolution, theta_method
    )
    mesh = Surface(line_a, line_b, "xy", r_resolution, r_method)
    mesh = translate(mesh, center)

    return mesh


def Surface(
    line_a: Optional[pv.PolyData | ArrayLike] = None,
    line_b: Optional[pv.PolyData | ArrayLike] = None,
    plane: Literal["xy", "yx", "xz", "zx", "yz", "zy"] = "xy",
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid:
    """
    Generate a surface mesh from two polylines.

    Parameters
    ----------
    line_a : pyvista.PolyData | ArrayLike, optional
        Starting polyline mesh or coordinates.
    line_b : pyvista.PolyData | ArrayLike, optional
        Ending polyline mesh or coordinates.
    plane : {'xy', 'yx', 'xz', 'zx', 'yz', 'zy'}, default 'xy'
        Surface plane.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the plane or relative position of subdivisions
        (in percentage) with respect to the starting line.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.StructuredGrid
        Surface mesh.

    """
    line_a = line_a if line_a is not None else [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    line_b = line_b if line_b is not None else [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    mesh = generate_surface_from_two_lines(line_a, line_b, plane, resolution, method)

    return mesh


@require_package("gmsh")
def Polygon(
    shell: Optional[pv.DataSet | ArrayLike] = None,
    holes: Optional[Sequence[pv.DataSet | ArrayLike]] = None,
    celltype: Optional[Literal["polygon", "quad", "triangle"]] = None,
    algorithm: int=6,
    optimization: Optional[Literal["Netgen", "Laplace2D", "Relocate2D"]] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a triangulated polygon with holes.

    Parameters
    ----------
    shell : pyvista.DataSet | ArrayLike, optional
        Polyline or a sequence of (x, y [,z]) numeric coordinate pairs or triples, or
        an array-like with shape (N, 2) or (N, 3).
    holes : Sequence[pyvista.DataSet | ArrayLike], optional
        A sequence of objects which satisfy the same requirements as the shell
        parameters above.
    celltype : {'polygon', 'quad', 'triangle'}, optional
        Preferred cell type. If `quad` or `triangle`, use Gmsh to perform 2D Delaunay
        triangulation.
    algorithm : int, default 6
        Gmsh algorithm.
    optimization : {'Netgen', 'Laplace2D', 'Relocate2D'}, optional
        Gmsh 2D optimization method.

    Returns
    -------
    pyvista.UnstructuredGrid
        Polygon mesh.

    """
    import gmsh

    def to_points(points: pv.PolyData) -> ArrayLike:
        """Convert to points array."""
        from .. import extract_boundary_polygons, split_lines

        if isinstance(points, pv.DataSet):
            edges = (
                split_lines(points, as_lines=False)
                if isinstance(points, pv.PolyData)
                else extract_boundary_polygons(points, fill=False)
            )

            if not edges:
                raise ValueError("could not extract boundary edges from input dataset")

            lines = edges[0].lines
            points = edges[0].points[lines[1 : lines[0] + 1]]

        else:
            points = np.asanyarray(points)

        if not (points[0] == points[-1]).all():
            points = np.row_stack((points, points[0]))

        if points.shape[1] == 2:
            points = np.insert(points, 2, 0.0, axis=-1)

        return points

    shell = shell if shell is not None else pv.Polygon()
    shell = to_points(shell)
    holes = [to_points(hole) for hole in holes] if holes is not None else []
    celltype = celltype if celltype else "triangle" if holes else "polygon"

    if celltype == "polygon":
        if holes:
            raise ValueError("could not generate a polygon of cell type 'polygon' with holes")

        points = shell[:-1]
        cells = np.insert(np.arange(len(points)), 0, len(points))
        celltypes = [pv.CellType.POLYGON]
        mesh = pv.UnstructuredGrid(cells, celltypes, points)

    elif celltype in {"quad", "triangle"}:
        try:
            gmsh.initialize()
            curve_tags = []

            for points in [shell, *holes]:
                # Compute mesh size
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
                lengths = np.insert(lengths, 0, lengths[-1])
                sizes = np.maximum(lengths[:-1], lengths[1:])
                sizes *= 1.0 if celltype == "triangle" else 2.0

                # Add points
                node_tags = []

                for (x, y, z), size in zip(points[:-1], sizes):
                    tag = gmsh.model.geo.add_point(x, y, z, size)
                    node_tags.append(tag)

                # Add lines
                line_tags = []

                for tag1, tag2 in zip(node_tags[:-1], node_tags[1:]):
                    tag = gmsh.model.geo.add_line(tag1, tag2)
                    line_tags.append(tag)

                # Close loop
                tag = gmsh.model.geo.add_line(node_tags[-1], node_tags[0])
                line_tags.append(tag)

                tag = gmsh.model.geo.add_curve_loop(line_tags)
                curve_tags.append(tag)

            # Add plane surface
            tag = gmsh.model.geo.add_plane_surface(curve_tags)

            # Generate mesh
            gmsh.option.set_number("Mesh.Algorithm", algorithm)
            gmsh.model.geo.synchronize()

            if celltype == "quad":
                gmsh.model.mesh.setRecombine(2, tag)

            gmsh.model.mesh.generate(2)

            if optimization:
                gmsh.model.mesh.optimize(optimization, force=True)

            # Convert to PyVista
            node_tags, coord, _ = gmsh.model.mesh.getNodes()
            element_types, _, element_node_tags = gmsh.model.mesh.getElements()
            gmsh_to_pyvista_type = {2: pv.CellType.TRIANGLE, 3: pv.CellType.QUAD}

            points = np.reshape(coord, (-1, 3))
            cells = {
                gmsh_to_pyvista_type[type_]: np.reshape(
                    node_tags,
                    (-1, gmsh.model.mesh.getElementProperties(type_)[3]),
                ) - 1
                for type_, node_tags in zip(element_types, element_node_tags)
                if type_ not in {1, 15}  # ignore line and vertex
            }
            mesh = pv.UnstructuredGrid(cells, points)

        finally:
            gmsh.clear()
            gmsh.finalize()

    else:
        raise ValueError(f"invalid cell type '{celltype}'")

    return mesh


def Quadrilateral(
    points: Optional[ArrayLike] = None,
    x_resolution: Optional[int | ArrayLike] = None,
    y_resolution: Optional[int | ArrayLike] = None,
    x_method: Optional[Literal["constant", "log", "log_r"]] = None,
    y_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate a quadrilateral mesh defined by 4 points.

    Parameters
    ----------
    points : ArrayLike, optional
        Points of the quadrilateral.
    x_resolution : int | ArrayLike, optional
        Number of subdivisions along the X axis or relative position of subdivisions
        (in percentage) with respect to the X coordinate of the first point.
    y_resolution : int | ArrayLike, optional
        Number of subdivisions along the Y axis or relative position of subdivisions
        (in percentage) with respect to the Y coordinate of the first point.
    x_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *x_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).
    y_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *y_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the quadrilateral.

    Returns
    -------
    pyvista.StructuredGrid
        Quadrilateral mesh.

    """
    if points is None:
        points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]

    line_a = generate_line_from_two_points(points[0], points[1], x_resolution, x_method)
    line_b = generate_line_from_two_points(points[3], points[2], x_resolution, x_method)
    mesh = Surface(line_a, line_b, "xy", y_resolution, y_method)
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
    """
    Generate a rectangle mesh of a given size.

    Parameters
    ----------
    dx : scalar, default 1.0
        Size of rectangle along X axis.
    dy : scalar, default 1.0
        Size of rectangle along Y axis.
    x_resolution : int | ArrayLike, optional
        Number of subdivisions along the X axis or relative position of subdivisions
        (in percentage) with respect to the X coordinate of the first point.
    y_resolution : int | ArrayLike, optional
        Number of subdivisions along the Y axis or relative position of subdivisions
        (in percentage) with respect to the Y coordinate of the first point.
    x_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *x_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).
    y_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *y_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the rectangle.

    Returns
    -------
    pyvista.StructuredGrid
        Rectangle mesh.

    """
    points = [
        (0.0, 0.0),
        (dx, 0.0),
        (dx, dy),
        (0.0, dy),
    ]

    return Quadrilateral(points, x_resolution, y_resolution, x_method, y_method, center)


def RegularLine(points: ArrayLike, resolution: Optional[int] = None) -> pv.PolyData:
    """
    Generate a polyline with regularly spaced points.

    Parameters
    ----------
    points : ArrayLike
        List of points defining a polyline.
    resolution : int, optional
        Number of points to interpolate along the points array. Defaults to `len(points)`.
    
    Returns
    -------
    pyvista.PolyData
        Line mesh with regularly spaced points.

    """
    resolution = resolution if resolution else len(points)

    xp = np.insert(
        np.sqrt(np.square(np.diff(points, axis=0)).sum(axis=1)),
        0,
        0.0,
    ).cumsum()
    x = np.linspace(0.0, xp.max(), resolution + 1)
    points = np.column_stack([np.interp(x, xp, fp) for fp in points.T])

    return pv.MultipleLines(points)


def Sector(
    radius: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a sector mesh.

    Parameters
    ----------
    radius : scalar, default 1.0
        Sector radius.
    theta_min : scalar, default 0.0
        Starting angle (in degree).
    theta_max : scalar, default 90.0
        Ending angle (in degree).
    resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the inner radius.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the sector.

    Returns
    -------
    pyvista.StructuredGrid
        Sector mesh.

    """
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
    """
    Generate a rectangle mesh with removed sector.

    Parameters
    ----------
    radius : scalar, default 0.5
        Sector radius.
    dx : scalar, default 1.0
        Size of rectangle along X axis.
    dy : scalar, default 1.0
        Size of rectangle along Y axis.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the annulus inner radius.
    theta_resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) between 0 and 45 degrees (and 45 and 90 degrees).
    r_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *r_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    theta_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *theta_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the sector.

    Returns
    -------
    pyvista.StructuredGrid
        Rectangle mesh with removed sector.

    """
    if not 0.0 < radius < min(dx, dy):
        raise ValueError("invalid sector radius")

    line_x = generate_line_from_two_points(
        [dx, dy], [0.0, dy], theta_resolution, theta_method
    )
    line_y = generate_line_from_two_points(
        [dx, 0.0], [dx, dy], theta_resolution, theta_method
    )
    line_45 = generate_arc(radius, 0.0, 45.0, theta_resolution)
    line_90 = generate_arc(radius, 45.0, 90.0, theta_resolution)
    mesh_y45 = Surface(line_45, line_y, "xy", r_resolution, r_method)
    mesh_x90 = Surface(line_90, line_x, "xy", r_resolution, r_method)
    mesh = mesh_y45.cast_to_unstructured_grid() + mesh_x90.cast_to_unstructured_grid()
    mesh = translate(mesh, center)

    return mesh


def Volume(
    surface_a: pv.StructuredGrid | pv.UnstructuredGrid,
    surface_b: pv.StructuredGrid | pv.UnstructuredGrid,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Generate a volume mesh.

    Parameters
    ----------
    surface_a : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Starting surface mesh.
    surface_b : pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Ending surface mesh.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the extrusion axis or relative position of
        subdivisions (in percentage) with respect to the starting surface.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    Returns
    -------
    pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Volume mesh.

    """
    mesh = generate_volume_from_two_surfaces(surface_a, surface_b, resolution, method)

    return mesh
