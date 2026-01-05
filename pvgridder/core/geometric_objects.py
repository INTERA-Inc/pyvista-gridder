from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pyvista as pv
from pyrequire import require_package

from ._helpers import (
    generate_arc,
    generate_line_from_two_points,
    generate_surface_from_two_lines,
    generate_volume_from_two_surfaces,
    translate,
)


if TYPE_CHECKING:
    from collections.abc import Sequence  # pragma: no cover
    from typing import Literal, Optional  # pragma: no cover

    from numpy.typing import ArrayLike, NDArray  # pragma: no cover


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
    if not 0.0 <= inner_radius < outer_radius:
        raise ValueError("invalid annular sector radii")

    line_a = generate_arc(
        inner_radius, theta_min, theta_max, theta_resolution, theta_method
    )
    line_b = generate_arc(
        outer_radius, theta_min, theta_max, theta_resolution, theta_method
    )
    mesh = StructuredSurface(line_a, line_b, "xy", r_resolution, r_method)
    mesh = translate(mesh, center)
    mesh = cast(pv.StructuredGrid, mesh)

    return mesh


def Annulus(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate an annulus mesh.

    Parameters
    ----------
    inner_radius : scalar, default 0.5
        Annulus inner radius.
    outer_radius : scalar, optional 1.0
        Annulus outer radius.
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
        Center of the annulus.

    Returns
    -------
    pyvista.StructuredGrid
        Annulus mesh.

    """
    mesh = AnnularSector(
        inner_radius,
        outer_radius,
        0.0,
        360.0,
        r_resolution,
        theta_resolution,
        r_method,
        theta_method,
        center,
    )

    return mesh


def Circle(
    radius: float = 1.0,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a circle mesh.

    Parameters
    ----------
    radius : scalar, default 1.0
        Circle radius.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the circle.

    Returns
    -------
    pyvista.UnstructuredGrid
        Circle mesh.

    """
    mesh = Sector(radius, 0.0, 360.0, resolution, method, center).clean(
        produce_merge_map=False
    )
    mesh = cast(pv.UnstructuredGrid, mesh)

    return mesh


def CurvedLine(
    origin: ArrayLike,
    length: float,
    start: ArrayLike,
    end: Optional[ArrayLike] = None,
    resolution: int = 1,
) -> pv.PolyData:
    """
    Generate a curved line mesh.

    Parameters
    ----------
    origin : ArrayLike
        Origin point of the curved line.
    length : float
        Length of the curved line.
    start : ArrayLike
        Starting direction vector of the curved line.
    end : ArrayLike, optional
        Ending direction vector of the curved line. If None, defaults to *start*.
    resolution : int, default 1
        Number of segments to divide the curved line into.

    Returns
    -------
    pyvista.PolyData
        Curved line mesh.

    """
    origin = np.asanyarray(origin)
    start = np.asanyarray(start)
    end = np.asanyarray(end) if end is not None else start

    # Normalize direction vectors
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)

    # Compute the direction vectors
    theta = np.arccos((start @ end).clip(-1.0, 1.0))
    directions = (
        np.tile(start, (resolution + 1, 1))
        if theta == 0.0
        else np.array(
            [
                (np.sin((1.0 - t) * theta) * start + np.sin(t * theta) * end)
                / np.sin(theta)
                for t in np.linspace(0.0, 1.0, resolution + 1)
            ]
        )
    )

    # Step size along the curve
    points = [origin]
    dl = length / resolution

    for d1, d2 in zip(directions[:-1], directions[1:]):
        vec = 0.5 * (d1 + d2)
        vec /= np.linalg.norm(vec)
        points.append(points[-1] + dl * vec)

    mesh = pv.MultipleLines(np.asanyarray(points))
    mesh.clear_data()

    return mesh


def CylindricalShell(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    height: float = 1.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    z_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    z_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate a cylindrical shell mesh.

    Parameters
    ----------
    inner_radius : scalar, default 0.5
        Annulus inner radius.
    outer_radius : scalar, optional 1.0
        Annulus outer radius.
    height : scalar, default 1.0
        Cylinder height.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the cylinder inner radius.
    theta_resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
    z_resolution : int | ArrayLike, optional
        Number of subdivisions along the height axis or relative position of
        subdivisions (in percentage) with respect to the bottom height.
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

    z_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *z_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the cylindrical shell.

    Returns
    -------
    pyvista.StructuredGrid
        Cylindrical shell mesh.

    """
    return CylindricalShellSector(
        inner_radius,
        outer_radius,
        0.0,
        360.0,
        height,
        r_resolution,
        theta_resolution,
        z_resolution,
        r_method,
        theta_method,
        z_method,
        center,
    )


def CylindricalShellSector(
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    height: float = 1.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    z_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    z_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate a cylindrical shell sector mesh.

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
    height : scalar, default 1.0
        Cylinder height.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the cylinder inner radius.
    theta_resolution : int | ArrayLike, optional
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
    z_resolution : int | ArrayLike, optional
        Number of subdivisions along the height axis or relative position of
        subdivisions (in percentage) with respect to the bottom height.
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

    z_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *z_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the cylindrical shell sector.

    Returns
    -------
    pyvista.StructuredGrid
        Cylindrical shell sector mesh.

    """
    center = list(np.asanyarray(center)) if center is not None else [0.0, 0.0, 0.0]
    center[2] -= 0.5 * height

    surface_a = AnnularSector(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        theta_min=theta_min,
        theta_max=theta_max,
        r_resolution=r_resolution,
        theta_resolution=theta_resolution,
        r_method=r_method,
        theta_method=theta_method,
    )
    surface_b = surface_a.translate([0.0, 0.0, height])
    mesh = generate_volume_from_two_surfaces(
        surface_a, surface_b, z_resolution, z_method
    )
    mesh = translate(mesh, center)
    mesh = cast(pv.StructuredGrid, mesh)

    return mesh


@require_package("gmsh")
def Polygon(
    shell: Optional[pv.DataSet | ArrayLike] = None,
    holes: Optional[Sequence[pv.DataSet | ArrayLike]] = None,
    celltype: Optional[Literal["polygon", "quad", "triangle"]] = None,
    cellsize: Optional[float] = None,
    algorithm: int = 6,
    optimization: Optional[Literal["Netgen", "Laplace2D", "Relocate2D"]] = None,
    engine: Literal["gmsh", "occ"] = "gmsh",
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
    cellsize : float, optional
        Size of the mesh elements. If None, the size is computed from the input points.
    algorithm : int, default 6
        Gmsh algorithm.
    optimization : {'Netgen', 'Laplace2D', 'Relocate2D'}, optional
        Gmsh 2D optimization method.
    engine : {'gmsh', 'occ'}, default 'gmsh'
        Geometry engine:

         - 'gmsh': Gmsh built-in engine
         - 'occ': OpenCascade

    Returns
    -------
    pyvista.UnstructuredGrid
        Polygon mesh.

    """
    import gmsh

    def to_points(points: ArrayLike | pv.DataSet) -> NDArray:
        """Convert to points array."""
        from .. import extract_boundary_polygons, split_lines

        if isinstance(points, pv.DataSet):
            edges = (
                split_lines(points, as_lines=False)
                if isinstance(points, pv.PolyData) and points.n_lines > 0
                else extract_boundary_polygons(points, fill=False)
            )

            if not edges:
                raise ValueError("could not extract boundary edges from input dataset")

            edges = cast(list[pv.PolyData], edges)
            lines = edges[0].lines
            points = edges[0].points[lines[1 : lines[0] + 1]]

        else:
            points = np.asanyarray(points)

        if not (points[0] == points[-1]).all():
            points = np.vstack((points, points[0]))

        if points.shape[1] == 2:
            points = np.insert(points, 2, 0.0, axis=-1)

        return points

    def add_surface(
        engine: type[gmsh.model.geo] | type[gmsh.model.occ],
        points: pv.DataSet | ArrayLike,
        celltype: str,
        cellsize: float | None,
        return_curve_loop: bool = False,
    ) -> int | tuple[int, int]:
        """Add a plane surface."""
        points = np.asanyarray(points)

        # Compute mesh size
        if cellsize is None:
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
            lengths = np.insert(lengths, 0, lengths[-1])
            sizes = np.maximum(lengths[:-1], lengths[1:])

        else:
            sizes = np.full(len(points) - 1, cellsize, dtype=float)

        sizes *= 1.0 if celltype == "triangle" else 2.0

        # Add points
        node_tags = []

        for (x, y, z), size in zip(points[:-1], sizes):
            tag = engine.add_point(x, y, z, size)
            node_tags.append(tag)

        # Add lines
        line_tags = []

        for tag1, tag2 in zip(node_tags[:-1], node_tags[1:]):
            tag = engine.add_line(tag1, tag2)
            line_tags.append(tag)

        # Close loop
        tag = engine.add_line(node_tags[-1], node_tags[0])
        line_tags.append(tag)
        tag = engine.add_curve_loop(line_tags)

        if return_curve_loop:
            return tag

        else:
            tag = engine.add_plane_surface([tag])

            return (2, tag)

    shell = shell if shell is not None else pv.Polygon()
    shell = to_points(shell)
    holes = [to_points(hole) for hole in holes] if holes is not None else []
    celltype = celltype if celltype else "triangle" if holes else "polygon"

    if celltype == "polygon":
        if holes:
            raise ValueError(
                "could not generate a polygon of cell type 'polygon' with holes"
            )

        points = shell[:-1]
        cells = np.insert(np.arange(len(points)), 0, len(points))
        celltypes = [pv.CellType.POLYGON]
        mesh = pv.UnstructuredGrid(cells, celltypes, points)

    elif celltype in {"quad", "triangle"}:
        try:
            gmsh.initialize()

            # Generate plane surfaces from points
            if engine == "gmsh":
                engine_ = gmsh.model.geo
                curve_tags = [
                    add_surface(
                        engine_, points, celltype, cellsize, return_curve_loop=True
                    )
                    for points in [shell, *holes]
                ]
                tag = engine_.add_plane_surface(curve_tags)
                dim_tags = [(2, tag)]

            elif engine == "occ":
                engine_ = gmsh.model.occ
                shell_tags = [add_surface(engine_, shell, celltype, cellsize)]

                if holes:
                    hole_tags = [
                        add_surface(engine_, hole, celltype, cellsize) for hole in holes
                    ]
                    dim_tags, _ = engine_.cut(
                        shell_tags,
                        hole_tags,
                        removeObject=True,
                        removeTool=True,
                    )

                else:
                    dim_tags = cast(list[tuple[int, int]], shell_tags)

            else:
                raise ValueError(f"invalid engine '{engine}'")

            # Generate mesh
            if cellsize is not None:
                gmsh.option.set_number("Mesh.MeshSizeMin", cellsize)
                gmsh.option.set_number("Mesh.MeshSizeMax", cellsize)
                # gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
                # gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)

            gmsh.option.set_number("Mesh.Algorithm", algorithm)
            gmsh.option.set_number("General.Verbosity", 0)
            engine_.synchronize()

            if celltype == "quad":
                for dim_tag in dim_tags:
                    gmsh.model.mesh.setRecombine(*dim_tag)

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
                )
                - 1
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
    points = (
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        if points is None
        else points
    )
    points = np.asanyarray(points)

    line_a = generate_line_from_two_points(points[0], points[1], x_resolution, x_method)
    line_b = generate_line_from_two_points(points[3], points[2], x_resolution, x_method)
    mesh = StructuredSurface(line_a, line_b, "xy", y_resolution, y_method)
    mesh = translate(mesh, center)
    mesh = cast(pv.StructuredGrid, mesh)

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
    mesh = Quadrilateral(points, x_resolution, y_resolution, x_method, y_method, center)

    return mesh


def RectangleSector(
    dx: float = 0.5,
    dy: float = 0.5,
    radius: float = 1.0,
    x_resolution: Optional[int | ArrayLike] = None,
    y_resolution: Optional[int | ArrayLike] = None,
    r_resolution: Optional[int | ArrayLike] = None,
    x_method: Optional[Literal["constant", "log", "log_r"]] = None,
    y_method: Optional[Literal["constant", "log", "log_r"]] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a sector mesh with rectangle removed at the center.

    Parameters
    ----------
    dx : scalar, default 0.5
        Size of rectangle along X axis.
    dy : scalar, default 0.5
        Size of rectangle along Y axis.
    radius : scalar, default 1.0
        Sector radius.
    x_resolution : int | ArrayLike, optional
        Number of subdivisions along the X axis or relative position of subdivisions
        (in percentage) with respect to the corner of the rectangle.
    y_resolution : int | ArrayLike, optional
        Number of subdivisions along the Y axis or relative position of subdivisions
        (in percentage) with respect to the corner of the rectangle.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the corner of the rectangle.
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

    r_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *r_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the rectangle sector.

    Returns
    -------
    pyvista.UnstructuredGrid
        Rectangle mesh with sector removed at the center.

    """
    if 0.0 < radius < (dx**2 + dy**2) ** 0.5:
        raise ValueError("invalid sector radius")

    line_x = generate_line_from_two_points([dx, dy], [0.0, dy], x_resolution, x_method)
    line_y = generate_line_from_two_points([dx, dy], [dx, 0.0], y_resolution, y_method)
    line_45 = generate_arc(radius, 45.0, 0.0, y_resolution, y_method)
    line_90 = generate_arc(radius, 45.0, 90.0, x_resolution, x_method)
    mesh_y45 = StructuredSurface(line_y, line_45, "xy", r_resolution, r_method)
    mesh_x90 = StructuredSurface(line_x, line_90, "xy", r_resolution, r_method)
    mesh = mesh_y45.cast_to_unstructured_grid() + mesh_x90.cast_to_unstructured_grid()
    mesh = translate(mesh, center)
    mesh = cast(pv.UnstructuredGrid, mesh)

    return mesh


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
    points = np.asanyarray(points)
    resolution = resolution if resolution else len(points)

    xp = np.insert(
        np.sqrt(np.square(np.diff(points, axis=0)).sum(axis=1)),
        0,
        0.0,
    ).cumsum()
    x = np.linspace(0.0, xp.max(), resolution + 1)
    points = np.column_stack([np.interp(x, xp, fp) for fp in points.T])
    mesh = pv.MultipleLines(points)
    mesh.clear_data()

    return mesh


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
        Number of subdivisions along the azimuthal axis or relative position of
        subdivisions (in percentage) with respect to the starting angle.
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
    points = np.vstack((np.zeros(3), points))
    mesh = pv.UnstructuredGrid({pv.CellType.TRIANGLE: cells}, points)
    mesh = translate(mesh, center)
    mesh = cast(pv.UnstructuredGrid, mesh)

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
    Generate a rectangle mesh with sector removed at the center.

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
    pyvista.UnstructuredGrid
        Rectangle mesh with sector removed at the center

    """
    if not 0.0 < radius < min(dx, dy):
        raise ValueError("invalid sector radius")

    line_x = generate_line_from_two_points(
        [dx, dy], [0.0, dy], theta_resolution, theta_method
    )
    line_y = generate_line_from_two_points(
        [dx, dy], [dx, 0.0], theta_resolution, theta_method
    )
    line_45 = generate_arc(radius, 45.0, 0.0, theta_resolution)
    line_90 = generate_arc(radius, 45.0, 90.0, theta_resolution)
    mesh_y45 = StructuredSurface(line_45, line_y, "xy", r_resolution, r_method)
    mesh_x90 = StructuredSurface(line_90, line_x, "xy", r_resolution, r_method)
    mesh = mesh_y45.cast_to_unstructured_grid() + mesh_x90.cast_to_unstructured_grid()
    mesh = translate(mesh, center)
    mesh = cast(pv.UnstructuredGrid, mesh)

    return mesh


def SectorSquare(
    radius: float = 0.5,
    dx: float = 1.0,
    r_resolution: Optional[int | ArrayLike] = None,
    theta_resolution: Optional[int | ArrayLike] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    theta_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a square mesh with sector removed at the center.

    Parameters
    ----------
    radius : scalar, default 0.5
        Sector radius.
    dx : scalar, default 1.0
        Size of square along X axis.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the sector radius.
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
    pyvista.UnstructuredGrid
        Square mesh with sector removed at the center.

    """
    mesh = SectorRectangle(
        radius=radius,
        dx=dx,
        dy=dx,
        r_resolution=r_resolution,
        theta_resolution=theta_resolution,
        r_method=r_method,
        theta_method=theta_method,
        center=center,
    )

    return mesh


def Square(
    dx: float = 1.0,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.StructuredGrid:
    """
    Generate a square mesh of a given size.

    Parameters
    ----------
    dx : scalar, default 1.0
        Size of square along X and Y axes.
    resolution : int | ArrayLike, optional
        Number of subdivisions along the X and Y axes or relative position of
        subdivisions (in percentage) with respect to the X coordinate of the first point.
    method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the square.

    Returns
    -------
    pyvista.StructuredGrid
        Square mesh.

    """
    mesh = Rectangle(
        dx=dx,
        dy=dx,
        x_resolution=resolution,
        y_resolution=resolution,
        x_method=method,
        y_method=method,
        center=center,
    )

    return mesh


def SquareSector(
    dx: float = 0.5,
    radius: float = 1.0,
    x_resolution: Optional[int | ArrayLike] = None,
    r_resolution: Optional[int | ArrayLike] = None,
    x_method: Optional[Literal["constant", "log", "log_r"]] = None,
    r_method: Optional[Literal["constant", "log", "log_r"]] = None,
    center: Optional[ArrayLike] = None,
) -> pv.UnstructuredGrid:
    """
    Generate a sector mesh with square removed at the center.

    Parameters
    ----------
    dx : scalar, default 0.5
        Size of square along X axis.
    radius : scalar, default 1.0
        Sector radius.
    x_resolution : int | ArrayLike, optional
        Number of subdivisions along the X axis or relative position of subdivisions
        (in percentage) with respect to the corner of the square.
    r_resolution : int | ArrayLike, optional
        Number of subdivisions along the radial axis or relative position of
        subdivisions (in percentage) with respect to the corner of the square.
    x_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *x_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    r_method : {'constant', 'log', 'log_r'}, optional
        Subdivision method if *r_resolution* is an integer:

         - if 'constant', subdivisions are equally spaced.
         - if 'log', subdivisions are logarithmically spaced (from small to large).
         - if 'log_r', subdivisions are logarithmically spaced (from large to small).

    center : ArrayLike, optional
        Center of the sector.

    Returns
    -------
    pyvista.UnstructuredGrid
        Sector mesh with square removed at the center.

    """
    mesh = RectangleSector(
        dx=dx,
        dy=dx,
        radius=radius,
        x_resolution=x_resolution,
        y_resolution=x_resolution,
        r_resolution=r_resolution,
        x_method=x_method,
        y_method=x_method,
        r_method=r_method,
        center=center,
    )

    return mesh


def StructuredSurface(
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


def Volume(
    surface_a: pv.ImageData
    | pv.RectilinearGrid
    | pv.StructuredGrid
    | pv.UnstructuredGrid,
    surface_b: pv.ImageData
    | pv.RectilinearGrid
    | pv.StructuredGrid
    | pv.UnstructuredGrid,
    resolution: Optional[int | ArrayLike] = None,
    method: Optional[Literal["constant", "log", "log_r"]] = None,
) -> pv.StructuredGrid | pv.UnstructuredGrid:
    """
    Generate a volume mesh.

    Parameters
    ----------
    surface_a : pyvista.ImageData | pyvista.RectilinearGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Starting surface mesh.
    surface_b : pyvista.ImageData | pyvista.RectilinearGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
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
