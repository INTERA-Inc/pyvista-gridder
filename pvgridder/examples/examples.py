from __future__ import annotations

import numpy as np
import pyvista as pv


def load_anticline_2d() -> pv.StructuredGrid:
    """
    Load 2D anticline mesh.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid.

    """
    from .. import MeshStack2D

    return (
        MeshStack2D(pv.Line([-3.14, 0.0, 0.0], [3.14, 0.0, 0.0], resolution=41))
        .add(0.0)
        .add(lambda x, y, z: np.cos(x) + 1.0, 4, group="Layer 1")
        .add(0.5, 2, group="Layer 2")
        .add(0.5, 2, group="Layer 3")
        .add(0.5, 2, group="Layer 4")
        .add(lambda x, y, z: np.full_like(x, 3.4), 4, group="Layer 5")
        .generate_mesh()
    )


def load_anticline_3d() -> pv.StructuredGrid:
    """
    Load 3D anticline mesh.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid.

    """
    from .. import MeshExtrude

    mesh = MeshExtrude(load_anticline_2d()).add([0.0, 6.28, 0.0], 10).generate_mesh()

    return mesh


def load_concave_polyhedron() -> pv.UnstructuredGrid:
    """
    Load L-shaped concave polyhedron.

    Returns
    -------
    pyvista.UnstructuredGrid
        L-shaped concave unstructured grid.

    """
    from .. import Polygon, Volume

    points = [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 7.0, 0.0],
        [0.0, 7.0, 0.0],
    ]
    polygon = Polygon(points)

    return Volume(polygon, polygon.translate((0.0, 0.0, 1.0)))


def load_half_stadium(resolution=16) -> pv.UnstructuredGrid:
    """
    Load half stadium mesh (single convex polyhedron).

    Returns
    -------
    pyvista.UnstructuredGrid
        Half stadium convex unstructured grid.

    """
    from .. import Polygon, Volume

    theta = np.deg2rad(np.linspace(0.0, 180.0, resolution + 1))
    points = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    points = np.vstack(
        (points[0] + (0.0, -1.0, 0.0), points, points[-1] + (0.0, -1.0, 0.0))
    )
    polygon = Polygon(points)

    return Volume(polygon, polygon.translate((0.0, 0.0, 1.0)))


def load_topographic_terrain() -> pv.StructuredGrid:
    """
    Load 3D mesh following topographic terrain.

    Returns
    -------
    pyvista.StructuredGrid
        Structured grid.

    """
    from .. import MeshStack3D

    terrain = pv.examples.download_crater_topo().extract_subset(
        (500, 900, 400, 800, 0, 0), (10, 10, 1)
    )
    terrain = terrain.cast_to_structured_grid()

    return (
        MeshStack3D(terrain)
        .add(0.0)
        .add(terrain.warp_by_scalar("scalar1of1"), 10, method="log_r")
        .generate_mesh()
    )


def load_well_2d(voronoi: bool = False) -> pv.UnstructuredGrid:
    """
    Load 2D mesh with a well at the center.

    Returns
    -------
    pyvista.UnstructuredGrid
        Unstructured grid.

    """
    from .. import (
        AnnularSector,
        MeshMerge,
        Rectangle,
        Sector,
        SectorRectangle,
        VoronoiMesh2D,
    )

    inner = 0.34
    outer = 0.445
    thickness = outer - inner

    mesh14 = (
        MeshMerge()
        .add(Sector(inner - thickness, 0.0, 90.0, 8), group="Core")
        .add(AnnularSector(inner - thickness, inner, 0.0, 90.0, 1, 8), group="Core")
        .add(AnnularSector(inner, outer, 0.0, 90.0, 1, 8), group="Casing")
        .add(AnnularSector(outer, outer + thickness, 0.0, 90.0, 1, 8), group="Cement")
        .add(AnnularSector(outer + thickness, 0.8, 0.0, 90.0, 2, 8), group="Cement")
        .add(AnnularSector(0.8, 1.0, 0.0, 90.0, 1, 8), group="Cement")
        .add(AnnularSector(1.0, 1.2, 0.0, 90.0, 1, 8), group="Matrix")
        .add(SectorRectangle(1.2, 5.0, 5.0, 8, 4, "log"), group="Matrix")
        .add(Rectangle(5.0, 5.0, 4, 4, center=[5.0, 0.0, 0.0]), group="Matrix")
        .add(Rectangle(5.0, 5.0, 4, 4, center=[5.0, 5.0, 0.0]), group="Matrix")
        .add(Rectangle(5.0, 5.0, 4, 4, center=[0.0, 5.0, 0.0]), group="Matrix")
        .generate_mesh()
    )
    mesh = (
        MeshMerge()
        .add(mesh14)
        .add(mesh14.rotate_z(90.0))
        .add(mesh14.rotate_z(180.0))
        .add(mesh14.rotate_z(270.0))
        .generate_mesh(tolerance=1.0e-4)
    )

    if voronoi:
        mesh = VoronoiMesh2D(mesh).generate_mesh()

    return mesh


def load_well_3d(voronoi: bool = False) -> pv.UnstructuredGrid:
    """
    Load 3D mesh with a well at the center.

    Returns
    -------
    pyvista.UnstructuredGrid
        Unstructured grid.

    """
    from .. import MeshExtrude, extract_cells

    mesh2d = load_well_2d(voronoi)
    mesh2d.points[:, 2] = -30.0
    groups = mesh2d.user_dict["CellGroup"]
    inactive = lambda x: [
        group in {groups["Cement"], groups["Matrix"]} for group in x["CellGroup"]
    ]

    mesh = (
        MeshExtrude(mesh2d)
        .add([0.0, 0.0, 30.0], 10)
        .add([0.0, 0.0, 3.0], 3, group={"Inactive": inactive})
        .generate_mesh()
    )
    mesh = extract_cells(
        mesh, mesh["CellGroup"] != mesh.user_dict["CellGroup"]["Inactive"]
    )
    mesh.point_data.pop("vtkOriginalPointIds", None)
    mesh.cell_data.pop("vtkOriginalCellIds", None)

    return mesh
