import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "bmesh, max_depth, min_cellsize, default_group, ignore_groups, n_cells_ref, max_depth_ref, n_point_cells",
    [
        pytest.param("structured_square", 1, None, "default", None, 12, 1, 4, id="base"),
        pytest.param(pv.ImageData(dimensions=(4, 4, 1)), 1, None, "default", None, 12, 1, 4, id="mesh=image_data"),
        pytest.param(pv.RectilinearGrid(range(4), range(4), [0.0]), 1, None, "default", None, 12, 1, 4, id="mesh=rectilinear"),
        pytest.param("structured_square", 2, None, "default", None, 24, 2, 4, id="max_depth=2"),
        pytest.param("structured_square", 1, 1.0, "default", None, 9, 0, 1, id="min_cellsize=1.0"),
        pytest.param("structured_square", 1, 0.5, "default", None, 12, 1, 4, id="min_cellsize=0.5"),
        pytest.param("structured_square", 1, 0.1, "default", None, 36, 3, 4, id="min_cellsize=0.1"),
        pytest.param("structured_square", 1, None, "test", None, 12, 1, 4, id="default_group=test"),
        pytest.param("structured_square", 1, None, "default", ["unused"], 12, 1, 4, id="ignore_groups=unused"),
    ],
)
def test_quadtree_init(bmesh, max_depth, min_cellsize, default_group, ignore_groups, n_cells_ref, max_depth_ref, n_point_cells, request):
    """Test the initialization of the QuadTree class."""
    bmesh = request.getfixturevalue(bmesh) if isinstance(bmesh, str) else bmesh
    qtree = pvg.QuadTree(
        bmesh,
        max_depth=max_depth,
        min_cellsize=min_cellsize,
        default_group=default_group,
        ignore_groups=ignore_groups,
    )
    mesh = qtree.add_point((1.5, 1.5), group="point").generate_mesh(balance=False)
    assert isinstance(qtree.mesh, pv.RectilinearGrid)
    assert qtree.max_depth == max_depth_ref
    assert qtree.ignore_groups == (ignore_groups or [])
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "point").sum() == n_point_cells
    assert (pvg.get_cell_group(mesh) == default_group).sum() == n_cells_ref - n_point_cells
    assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
    "default_group, n_cells_ref, n_layer1_cells, n_layer2_cells, n_layer3_cells, n_point_cells",
    [
        pytest.param("default", 12, 3, 2, 3, 4, id="base"),
        pytest.param("test", 12, 3, 2, 3, 4, id="default_group=test"),
    ],
)
def test_quadtree_init_cell_groups(default_group, n_cells_ref, n_layer1_cells, n_layer2_cells, n_layer3_cells, n_point_cells):
    """Test the initialization of the QuadTree class with cell groups."""
    bmesh = (
        pvg.MeshStack2D(np.arange(4), axis=1)
        .add(0.0)
        .add(1.0, group="layer1")
        .add(1.0, group="layer2")
        .add(1.0, group="layer3")
        .generate_mesh()
    )
    mesh = (
        pvg.QuadTree(bmesh, max_depth=1, default_group=default_group)
        .add_point((1.5, 1.5), group="point")
        .generate_mesh(balance=False)
    )
    cell_groups = pvg.get_cell_group(mesh)
    assert mesh.n_cells == n_cells_ref
    assert (cell_groups == "layer1").sum() == n_layer1_cells
    assert (cell_groups == "layer2").sum() == n_layer2_cells
    assert (cell_groups == "layer3").sum() == n_layer3_cells
    assert (cell_groups == "point").sum() == n_point_cells


@pytest.mark.parametrize(
    "depth, cellsize, n_cells_ref",
    [
        pytest.param(0, None, 5, id="base"),
        pytest.param(1, None, 9, id="depth=1"),
        pytest.param(2, None, 45, id="depth=2"),
        pytest.param(0, 0.25, 45, id="cellsize=0.25"),
    ],
)
def test_quadtree_add_boundary_polygon(depth, cellsize, n_cells_ref, request):
    """Test the method pvgridder.QuadTree.add_boundary_polygon."""
    bmesh = request.getfixturevalue("structured_square")
    boundary_polygon = [(0.0, 1.5), (1.5, 0.0), (3.0, 1.5), (1.5, 3.0)]
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_boundary_polygon(boundary_polygon, depth=depth, cellsize=cellsize)
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref


@pytest.mark.parametrize(
    "radius, center, boundary_only, depth, group, n_cells_ref, n_circle_cells",
    [
        pytest.param(1.0, (1.5, 1.5), False, 1, "circle", 36, 12, id="base"),
        pytest.param(0.5, (1.5, 1.5), False, 1, "circle", 24, 4, id="radius=0.5"),
        pytest.param(1.0, None, False, 1, "circle", 18, 3, id="center=None"),
        pytest.param(1.0, (1.5, 1.5), True, 1, "circle", 33, 9, id="boundary_only=True"),
        pytest.param(1.0, (1.5, 1.5), False, 2, "circle", 102, 52, id="depth=2"),
        pytest.param(1.0, (1.5, 1.5), False, None, "circle", 102, 52, id="depth=None"),
        pytest.param(1.0, (1.5, 1.5), False, 1, None, 36, 0, id="group=None"),
    ],
)
def test_quadtree_add_circle(radius, center, boundary_only, depth, group, n_cells_ref, n_circle_cells, request):
    """Test the method pvgridder.QuadTree.add_circle."""
    bmesh = request.getfixturevalue("structured_square")
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_circle(
            radius=radius,
            center=center,
            boundary_only=boundary_only,
            depth=depth,
            group=group,
        )
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "circle").sum() == n_circle_cells


@pytest.mark.parametrize(
    "point, depth, cellsize, group, n_cells_ref, n_point_cells",
    [
        pytest.param((1.5, 1.5), 1, None, "point", 12, 4, id="base"),
        pytest.param((1.6, 1.6), 1, None, "point", 12, 1, id="point=(1.6,1.6)"),
        pytest.param((1.5, 1.5), 2, None, "point", 24, 4, id="depth=2"),
        pytest.param((1.5, 1.5), None, None, "point", 60, 4, id="depth=None"),
        pytest.param((1.5, 1.5), 1, 0.25, "point", 24, 4, id="cellsize=0.25"),
        pytest.param((1.5, 1.5), 1, 0.03125, "point", 60, 4, id="cellsize=0.03125"),
        pytest.param((1.5, 1.5), 1, None, None, 12, 0, id="group=None"),
    ],
)
def test_quadtree_add_point(point, depth, cellsize, group, n_cells_ref, n_point_cells, request):
    """Test the method pvgridder.QuadTree.add_point."""
    bmesh = request.getfixturevalue("structured_square")
    mesh = (
        pvg.QuadTree(bmesh, max_depth=5)
        .add_point(point, depth=depth, cellsize=cellsize, group=group)
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "point").sum() == n_point_cells

    if cellsize is not None:
        assert mesh.compute_cell_sizes().cell_data["Area"].min() == cellsize**2


@pytest.mark.parametrize(
    "boundary_only, depth, cellsize, group, n_cells_ref, n_polygon_cells",
    [
        pytest.param(False, 1, None, "polygon", 36, 12, id="base"),
        pytest.param(True, 1, None, "polygon", 33, 9, id="boundary_only=True"),
        pytest.param(False, 2, None, "polygon", 132, 60, id="depth=2"),
        pytest.param(False, None, None, "polygon", 132, 60, id="depth=None"),
        pytest.param(False, 1, 0.25, "polygon", 132, 60, id="cellsize=0.25"),
        pytest.param(False, 1, None, None, 36, 0, id="group=None"),
    ],
)
def test_quadtree_add_polygon(boundary_only, depth, cellsize, group, n_cells_ref, n_polygon_cells, request):
    """Test the method pvgridder.QuadTree.add_polygon."""
    bmesh = request.getfixturevalue("structured_square")
    polygon = [(0.0, 1.5), (1.5, 0.0), (3.0, 1.5), (1.5, 3.0)]
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_polygon(
            polygon,
            boundary_only=boundary_only,
            depth=depth,
            cellsize=cellsize,
            group=group,
        )
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "polygon").sum() == n_polygon_cells


@pytest.mark.parametrize(
    "depth, cellsize, group, n_cells_ref, n_line_cells",
    [
        pytest.param(1, None, "polyline", 24, 16, id="base"),
        pytest.param(2, None, "polyline", 72, 28, id="depth=2"),
        pytest.param(None, None, "polyline", 72, 28, id="depth=None"),
        pytest.param(1, 0.25, "polyline", 72, 28, id="cellsize=0.25"),
        pytest.param(1, None, None, 24, 0, id="group=None"),
    ],
)
def test_quadtree_add_polyline(depth, cellsize, group, n_cells_ref, n_line_cells, request):
    """Test the method pvgridder.QuadTree.add_polyline."""
    bmesh = request.getfixturevalue("structured_square")
    polyline = [(0.5, 0.5), (0.5, 1.5), (2.5, 2.5)]
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_polyline(polyline, depth=depth, cellsize=cellsize, group=group)
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "polyline").sum() == n_line_cells


@pytest.mark.parametrize(
    "dx, dy, origin, boundary_only, depth, cellsize, group, n_cells_ref, n_rectangle_cells",
    [
        pytest.param(1.0, 1.5, (1.0, 1.0), False, 2, None, "rectangle", 96, 24, id="base"),
        pytest.param(0.5, 1.5, (1.0, 1.0), False, 2, None, "rectangle", 72, 12, id="dx=0.5"),
        pytest.param(1.0, 1.0, (1.0, 1.0), False, 2, None, "rectangle", 84, 16, id="dy=1.0"),
        pytest.param(1.0, 1.5, None, False, 2, None, "rectangle", 57, 24, id="origin=None"),
        pytest.param(1.0, 1.5, (1.0, 1.0), True, 2, None, "rectangle", 96, 24, id="boundary_only=True"),
        pytest.param(1.0, 1.5, (1.0, 1.0), False, 1, None, "rectangle", 36, 6, id="depth=1"),
        pytest.param(1.0, 1.5, (1.0, 1.0), False, None, None, "rectangle", 96, 24, id="depth=None"),
        pytest.param(1.0, 1.5, (1.0, 1.0), False, 2, 0.5, "rectangle", 36, 6, id="cellsize=0.5"),
        pytest.param(1.0, 1.5, (1.0, 1.0), False, 2, None, None, 96, 0, id="group=None"),
    ],
)
def test_quadtree_add_rectangle(dx, dy, origin, boundary_only, depth, cellsize, group, n_cells_ref, n_rectangle_cells, request):
    """Test the method pvgridder.QuadTree.add_rectangle."""
    bmesh = request.getfixturevalue("structured_square")
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_rectangle(
            dx,
            dy,
            origin=origin,
            boundary_only=boundary_only,
            depth=depth,
            cellsize=cellsize,
            group=group,
        )
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "rectangle").sum() == n_rectangle_cells


@pytest.mark.parametrize(
    "dx, origin, boundary_only, depth, cellsize, group, n_cells_ref, n_square_cells",
    [
        pytest.param(1.0, (1.0, 1.0), False, 2, None, "square", 84, 16, id="base"),
        pytest.param(0.5, (1.0, 1.0), False, 2, None, "square", 48, 4, id="dx=0.5"),
        pytest.param(1.0, None, False, 2, None, "square", 48, 16, id="origin=None"),
        pytest.param(1.0, (1.0, 1.0), True, 2, None, "square", 84, 16, id="boundary_only=True"),
        pytest.param(1.0, (1.0, 1.0), False, 1, None, "square", 36, 4, id="depth=1"),
        pytest.param(1.0, (1.0, 1.0), False, None, None, "square", 84, 16, id="depth=None"),
        pytest.param(1.0, (1.0, 1.0), False, 2, 0.5, "square", 36, 4, id="cellsize=0.5"),
        pytest.param(1.0, (1.0, 1.0), False, 2, None, None, 84, 0, id="group=None"),
    ],
)
def test_quadtree_add_square(dx, origin, boundary_only, depth, cellsize, group, n_cells_ref, n_square_cells, request):
    """Test the method pvgridder.QuadTree.add_square."""
    bmesh = request.getfixturevalue("structured_square")
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_square(
            dx,
            origin=origin,
            boundary_only=boundary_only,
            depth=depth,
            cellsize=cellsize,
            group=group,
        )
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert (pvg.get_cell_group(mesh) == "square").sum() == n_square_cells


@pytest.mark.parametrize(
    "tolerance, balance, conformal, n_cells_ref, n_connections_ref",
    [
        pytest.param(1.0e-8, False, False, 24, 32, id="base"),
        pytest.param(1.0e-6, False, False, 24, 32, id="tolerance=1.0e-6"),
        pytest.param(1.0e-8, True, False, 36, 40, id="balance=True"),
        pytest.param(1.0e-8, False, True, 24, 48, id="conformal=True"),
    ],
)
def test_quadtree_generate_mesh(tolerance, balance, conformal, n_cells_ref, n_connections_ref, request):
    """Test the method pvgridder.QuadTree.generate_mesh."""
    bmesh = request.getfixturevalue("structured_square")
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_point((1.5, 1.5))
        .generate_mesh(tolerance=tolerance, balance=balance, conformal=conformal)
    )
    assert mesh.n_cells == n_cells_ref
    assert pvg.get_connectivity(mesh).n_cells == n_connections_ref
    assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)
