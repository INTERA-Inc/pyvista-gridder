import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "bmesh",
    [
        pytest.param(pv.ImageData(dimensions=(4, 4, 1)), id="image_data"),
        pytest.param(pv.RectilinearGrid(range(3), range(3), [0.0]), id="rectilinear"),
        pytest.param(pvg.Square(3.0, resolution=3), id="structured"),
    ],
)
def test_quadtree_base_mesh(bmesh):
    """Test the base mesh type."""
    qtree = pvg.QuadTree(bmesh)
    mesh = qtree.generate_mesh(balance=False)
    assert isinstance(qtree.mesh, pv.RectilinearGrid)
    assert mesh.n_cells == bmesh.n_cells


@pytest.mark.parametrize(
    "min_cellsize, max_depth_ref",
    [
        pytest.param(1.0, 0, id="min_cellsize_1.0"),
        pytest.param(0.5, 1, id="min_cellsize_0.5"),
        pytest.param(0.1, 3, id="min_cellsize_0.1"),
    ],
)
def test_quadtree_min_cellsize(min_cellsize, max_depth_ref):
    """Test the minimum cell size option."""
    bmesh = pvg.Square(3.0, resolution=3)
    qtree = pvg.QuadTree(bmesh, min_cellsize=min_cellsize)
    assert qtree.max_depth == max_depth_ref


def test_quadtree_cell_group():
    """Test cell groups in base mesh being correctly passed."""
    bmesh = (
        pvg.MeshStack2D(np.arange(4), axis=1)
        .add(0.0)
        .add(1.0, group="layer1")
        .add(1.0, group="layer2")
        .add(1.0, group="layer3")
        .generate_mesh()
    )
    mesh = (
        pvg.QuadTree(bmesh, max_depth=1)
        .add_point((1.5, 1.5), group="point")
        .generate_mesh(balance=False)
    )
    cell_groups = pvg.get_cell_group(mesh)
    assert mesh.extract_cells(cell_groups == "layer1").n_cells == 3
    assert mesh.extract_cells(cell_groups == "layer2").n_cells == 2
    assert mesh.extract_cells(cell_groups == "layer3").n_cells == 3
    assert mesh.extract_cells(cell_groups == "point").n_cells == 4


def test_quadtree_conformal():
    """Test the option conformal."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh, max_depth=1)
        .add_point((1.5, 1.5))
        .generate_mesh(conformal=True)
    )
    connectivity = pvg.get_connectivity(mesh)
    assert connectivity.n_cells == 20


@pytest.mark.parametrize(
    "max_depth, balance, n_cells_ref",
    [
        pytest.param(1, False, 12, id="max_depth_1_balance_false"),
        pytest.param(1, True, 12, id="max_depth_1_balance_true"),
        pytest.param(2, False, 24, id="max_depth_2_balance_false"),
        pytest.param(2, True, 36, id="max_depth_2_balance_true"),
    ],
)
def test_quadtree_max_depth(max_depth, balance, n_cells_ref):
    """Test the options max_depth and balance."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh, max_depth=max_depth)
        .add_point((1.5, 1.5))
        .generate_mesh(balance=balance)
    )
    assert mesh.n_cells == n_cells_ref


@pytest.mark.parametrize(
    "depth, n_cells_ref",
    [
        pytest.param(0, 5, id="depth_0"),
        pytest.param(1, 9, id="depth_1"),
        pytest.param(2, 45, id="depth_2"),
    ],
)
def test_quadtree_add_boundary_polygon(depth, n_cells_ref):
    """Test method pvgridder.QuadTree.add_boundary_polygon."""
    bmesh = pvg.Square(3.0, resolution=3)
    boundary_polygon = [(0.0, 1.5), (1.5, 0.0), (3.0, 1.5), (1.5, 3.0)]
    mesh = (
        pvg.QuadTree(bmesh)
        .add_boundary_polygon(boundary_polygon, depth=depth)
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref



@pytest.mark.parametrize(
    "depth, boundary_only, n_cells_ref",
    [
        pytest.param(1, False, 12, id="depth_1_boundary_only_false"),
        pytest.param(1, True, 9, id="depth_1_boundary_only_true"),
        pytest.param(2, False, 52, id="depth_2_boundary_only_false"),
        pytest.param(2, True, 37, id="depth_2_boundary_only_true"),
    ],
)
def test_quadtree_add_circle(depth, boundary_only, n_cells_ref):
    """Test method pvgridder.QuadTree.add_circle."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh)
        .add_circle(radius=1.0, center=(1.5, 1.5), boundary_only=boundary_only, depth=depth, group="circle")
        .generate_mesh(balance=False)
    )
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "circle").n_cells == n_cells_ref


@pytest.mark.parametrize(
    "depth, n_cells_ref",
    [
        pytest.param(1, 15, id="depth_1"),
        pytest.param(2, 30, id="depth_2"),
    ],
)
def test_quadtree_add_point(depth, n_cells_ref):
    """Test method pvgridder.QuadTree.add_point."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh)
        .add_point((0.5, 0.5), depth=depth, group="point")  # center of a cell, 4 points in group 'point'
        .add_point((1.6, 1.6), depth=depth, group="point")  # 1 point in group 'point'
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == n_cells_ref
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "point").n_cells == 5


@pytest.mark.parametrize(
    "depth, boundary_only, n_cells_ref",
    [
        pytest.param(1, False, 12, id="depth_1_boundary_only_false"),
        pytest.param(1, True, 9, id="depth_1_boundary_only_true"),
        pytest.param(2, False, 60, id="depth_2_boundary_only_false"),
        pytest.param(2, True, 45, id="depth_2_boundary_only_true"),
    ],
)
def test_quadtree_add_polygon(depth, boundary_only, n_cells_ref):
    """Test method pvgridder.QuadTree.add_polygon."""
    bmesh = pvg.Square(3.0, resolution=3)
    polygon = [(0.0, 1.5), (1.5, 0.0), (3.0, 1.5), (1.5, 3.0)]
    mesh = (
        pvg.QuadTree(bmesh)
        .add_polygon(polygon, depth=depth, boundary_only=boundary_only, group="polygon")
        .generate_mesh(balance=False)
    )
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "polygon").n_cells == n_cells_ref


@pytest.mark.parametrize(
    "depth, n_cells_ref",
    [
        pytest.param(1, 16, id="depth_1"),
        pytest.param(2, 28, id="depth_2"),
    ],
)
def test_quadtree_add_polyline(depth, n_cells_ref):
    """Test method pvgridder.QuadTree.add_polyline."""
    bmesh = pvg.Square(3.0, resolution=3)
    polyline = [(0.5, 0.5), (0.5, 1.5), (2.5, 2.5)]
    mesh = (
        pvg.QuadTree(bmesh)
        .add_polyline(polyline, depth=depth, group="polyline")
        .generate_mesh(balance=False)
    )
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "polyline").n_cells == n_cells_ref


def test_quadtree_add_rectangle():
    """Test method pvgridder.QuadTree.add_rectangle."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_rectangle(1.0, 1.5, origin=(1.0, 1.0), group="rectangle")
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == 96
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "rectangle").n_cells == 24


def test_quadtree_add_square():
    """Test method pvgridder.QuadTree.add_square."""
    bmesh = pvg.Square(3.0, resolution=3)
    mesh = (
        pvg.QuadTree(bmesh, max_depth=2)
        .add_square(1.0, origin=(1.0, 1.0), group="square")
        .generate_mesh(balance=False)
    )
    assert mesh.n_cells == 84
    assert mesh.extract_cells(pvg.get_cell_group(mesh) == "square").n_cells == 16
