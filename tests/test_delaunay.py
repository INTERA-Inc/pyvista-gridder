import numpy as np
import pytest

import pvgridder as pvg


pytest.importorskip("gmsh")


@pytest.mark.parametrize(
    "polygon, cellsize, default_group, n_points_ref",
    [
        pytest.param("polygon_square", None, "default", 31, id="base"),
        pytest.param("polygon_square_with_hole", None, "default", 36, id="polygon-with-hole"),
        pytest.param("polygon_square", 0.5, "default", 99, id="cellsize=0.5"),
        pytest.param("polygon_square", None, "test", 31, id="default_group=test"),
    ],
)
def test_delaunay2d_init(polygon, cellsize, default_group, n_points_ref, request):
    """Test the initialization of the DelaunayMesh2D class."""
    polygon = request.getfixturevalue(polygon)
    mesh = (
		pvg.DelaunayMesh2D(polygon, cellsize=cellsize, default_group=default_group)
        .generate_mesh(max_cellsize=1.0, engine="gmsh")
    )
    assert mesh.n_points == n_points_ref
    assert (pvg.get_point_group(mesh) == default_group).sum() == n_points_ref
    assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), polygon.area)


@pytest.mark.parametrize(
    "engine, cellsize, constraint, growth, n_points_ref",
    [
        pytest.param("gmsh", 0.4, 0, 1.0, 42, id="base"),
        pytest.param("gmsh", None, 0, 1.0, 30, id="cellsize=None"),
        pytest.param("gmsh", 0.2, 0, 1.0, 49, id="cellsize=0.2"),
        pytest.param("gmsh", 0.4, 6, 1.0, 53, id="constraint=6"),
        pytest.param("gmsh", 0.4, 6, 1.05, 110, id="growth=1.05"),
        pytest.param("occ", 0.4, 0, 1.0, 42, id="engine=occ"),
    ],
)
def test_delaunay2d_add_point(engine, cellsize, constraint, growth, n_points_ref, request):
	"""Test the method pvgridder.DelaunayMesh2D.add_point."""
	polygon = request.getfixturevalue("polygon_square")
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.add_point(
			(2.0, 2.0),
            cellsize=cellsize,
            constraint=constraint,
            growth=growth,
            group="point",
        )
		.generate_mesh(max_cellsize=1.0, engine=engine)
	)
	assert mesh.n_points == n_points_ref
	assert (pvg.get_point_group(mesh) == "point").sum() == 1


@pytest.mark.parametrize(
    "engine, boundary_only, cellsize, growth, hole, n_points_ref, n_polygon_points",
    [
        pytest.param("gmsh", False, 0.5, 1.0, False, 65, 31, id="base"),
        pytest.param("gmsh", True, 0.5, 1.0, False, 65, 16, id="boundary_only=True"),
        pytest.param("gmsh", False, None, 1.0, False, 40, 12, id="cellsize=None"),
        pytest.param("gmsh", False, 0.2, 1.0, False, 205, 144, id="cellsize=0.2"),
        pytest.param("gmsh", False, 0.5, 1.05, False, 91, 31, id="growth=1.05"),
        pytest.param("gmsh", False, 0.5, 1.0, True, 50, 16, id="hole=True"),
        pytest.param("occ", False, 0.5, 1.0, False, 59, 31, id="engine=occ"),
    ],
)
def test_delaunay2d_add_polygon(engine, boundary_only, cellsize, growth, hole, n_points_ref, n_polygon_points, request):
    """Test the method pvgridder.DelaunayMesh2D.add_polygon."""
    polygon = request.getfixturevalue("polygon_square")
    mesh = (
        pvg.DelaunayMesh2D(polygon)
        .add_polygon(
            [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
            boundary_only=boundary_only,
            cellsize=cellsize,
            growth=growth,
            hole=hole,
            group="polygon",
        )
        .generate_mesh(max_cellsize=1.0, engine=engine)
    )
    assert mesh.n_points == n_points_ref
    assert (pvg.get_point_group(mesh) == "polygon").sum() == n_polygon_points

    if hole:
        assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), polygon.area - 4.0)


@pytest.mark.parametrize(
    "engine, cellsize, growth, n_points_ref, n_line_points",
    [
        pytest.param("gmsh", 0.5, 1.0, 61, 9, id="base"),
        pytest.param("gmsh", None, 1.0, 31, 5, id="cellsize=None"),
        pytest.param("gmsh", 0.2, 1.0, 156, 21, id="cellsize=0.2"),
        pytest.param("gmsh", 0.5, 1.05, 93, 9, id="growth=1.05"),
        pytest.param("occ", 0.5, 1.0, 61, 9, id="engine=occ"),
    ],
)
def test_delaunay2d_add_polyline(engine, cellsize, growth, n_points_ref, n_line_points, request):
    """Test the method pvgridder.DelaunayMesh2D.add_polyline."""
    polygon = request.getfixturevalue("polygon_square")
    mesh = (
	    pvg.DelaunayMesh2D(polygon)
	    .add_polyline(
	        [(-1.0, 2.0), (2.0, 2.0), (5.0, 2.0)],
		    cellsize=cellsize,
            growth=growth,
		    group="line",
	    )
	    .generate_mesh(max_cellsize=1.0, engine=engine)
    )
    assert mesh.n_points == n_points_ref
    assert (pvg.get_point_group(mesh) == "line").sum() == n_line_points


@pytest.mark.parametrize(
    "engine, center, boundary_only, cellsize, growth, hole, n_points_ref, n_circle_points",
    [
        pytest.param("gmsh", (2.0, 2.0), False, 0.5, 1.0, False, 60, 27, id="base"),
        pytest.param("gmsh", (3.0, 3.0), False, 0.5, 1.0, False, 64, 28, id="center=(3.0,3.0)"),
        pytest.param("gmsh", (2.0, 2.0), True, 0.5, 1.0, False, 60, 13, id="boundary_only=True"),
        pytest.param("gmsh", (2.0, 2.0), False, 0.2, None, False, 167, 123, id="cellsize=None"),
        pytest.param("gmsh", (2.0, 2.0), False, 0.2, 1.0, False, 167, 123, id="cellsize=0.2"),
        pytest.param("gmsh", (2.0, 2.0), False, 0.5, 1.05, False, 107, 27, id="growth=1.05"),
        pytest.param("gmsh", (2.0, 2.0), False, 0.5, 1.0, True, 46, 13, id="hole=True"),
        pytest.param("gmsh", (2.0, 2.0), False, None, 1.0, True, 162, 64, id="hole=True,cellsize=None"),
    ]
)
def test_delaunay2d_add_circle(engine, center, boundary_only, cellsize, growth, hole, n_points_ref, n_circle_points, request):
    """Test the method pvgridder.DelaunayMesh2D.add_circle."""
    polygon = request.getfixturevalue("polygon_square") 
    mesh = (
        pvg.DelaunayMesh2D(polygon)
        .add_circle(
            1.0,
            center=center,
            boundary_only=boundary_only,
            cellsize=cellsize,
            growth=growth,
            hole=hole,
            group="circle",
        )
        .generate_mesh(max_cellsize=1.0, engine=engine)
    )
    assert mesh.n_points == n_points_ref
    assert (pvg.get_point_group(mesh) == "circle").sum() == n_circle_points

    if hole:
        radius = 1.0
        resolution = max(np.ceil(2.0 * np.pi * radius / cellsize), 3) if cellsize else 64
        unit_circle = pvg.Circle(resolution=int(resolution)).compute_cell_sizes()
        assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), polygon.area - unit_circle.cell_data["Area"].sum())


@pytest.mark.parametrize(
    "min_cellsize, max_cellsize, celltype, algorithm, optimization, engine, n_points_ref",
    [
        pytest.param(0.1, 2.0, "triangle", 6, None, "gmsh", 99, id="base"),
        pytest.param(None, 2.0, "triangle", 6, None, "gmsh", 99, id="min_cellsize=None"),
        pytest.param(1.0, 2.0, "triangle", 6, None, "gmsh", 31, id="min_cellsize=1.0"),
        pytest.param(0.1, None, "triangle", 6, None, "gmsh", 99, id="max_cellsize=None"),
        pytest.param(0.1, 0.4, "triangle", 6, None, "gmsh", 144, id="max_cellsize=0.4"),
        pytest.param(0.1, 2.0, "quad", 6, None, "gmsh", 98, id="celltype=quad"),
        pytest.param(0.1, 2.0, "quad", 8, None, "gmsh", 81, id="celltype=quad,algorithm=8"),
        pytest.param(0.1, 2.0, "triangle", 6, "Netgen", "gmsh", 99, id="algorithm=Netgen"),
        pytest.param(0.1, 2.0, "triangle", 6, "Laplace2D", "gmsh", 99, id="algorithm=Laplace2D"),
        pytest.param(0.1, 2.0, "triangle", 6, "Relocate2D", "gmsh", 99, id="algorithm=Relocate2D"),
        pytest.param(0.1, 2.0, "triangle", 6, None, "occ", 97, id="engine=occ"),
    ]
)
def test_delaunay2d_generate_mesh(min_cellsize, max_cellsize, celltype, algorithm, optimization, engine, n_points_ref, request):
    """Test the method pvgridder.DelaunayMesh2D.generate_mesh."""
    polygon = request.getfixturevalue("polygon_square")
    mesh = (
        pvg.DelaunayMesh2D(polygon, cellsize=0.5)
        .generate_mesh(
            min_cellsize=min_cellsize,
            max_cellsize=max_cellsize,
            celltype=celltype,
            algorithm=algorithm,
            optimization=optimization,
            engine=engine,
        )
    )
    assert mesh.n_points == n_points_ref
    assert (mesh.celltypes == 5 if celltype == "triangle" else mesh.celltypes == 9).all()
	