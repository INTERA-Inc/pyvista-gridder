import numpy as np
import pytest
import pyvista as pv
from shapely import Polygon

import pvgridder as pvg


pytest.importorskip("gmsh")


@pytest.mark.parametrize(
	"celltype, engine",
	[
		pytest.param("triangle", "gmsh", id="triangle_gmsh"),
		pytest.param("triangle", "occ", id="triangle_occ"),
		pytest.param("quad", "gmsh", id="quad_gmsh"),
		pytest.param("quad", "occ", id="quad_occ"),
	],
)
def test_delaunay2d_base_mesh(celltype, engine):
	"""Test the base mesh and output cell type."""
	polygon = [(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.generate_mesh(celltype=celltype, engine=engine)
		.compute_cell_sizes()
	)
	assert getattr(pv.CellType, celltype.upper()) in mesh.celltypes
	assert np.isclose(np.abs(mesh.cell_data["Area"]).sum(), 9.0)
	assert "PointGroup" in mesh.point_data
	assert "default" in mesh.user_dict["PointGroup"]


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_hole(engine):
	"""Test the base mesh with a hole."""
	domain = Polygon(
		[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
		holes=[[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]],
	)
	mesh = pvg.DelaunayMesh2D(domain).generate_mesh(engine=engine).compute_cell_sizes()
	centers = mesh.cell_centers().points
	inside_hole = (
		(centers[:, 0] > 1.0)
		& (centers[:, 0] < 3.0)
		& (centers[:, 1] > 1.0)
		& (centers[:, 1] < 3.0)
	)
	assert not inside_hole.any()
	assert np.isclose(np.abs(mesh.cell_data["Area"]).sum(), 12.0)


def test_delaunay2d_cellsize():
	"""Test the cellsize option."""
	polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
	mesh = pvg.DelaunayMesh2D(polygon)
	assert list(mesh._get_ring_sizes(mesh._polygon.exterior, None).values()) == [
		4.0,
		4.0,
		4.0,
		4.0,
	]

	cellsize = np.array([0.1, 0.2, 0.3, 0.4])
	mesh = pvg.DelaunayMesh2D(polygon, cellsize=cellsize)
	assert list(mesh._get_ring_sizes(mesh._polygon.exterior, cellsize).values()) == (
		cellsize.tolist()
	)


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_polygon(engine):
	"""Test method pvgridder.DelaunayMesh2D.add_polygon."""
	bmesh = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
	polygon = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
	for boundary_only in (False, True):
		mesh = (
			pvg.DelaunayMesh2D(bmesh)
			.add_polygon(
				polygon,
				boundary_only=boundary_only,
				cellsize=0.5,
				group="polygon",
			)
			.generate_mesh(max_cellsize=0.8, engine=engine)
		)
		group_id = mesh.user_dict["PointGroup"]["polygon"]
		inside = (
			(mesh.points[:, 0] > 1.0 + 1.0e-8)
			& (mesh.points[:, 0] < 3.0 - 1.0e-8)
			& (mesh.points[:, 1] > 1.0 + 1.0e-8)
			& (mesh.points[:, 1] < 3.0 - 1.0e-8)
		)
		assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 16.0)
		assert bool((mesh.point_data["PointGroup"][inside] == group_id).any()) is (
			not boundary_only
		)


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_point(engine):
	"""Test method pvgridder.DelaunayMesh2D.add_point."""
	polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.add_point((2.0, 2.0), cellsize=0.4, group="point")
		.generate_mesh(max_cellsize=1.0, engine=engine)
	)
	point_id = mesh.find_closest_point((2.0, 2.0, 0.0))
	assert np.allclose(mesh.points[point_id], (2.0, 2.0, 0.0))
	assert mesh.point_data["PointGroup"][point_id] == mesh.user_dict["PointGroup"]["point"]


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_point_constraint(engine):
	"""Test option constraint of method pvgridder.DelaunayMesh2D.add_point."""
	polygon = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.add_point((3.0, 3.0), cellsize=0.5, constraint=8, group="point")
		.generate_mesh(max_cellsize=0.5, engine=engine)
	)
	groups = mesh.user_dict["PointGroup"]
	tags = mesh.point_data["PointGroup"]
	point_id = mesh.find_closest_point((3.0, 3.0, 0.0))
	assert tags[point_id] == groups["point"]

	constraint_ids = []
	
	for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
		expected = np.array([3.0 + 0.5 * np.cos(angle), 3.0 + 0.5 * np.sin(angle), 0.0])
		constraint_id = mesh.find_closest_point(expected)
		assert np.allclose(mesh.points[constraint_id], expected)
		assert tags[constraint_id] == groups["default"]
		constraint_ids.append(constraint_id)

	assert len(set(constraint_ids)) == 8


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_polyline(engine):
	"""Test method pvgridder.DelaunayMesh2D.add_polyline."""
	polygon = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.add_polyline(
			[(-1.0, 2.0), (2.0, 2.0), (5.0, 2.0)],
			cellsize=0.4,
			group="line",
		)
		.generate_mesh(max_cellsize=1.0, engine=engine)
	)
	group_id = mesh.user_dict["PointGroup"]["line"]
	on_line = np.isclose(mesh.points[:, 1], 2.0)
	assert np.count_nonzero(mesh.point_data["PointGroup"][on_line] == group_id) > 3


@pytest.mark.parametrize("engine", ["gmsh", "occ"])
def test_delaunay2d_add_circle(engine):
	"""Test method pvgridder.DelaunayMesh2D.add_circle."""
	polygon = [(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)]
	mesh = (
		pvg.DelaunayMesh2D(polygon)
		.add_circle(
			2.0,
			center=(3.0, 3.0),
			cellsize=0.4,
			group="circle",
		)
		.generate_mesh(max_cellsize=1.0, engine=engine)
	)
	group_id = mesh.user_dict["PointGroup"]["circle"]
	radii = np.linalg.norm(mesh.points[:, :2] - (3.0, 3.0), axis=1)
	inside = radii < 2.0 - 1.0e-8
	assert (mesh.point_data["PointGroup"] == group_id).any()
	assert (mesh.point_data["PointGroup"][inside] == group_id).any()
