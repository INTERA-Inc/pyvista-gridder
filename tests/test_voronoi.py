import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
	"axis, preference, default_group, ignore_groups, n_cells_ref",
	[
		pytest.param(2, "cell", "default", None, 9, id="base"),
		pytest.param(0, "cell", "default", None, 9, id="axis=0"),
		pytest.param(1, "cell", "default", None, 9, id="axis=1"),
		pytest.param(2, "point", "default", None, 16, id="preference=point"),
		pytest.param(2, "cell", "test", None, 9, id="default_group=test"),
		pytest.param(2, "cell", "default", ["unused"], 9, id="ignore_groups=unused"),
	],
)
def test_voronoi2d_init(axis, preference, default_group, ignore_groups, n_cells_ref, request):
	"""Test the initialization of the VoronoiMesh2D class."""
	bmesh = request.getfixturevalue("structured_square").copy()
	bmesh.points = np.roll(bmesh.points, axis - 2, axis=1)
	voronoi = pvg.VoronoiMesh2D(
		bmesh,
		axis=axis,
		preference=preference,
		default_group=default_group,
		ignore_groups=ignore_groups,
	)
	mesh = voronoi.generate_mesh()
	assert voronoi.mesh is not bmesh
	assert voronoi.axis == axis
	assert voronoi.preference == preference
	assert voronoi.ignore_groups == (ignore_groups or [])
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == default_group).sum() == n_cells_ref
	assert np.allclose(mesh.points[:, axis], 0.0)
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
	"preference, ignore_groups, n_cells_ref, n_background_cells",
	[
		pytest.param("cell", None, 9, 1, id="base"),
		pytest.param("point", None, 16, 1, id="preference=point"),
		pytest.param("cell", ["background"], 9, 0, id="ignore_groups=background"),
	],
)
def test_voronoi2d_init_cell_groups(preference, ignore_groups, n_cells_ref, n_background_cells, request):
	"""Test the initialization of the VoronoiMesh2D class with groups."""
	bmesh = request.getfixturevalue("structured_square").copy()
	bmesh.cell_data["CellGroup"] = np.zeros(bmesh.n_cells, dtype=int)
	bmesh.point_data["PointGroup"] = np.zeros(bmesh.n_points, dtype=int)
	bmesh.cell_data["CellGroup"][0] = 1
	bmesh.point_data["PointGroup"][0] = 1
	bmesh.user_dict["CellGroup"] = {"default": 0, "background": 1}
	bmesh.user_dict["PointGroup"] = {"default": 0, "background": 1}
	mesh = (
		pvg.VoronoiMesh2D(bmesh, preference=preference, ignore_groups=ignore_groups)
		.generate_mesh()
	)
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "background").sum() == n_background_cells
	assert (pvg.get_cell_group(mesh) == "default").sum() == n_cells_ref - n_background_cells


@pytest.mark.parametrize(
	"mesh_or_points, priority, group, n_points_ref, n_cells_ref, n_added_cells",
	[
		pytest.param(pvg.Square(1.0, resolution=2, origin=(1.0, 1.0)), 0, "added", 25, 12, 4, id="base"),
		pytest.param([[1.25, 1.25], [1.75, 1.25], [1.25, 1.75], [1.75, 1.75]], 0, "added", 28, 13, 4, id="mesh_or_points=array"),
		pytest.param(pv.PolyData([[1.25, 1.25, 0.0], [1.75, 1.25, 0.0], [1.25, 1.75, 0.0], [1.75, 1.75, 0.0]]), 0, "added", 28, 13, 4, id="mesh_or_points=polydata"),
		pytest.param(pvg.Square(1.0, resolution=2, origin=(1.0, 1.0)), 1, "added", 25, 12, 4, id="priority=1"),
		pytest.param(pvg.Square(1.0, resolution=2, origin=(1.0, 1.0)), 0, None, 25, 12, 0, id="group=None"),
	],
)
def test_voronoi2d_add(mesh_or_points, priority, group, n_points_ref, n_cells_ref, n_added_cells, request):
	"""Test the method pvgridder.VoronoiMesh2D.add."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.VoronoiMesh2D(bmesh)
		.add(mesh_or_points, priority=priority, group=group)
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "added").sum() == n_added_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
	"priority, n_points_ref, n_cells_ref, n_added_cells, n_existing_cells",
	[
		pytest.param(2, 25, 12, 4, 0, id="base"),
		pytest.param(0, 16, 9, 0, 1, id="priority=0"),
		pytest.param(1, 25, 12, 4, 0, id="priority=1"),
		pytest.param(-1, 28, 13, 4, 1, id="priority=-1"),
		pytest.param(-2, 25, 12, 4, 0, id="priority=-2"),
	],
)
def test_voronoi2d_add_priority(priority, n_points_ref, n_cells_ref, n_added_cells, n_existing_cells, request):
	"""Test the method pvgridder.VoronoiMesh2D.add with overlapping items."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.VoronoiMesh2D(bmesh)
		.add(pvg.Square(1.0, resolution=1, origin=(1.0, 1.0)), priority=1, group="existing")
		.add(pvg.Square(1.0, resolution=2, origin=(1.0, 1.0)), priority=priority, group="added")
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "added").sum() == n_added_cells
	assert (pvg.get_cell_group(mesh) == "existing").sum() == n_existing_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
	"radius, constraint_radius, resolution, center, plain, priority, group, n_points_ref, n_cells_ref, n_circle_cells",
	[
		pytest.param(0.5, None, 8, (1.5, 1.5), False, 0, "circle", 45, 32, 16, id="base"),
		pytest.param(0.3, None, 8, (1.5, 1.5), False, 0, "circle", 45, 32, 16, id="radius=0.3"),
		pytest.param(0.5, 0.5, 8, (1.5, 1.5), False, 0, "circle", 29, 16, 8, id="constraint_radius=0.5"),
		pytest.param(0.5, 0.6, 8, (1.5, 1.5), False, 0, "circle", 45, 32, 16, id="constraint_radius=0.6"),
		pytest.param(0.5, None, 12, (1.5, 1.5), False, 0, "circle", 57, 44, 24, id="resolution=12"),
		pytest.param(0.5, None, [0.0, 0.25, 0.5, 0.75, 1.0], (1.5, 1.5), False, 0, "circle", 33, 20, 8, id="resolution=array"),
		pytest.param(0.5, None, 8, (1.0, 1.0), False, 0, "circle", 35, 29, 16, id="center=(1.0,1.0)"),
		pytest.param(0.5, None, 8, (1.5, 1.5), True, 0, "circle", 36, 17, 1, id="plain=True"),
		pytest.param(0.5, None, 8, (1.5, 1.5), False, 1, "circle", 45, 32, 16, id="priority=1"),
		pytest.param(0.5, None, 8, (1.5, 1.5), False, 0, None, 45, 32, 0, id="group=None"),
	],
)
def test_voronoi2d_add_circle(radius, constraint_radius, resolution, center, plain, priority, group, n_points_ref, n_cells_ref, n_circle_cells, request):
	"""Test the method pvgridder.VoronoiMesh2D.add_circle."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.VoronoiMesh2D(bmesh)
		.add_circle(
			radius,
			constraint_radius=constraint_radius,
			resolution=resolution,
			center=center,
			plain=plain,
			priority=priority,
			group=group,
		)
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "circle").sum() == n_circle_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
	"width, preference, padding, constraint, resolution, priority, group, n_points_ref, n_cells_ref, n_line_cells",
	[
		pytest.param(0.3, "cell", None, 1, 2, 0, "line", 54, 26, 6, id="base"),
		pytest.param(0.5, "cell", None, 1, 2, 0, "line", 38, 18, 6, id="width=0.5"),
		pytest.param(0.3, "point", None, 1, 2, 0, "line", 39, 20, 4, id="preference=point"),
		pytest.param(0.3, "cell", 0.1, 1, 2, 0, "line", 58, 28, 6, id="padding=0.1"),
		pytest.param(0.3, "cell", None, 0, 2, 0, "line", 42, 20, 6, id="constraint=0"),
		pytest.param(0.3, "cell", None, (0, 2), 2, 0, "line", 48, 23, 6, id="constraint=(0,2)"),
		pytest.param(0.3, "cell", None, 1, 3, 0, "line", 64, 32, 9, id="resolution=3"),
		pytest.param(0.3, "cell", None, 1, [0.0, 0.25, 1.0], 0, "line", 54, 26, 6, id="resolution=array"),
		pytest.param(0.3, "cell", None, 1, 2, -2, "line", 54, 26, 6, id="priority=-2"),
		pytest.param(0.3, "cell", None, 1, 2, 0, None, 54, 26, 0, id="group=None"),
	],
)
def test_voronoi2d_add_polyline(width, preference, padding, constraint, resolution, priority, group, n_points_ref, n_cells_ref, n_line_cells, request):
	"""Test the method pvgridder.VoronoiMesh2D.add_polyline."""
	bmesh = request.getfixturevalue("structured_square")
	line = [(0.5, 1.0, 0.0), (1.5, 1.5, 0.0), (2.5, 1.0, 0.0)]
	mesh = (
		pvg.VoronoiMesh2D(bmesh)
		.add_polyline(
			line,
			width=width,
			preference=preference,
			padding=padding,
			constraint=constraint,
			resolution=resolution,
			priority=priority,
			group=group,
		)
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "line").sum() == n_line_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.parametrize(
	"infinity, min_length, tolerance, qhull_options, orientation, n_points_ref, n_cells_ref",
	[
		pytest.param(None, 1.0e-4, 1.0e-8, "Qc QJ", "CCW", 16, 9, id="base"),
		pytest.param(100.0, 1.0e-4, 1.0e-8, "Qc QJ", "CCW", 16, 9, id="infinity=100.0"),
		pytest.param(None, 0.0, 1.0e-8, "Qc QJ", "CCW", 16, 9, id="min_length=0.0"),
		pytest.param(None, 0.1, 1.0e-8, "Qc QJ", "CCW", 16, 9, id="min_length=0.1"),
		pytest.param(None, 1.0e-4, 1.0e-6, "Qc QJ", "CCW", 16, 9, id="tolerance=1.0e-6"),
		pytest.param(None, 1.0e-4, 1.0e-8, "Qbb Qc Qz", "CCW", 16, 9, id="qhull_options=Qbb Qc Qz"),
		pytest.param(None, 1.0e-4, 1.0e-8, "Qc QJ", "CW", 16, 9, id="orientation=CW"),
	],
)
def test_voronoi2d_generate_mesh(infinity, min_length, tolerance, qhull_options, orientation, n_points_ref, n_cells_ref, request):
	"""Test the method pvgridder.VoronoiMesh2D.generate_mesh."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.VoronoiMesh2D(bmesh)
		.generate_mesh(
			infinity=infinity,
			min_length=min_length,
			tolerance=tolerance,
			qhull_options=qhull_options,
			orientation=orientation,
		)
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "default").sum() == n_cells_ref
	assert np.allclose(mesh.cell_data["CellCenter"][:, 0], bmesh.cell_centers().points[:, 0])
	assert np.allclose(mesh.cell_data["CellCenter"][:, 1], bmesh.cell_centers().points[:, 1])
	assert np.allclose(mesh.cell_data["CellCenter"][:, 2], 0.0)
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)


@pytest.mark.xfail(strict=True, reason="VoronoiMesh2D currently generates mixed polygon winding for both orientations")
@pytest.mark.parametrize(
	"orientation, n_cells_ref",
	[
		pytest.param("CCW", 9, id="base"),
		pytest.param("CW", 9, id="orientation=CW"),
	],
)
def test_voronoi2d_generate_mesh_orientation(orientation, n_cells_ref, request):
	"""Test the method pvgridder.VoronoiMesh2D.generate_mesh with orientation."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = pvg.VoronoiMesh2D(bmesh).generate_mesh(orientation=orientation)
	assert mesh.n_cells == n_cells_ref

	for connectivity in pvg.get_cell_connectivity(mesh):
		points = mesh.points[connectivity]
		signed_area = np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))
		assert signed_area > 0.0 if orientation == "CCW" else signed_area < 0.0