import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
	"bmesh, axis, bottom_up, default_group, ignore_groups, n_points_ref, n_cells_ref",
	[
		pytest.param(np.arange(4.0), 2, True, "default", None, 12, 6, id="base"),
		pytest.param(pv.lines_from_points(np.column_stack((np.arange(4.0), np.zeros((4, 2))))), 2, True, "default", None, 12, 6, id="mesh=lines"),
		pytest.param(pv.MultipleLines(np.column_stack((np.arange(4.0), np.zeros((4, 2))))), 2, True, "default", None, 12, 6, id="mesh=polyline"),
		pytest.param(np.arange(4.0), 0, True, "default", None, 12, 6, id="axis=0"),
		pytest.param(np.arange(4.0), 1, True, "default", None, 12, 6, id="axis=1"),
		pytest.param(np.arange(4.0), 2, False, "default", None, 12, 6, id="bottom_up=False"),
		pytest.param(np.arange(4.0), 2, True, "test", None, 12, 6, id="default_group=test"),
		pytest.param(np.arange(4.0), 2, True, "default", ["unused"], 12, 6, id="ignore_groups=unused"),
	],
)
def test_meshstack2d_init(bmesh, axis, bottom_up, default_group, ignore_groups, n_points_ref, n_cells_ref):
	"""Test the initialization of the MeshStack2D class."""
	stack = pvg.MeshStack2D(bmesh, axis=axis, bottom_up=bottom_up, default_group=default_group, ignore_groups=ignore_groups)
	mesh = stack.add(0.0).add(2.0, resolution=2).generate_mesh()
	assert stack.axis == axis
	assert stack.bottom_up == bottom_up
	assert stack.ignore_groups == (ignore_groups or [])
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == default_group).sum() == n_cells_ref
	assert np.allclose(np.unique(mesh.cell_data["CellCenter"][:, axis]), [0.5, 1.5] if bottom_up else [-1.5, -0.5])
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 6.0)


@pytest.mark.parametrize(
	"bmesh, axis, bottom_up, default_group, ignore_groups, n_points_ref, n_cells_ref",
	[
		pytest.param("structured_square", 2, True, "default", None, 48, 18, id="base"),
		pytest.param(pv.ImageData(dimensions=(4, 4, 1)), 2, True, "default", None, 48, 18, id="mesh=image_data"),
		pytest.param(pv.RectilinearGrid(range(4), range(4), [0.0]), 2, True, "default", None, 48, 18, id="mesh=rectilinear"),
		pytest.param(pvg.Square(3.0, resolution=3).cast_to_unstructured_grid(), 2, True, "default", None, 48, 18, id="mesh=unstructured"),
		pytest.param("structured_square", 0, True, "default", None, 48, 18, id="axis=0"),
		pytest.param("structured_square", 1, True, "default", None, 48, 18, id="axis=1"),
		pytest.param("structured_square", 2, False, "default", None, 48, 18, id="bottom_up=False"),
		pytest.param("structured_square", 2, True, "test", None, 48, 18, id="default_group=test"),
		pytest.param("structured_square", 2, True, "default", ["unused"], 48, 18, id="ignore_groups=unused"),
	],
)
def test_meshstack3d_init(bmesh, axis, bottom_up, default_group, ignore_groups, n_points_ref, n_cells_ref, request):
	"""Test the initialization of the MeshStack3D class."""
	bmesh = request.getfixturevalue(bmesh).copy() if isinstance(bmesh, str) else bmesh.copy()

	if axis != 2:
		bmesh.points = np.roll(bmesh.points, axis - 2, axis=1)
		bmesh.dimensions = np.roll(bmesh.dimensions, axis - 2)
		bmesh.cell_data["CellCenter"] = np.roll(bmesh.cell_data["CellCenter"], axis - 2, axis=1)

	stack = pvg.MeshStack3D(bmesh, axis=axis, bottom_up=bottom_up, default_group=default_group, ignore_groups=ignore_groups)
	mesh = stack.add(0.0).add(2.0, resolution=2).generate_mesh()
	assert stack.mesh is not bmesh
	assert stack.axis == axis
	assert stack.bottom_up == bottom_up
	assert stack.ignore_groups == (ignore_groups or [])
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == default_group).sum() == n_cells_ref
	assert np.allclose(np.unique(mesh.cell_data["CellCenter"][:, axis]), [0.5, 1.5] if bottom_up else [-1.5, -0.5])
	assert np.isclose(np.abs(mesh.compute_cell_sizes().cell_data["Volume"]).sum(), 18.0)


@pytest.mark.parametrize(
	"arg, resolution, method, extrapolation, group, n_points_ref, n_cells_ref, n_layer_cells",
	[
		pytest.param(3.0, 2, None, None, "layer", 12, 6, 6, id="base"),
		pytest.param(lambda x, y, z: np.full_like(x, 3.0), 2, None, None, "layer", 12, 6, 6, id="arg=callable"),
		pytest.param(np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 3.0]]), 2, None, None, "layer", 12, 6, 6, id="arg=array"),
		pytest.param(pv.Line((0.0, 0.0, 3.0), (3.0, 0.0, 3.0)), 2, None, None, "layer", 12, 6, 6, id="arg=polydata"),
		pytest.param(3.0, None, None, None, "layer", 8, 3, 3, id="resolution=None"),
		pytest.param(3.0, 3, None, None, "layer", 16, 9, 9, id="resolution=3"),
		pytest.param(3.0, [0.0, 0.25, 1.0], None, None, "layer", 12, 6, 6, id="resolution=array"),
		pytest.param(3.0, 2, "log", None, "layer", 12, 6, 6, id="method=log"),
		pytest.param(3.0, 2, "log_r", None, "layer", 12, 6, 6, id="method=log_r"),
		pytest.param(3.0, 2, None, "nearest", "layer", 12, 6, 6, id="extrapolation=nearest"),
		pytest.param(3.0, 2, None, None, None, 12, 6, 0, id="group=None"),
		pytest.param(3.0, 2, None, None, {"layer": [0, 2]}, 12, 6, 4, id="group=mapping"),
	],
)
def test_meshstack2d_add(arg, resolution, method, extrapolation, group, n_points_ref, n_cells_ref, n_layer_cells):
	"""Test the method pvgridder.MeshStack2D.add."""
	mesh = (
		pvg.MeshStack2D(np.arange(4.0))
		.add(0.0)
		.add(arg, resolution=resolution, method=method, extrapolation=extrapolation, group=group)
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_layer_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 9.0)
	assert np.isclose(mesh.points[:, 2].max(), 3.0)


@pytest.mark.parametrize(
	"arg, resolution, method, extrapolation, group, n_points_ref, n_cells_ref, n_layer_cells",
	[
		pytest.param(3.0, 2, None, None, "layer", 48, 18, 18, id="base"),
		pytest.param(lambda x, y, z: np.full_like(x, 3.0), 2, None, None, "layer", 48, 18, 18, id="arg=callable"),
		pytest.param(np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 3.0], [0.0, 3.0, 3.0]]), 2, None, None, "layer", 48, 18, 18, id="arg=array"),
		pytest.param(pv.Plane(center=(1.5, 1.5, 3.0), i_size=3.0, j_size=3.0), 2, None, None, "layer", 48, 18, 18, id="arg=polydata"),
		pytest.param(3.0, None, None, None, "layer", 32, 9, 9, id="resolution=None"),
		pytest.param(3.0, 3, None, None, "layer", 64, 27, 27, id="resolution=3"),
		pytest.param(3.0, [0.0, 0.25, 1.0], None, None, "layer", 48, 18, 18, id="resolution=array"),
		pytest.param(3.0, 2, "log", None, "layer", 48, 18, 18, id="method=log"),
		pytest.param(3.0, 2, "log_r", None, "layer", 48, 18, 18, id="method=log_r"),
		pytest.param(3.0, 2, None, "nearest", "layer", 48, 18, 18, id="extrapolation=nearest"),
		pytest.param(3.0, 2, None, None, None, 48, 18, 0, id="group=None"),
		pytest.param(3.0, 2, None, None, {"layer": [0, 2]}, 48, 18, 4, id="group=mapping"),
	],
)
def test_meshstack3d_add(arg, resolution, method, extrapolation, group, n_points_ref, n_cells_ref, n_layer_cells, request):
	"""Test the method pvgridder.MeshStack3D.add."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.MeshStack3D(bmesh)
		.add(0.0)
		.add(arg, resolution=resolution, method=method, extrapolation=extrapolation, group=group)
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_layer_cells
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Volume"].sum(), 27.0)
	assert np.isclose(mesh.points[:, 2].max(), 3.0)


@pytest.mark.parametrize(
	"priority, thickness, n_cells_ref, lower_ref, upper_ref",
	[
		pytest.param(2, 1.0, 3, 0.25, 1.25, id="base"),
		pytest.param(0, 1.0, 3, 1.0, 2.0, id="priority=0"),
		pytest.param(1, 1.0, 3, 0.25, 1.25, id="priority=1"),
		pytest.param(2, 0.1, 3, 1.0, 1.25, id="thickness=0.1"),
	],
)
def test_meshstack2d_add_priority(priority, thickness, n_cells_ref, lower_ref, upper_ref):
	"""Test the method pvgridder.MeshStack2D.add with overlapping layers."""
	mesh = (
		pvg.MeshStack2D(np.arange(4.0))
		.add(1.0, priority=1)
		.add(0.25, priority=priority, thickness=thickness, group="layer")
		.generate_mesh()
	)
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_cells_ref
	assert np.isclose(mesh.points[:, 2].min(), lower_ref)
	assert np.isclose(mesh.points[:, 2].max(), upper_ref)
	assert np.allclose(mesh.cell_data["CellCenter"][:, 2], 0.5 * (lower_ref + upper_ref))
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 3.0 * (upper_ref - lower_ref))


@pytest.mark.parametrize(
	"priority, thickness, n_cells_ref, lower_ref, upper_ref",
	[
		pytest.param(2, 1.0, 9, 0.25, 1.25, id="base"),
		pytest.param(0, 1.0, 9, 1.0, 2.0, id="priority=0"),
		pytest.param(1, 1.0, 9, 0.25, 1.25, id="priority=1"),
		pytest.param(2, 0.1, 9, 1.0, 1.25, id="thickness=0.1"),
	],
)
def test_meshstack3d_add_priority(priority, thickness, n_cells_ref, lower_ref, upper_ref, request):
	"""Test the method pvgridder.MeshStack3D.add with overlapping layers."""
	bmesh = request.getfixturevalue("structured_square")
	mesh = (
		pvg.MeshStack3D(bmesh)
		.add(1.0, priority=1)
		.add(0.25, priority=priority, thickness=thickness, group="layer")
		.generate_mesh()
	)
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_cells_ref
	assert np.isclose(mesh.points[:, 2].min(), lower_ref)
	assert np.isclose(mesh.points[:, 2].max(), upper_ref)
	assert np.allclose(mesh.cell_data["CellCenter"][:, 2], 0.5 * (lower_ref + upper_ref))
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Volume"].sum(), 9.0 * (upper_ref - lower_ref))


@pytest.mark.parametrize(
	"angles, point, resolution, group, n_points_ref, n_cells_ref, n_plane_cells",
	[
		pytest.param((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), 2, "plane", 48, 18, 18, id="base"),
		pytest.param((10.0, 0.0, 0.0), (0.0, 0.0, 3.0), 2, "plane", 48, 18, 18, id="angles=x"),
		pytest.param((0.0, 10.0, 0.0), (0.0, 0.0, 3.0), 2, "plane", 48, 18, 18, id="angles=y"),
		pytest.param((0.0, 0.0, 30.0), (0.0, 0.0, 3.0), 2, "plane", 48, 18, 18, id="angles=z"),
		pytest.param((0.0, 0.0, 0.0), (1.0, 1.0, 4.0), 2, "plane", 48, 18, 18, id="point=(1.0,1.0,4.0)"),
		pytest.param((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), 3, "plane", 64, 27, 27, id="resolution=3"),
		pytest.param((0.0, 0.0, 0.0), (0.0, 0.0, 3.0), 2, None, 48, 18, 0, id="group=None"),
	],
)
def test_meshstack3d_add_plane(angles, point, resolution, group, n_points_ref, n_cells_ref, n_plane_cells, request):
	"""Test the method pvgridder.MeshStack3D.add_plane."""
	bmesh = request.getfixturevalue("structured_square")
	stack = pvg.MeshStack3D(bmesh).add(0.0).add_plane(angles, point, resolution=resolution, group=group)
	mesh = stack.generate_mesh()
	expected = point[2] + np.sin(np.deg2rad(angles[1])) * (bmesh.points[:, 0] - point[0]) - np.sin(np.deg2rad(angles[0])) * (bmesh.points[:, 1] - point[1])
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "plane").sum() == n_plane_cells
	assert np.allclose(stack.items[-1].mesh.points[:, 2], expected)
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Volume"].sum(), 9.0 * expected.mean())


@pytest.mark.parametrize(
	"mesh_or_resolution, n_points_ref, n_cells_ref",
	[
		pytest.param(2, 7, 5, id="base"),
		pytest.param(4, 9, 7, id="mesh_or_resolution=4"),
		pytest.param(pv.Line((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), resolution=2), 7, 5, id="mesh_or_resolution=polydata"),
	],
)
def test_meshstack2d_set_transition(mesh_or_resolution, n_points_ref, n_cells_ref):
	"""Test the method pvgridder.MeshStack2D.set_transition."""
	pytest.importorskip("gmsh")
	mesh = (
		pvg.MeshStack2D(np.arange(4.0), axis=1)
		.add(0.0)
		.set_transition(mesh_or_resolution)
		.add(1.0, group="transition")
		.generate_mesh()
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "transition").sum() == n_cells_ref
	assert (mesh.cell_data["ColumnId"] == -1).all()
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 3.0)


@pytest.mark.parametrize(
	"n_layers, n_points_ref, n_cells_ref",
	[
		pytest.param(1, 8, 3, id="base"),
		pytest.param(2, 12, 6, id="n_layers=2"),
		pytest.param(3, 16, 9, id="n_layers=3"),
	],
)
def test_meshstack2d_generate_mesh(n_layers, n_points_ref, n_cells_ref):
	"""Test the method pvgridder.MeshStack2D.generate_mesh."""
	stack = pvg.MeshStack2D(np.arange(4.0)).add(0.0)

	for _ in range(n_layers):
		stack.add(1.0, group="layer")

	mesh = stack.generate_mesh()
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_cells_ref
	assert np.array_equal(mesh.cell_data["ColumnId"], np.tile(np.arange(3), n_layers))
	assert np.array_equal(mesh.cell_data["LayerId"], np.repeat(np.arange(n_layers), 3))
	assert np.array_equal(mesh.cell_data["StackItem"], np.repeat(np.arange(n_layers), 3))
	assert np.allclose(mesh.cell_data["CellCenter"][:, 2], mesh.cell_data["LayerId"] + 0.5)
	assert np.isclose(mesh.compute_cell_sizes().cell_data["Area"].sum(), 3.0 * n_layers)


@pytest.mark.parametrize(
	"bmesh, tolerance, invert_polyhedron_faces, n_points_ref, n_cells_ref",
	[
		pytest.param(pv.Polygon(n_sides=5).extract_cells_by_type(pv.CellType.POLYGON).cast_to_unstructured_grid(), 1.0e-8, True, 15, 2, id="base"),
		pytest.param("structured_square", 1.0e-8, True, 48, 18, id="mesh=structured"),
		pytest.param(pv.Polygon(n_sides=5).extract_cells_by_type(pv.CellType.POLYGON).cast_to_unstructured_grid(), 1.0e-6, True, 15, 2, id="tolerance=1.0e-6"),
		pytest.param(pv.Polygon(n_sides=5).extract_cells_by_type(pv.CellType.POLYGON).cast_to_unstructured_grid(), 1.0e-8, False, 15, 2, id="invert_polyhedron_faces=False"),
	],
)
def test_meshstack3d_generate_mesh(bmesh, tolerance, invert_polyhedron_faces, n_points_ref, n_cells_ref, request):
	"""Test the method pvgridder.MeshStack3D.generate_mesh."""
	bmesh = request.getfixturevalue(bmesh) if isinstance(bmesh, str) else bmesh
	area = bmesh.compute_cell_sizes().cell_data["Area"].sum()
	mesh = (
		pvg.MeshStack3D(bmesh)
		.add(0.0)
		.add(2.0, resolution=2, group="layer")
		.generate_mesh(tolerance=tolerance, invert_polyhedron_faces=invert_polyhedron_faces)
	)
	assert mesh.n_points == n_points_ref
	assert mesh.n_cells == n_cells_ref
	assert (pvg.get_cell_group(mesh) == "layer").sum() == n_cells_ref
	assert np.allclose(np.unique(mesh.cell_data["CellCenter"][:, 2]), [0.5, 1.5])
	assert np.isclose(np.abs(mesh.compute_cell_sizes().cell_data["Volume"]).sum(), 2.0 * area)