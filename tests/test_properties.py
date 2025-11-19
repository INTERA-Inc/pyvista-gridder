import numpy as np
import pytest
import pyvista as pv
import vtk

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh_fixture, flatten",
    [
        # Simple unstructured grid
        pytest.param("simple_unstructured_grid", False, id="simple_ugrid_nested"),
        pytest.param("simple_unstructured_grid", True, id="simple_ugrid_flat"),
        # Other 3D cell types
        pytest.param("tetrahedron_grid", False, id="tetra_nested"),
        pytest.param(
            "tetrahedron_grid", True, id="tetra_flat"
        ),  # 4 vertices + 1 for cell size
        pytest.param("pyramid_grid", False, id="pyramid_nested"),
        pytest.param(
            "pyramid_grid", True, id="pyramid_flat"
        ),  # 5 vertices + 1 for cell size
        # Example mesh - complex unstructured grid
        pytest.param("well_2d", False, id="well_2d_nested"),
        pytest.param("well_2d", True, id="well_2d_flat"),
        pytest.param("well_3d_voronoi", False, id="voronoi_well_3d_nested"),
        pytest.param("well_3d_voronoi", True, id="voronoi_well_3d_flat"),
    ],
)
def test_get_cell_connectivity(request, mesh_fixture, flatten):
    """Test retrieving cell connectivity with different meshes and flatten options."""
    # Get the actual mesh
    actual_mesh = request.getfixturevalue(mesh_fixture)

    # Get cell connectivity
    result = pvg.get_cell_connectivity(actual_mesh, flatten=flatten)

    # Basic verification based on flatten option
    if flatten:
        assert np.ndim(result) == 1

    else:
        # Should be a sequence (tuple) of cells
        assert isinstance(result, tuple)

        # Check number of cells
        assert len(result) == actual_mesh.n_cells

    # Verify the result can be used to reconstruct a grid
    if flatten:
        reconstructed = pv.UnstructuredGrid(
            result, actual_mesh.celltypes, actual_mesh.points
        )
        assert reconstructed.n_cells == actual_mesh.n_cells
        assert np.allclose(
            reconstructed.compute_cell_sizes()["Volume"],
            actual_mesh.compute_cell_sizes()["Volume"],
        )


@pytest.mark.parametrize(
    "mesh_fixture, expected_dimension",
    [
        # Structured grids with different dimensions
        pytest.param("structured_grid_1d", 1, id="structured_1d"),
        pytest.param("structured_grid_2d", 2, id="structured_2d"),
        pytest.param("structured_grid_3d", 3, id="structured_3d"),
        # Explicit structured grid
        # pytest.param("explicit_structured_grid", 3, id="explicit_structured"),
        # Unstructured grids with different cell types
        pytest.param("simple_unstructured_grid", 3, id="mixed_ugrid"),
        pytest.param(pv.examples.cells.Quadrilateral, 2, id="quad_ugrid"),
        pytest.param(lambda: pv.Line().cast_to_unstructured_grid(), 1, id="line_ugrid"),
        pytest.param(pv.examples.cells.Vertex, 0, id="vertex_ugrid"),
        # Example meshes
        pytest.param("anticline_2d", 2, id="anticline_2d"),
        pytest.param("anticline_3d", 3, id="anticline_3d"),
    ],
)
def test_get_dimension(request, mesh_fixture, expected_dimension):
    """Test retrieving mesh dimension with different mesh types."""
    # Get the actual mesh
    if callable(mesh_fixture):
        actual_mesh = mesh_fixture()

    else:
        actual_mesh = request.getfixturevalue(mesh_fixture)

    # Get mesh dimension
    result = pvg.get_dimension(actual_mesh)

    # Verify dimension matches expected value
    assert result == expected_dimension


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param("structured_grid_2d", id="structured-2d"),
        pytest.param("structured_grid_3d", id="structured-3d"),
        pytest.param("well_2d_voronoi", id="unstructured-2d"),
        pytest.param("well_3d_voronoi", id="unstructured-3d"),
    ],
)
def test_get_cell_centers(request, mesh):
    """Test retrieving cell centers for different mesh types."""
    # No ghost cells nor empty cells
    mesh = request.getfixturevalue(mesh)
    centers = pvg.get_cell_centers(mesh)
    assert centers.shape == (mesh.n_cells, 3)

    # With ghost cells and empty cells
    if isinstance(mesh, pv.UnstructuredGrid):
        empty_mesh = pv.UnstructuredGrid([0], [pv.CellType.EMPTY_CELL], [])
        mesh = mesh + empty_mesh

    mesh.cell_data["vtkGhostType"] = np.zeros(mesh.n_cells, dtype=np.uint8)
    mesh.cell_data["vtkGhostType"][1 : mesh.n_cells // 2] = 32
    ghost_cells = mesh.cell_data["vtkGhostType"].copy()
    centers = pvg.get_cell_centers(mesh)

    assert centers.shape == (mesh.n_cells, 3)
    assert np.allclose(mesh.cell_data["vtkGhostType"], ghost_cells)
    assert not np.isnan(centers[ghost_cells > 0]).any()

    if isinstance(mesh, pv.UnstructuredGrid):
        mask = mesh.celltypes == pv.CellType.EMPTY_CELL
        assert np.isnan(centers[mask]).all()
        assert not np.isnan(centers[~mask]).any()


@pytest.mark.skipif(
    vtk.__version__.startswith("9.5"),
    reason="Skipped VTK 9.5 due to compatibility issues",
)
@pytest.mark.parametrize(
    "mesh, method, ref_sum",
    [
        pytest.param("concave_polyhedron", "box", (2.5, 3.5, 0.5), id="concave-box"),
        pytest.param(
            "concave_polyhedron",
            "geometric",
            (2.0, 2.66666667, 0.5),
            id="concave-geometric",
        ),
        pytest.param(
            "concave_polyhedron",
            "tetra",
            (1.40909091, 2.40909091, 0.5),
            id="concave-tetra",
        ),
        pytest.param("half_stadium", "box", (0.0, 0.0, 0.5), id="convex-box"),
        pytest.param(
            "half_stadium", "geometric", (0.0, 0.606742292, 0.5), id="convex-geometric"
        ),
        pytest.param(
            "half_stadium", "tetra", (0.0, -0.0933820983, 0.5), id="convex-tetra"
        ),
    ],
)
def test_get_cell_centers_polyhedron_method(request, mesh, method, ref_sum):
    """Test retrieving cell centers with different polyhedron method."""
    mesh = request.getfixturevalue(mesh)
    centers = pvg.get_cell_centers(mesh, polyhedron_method=method)
    assert np.allclose(centers.sum(axis=0), ref_sum)


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param("simple_unstructured_grid", id="simple-ugrid"),
        pytest.param("structured_grid_3d", id="structured-grid-3d"),
    ],
)
def test_get_cell_group(request, mesh):
    """Test retrieving cell group."""
    mesh = request.getfixturevalue(mesh)

    cell_groups = pvg.get_cell_group(mesh)
    assert cell_groups is None

    mesh.cell_data["CellGroup"] = np.zeros(mesh.n_cells, dtype=int)
    mesh.cell_data["CellGroup"][mesh.n_cells // 2 :] = 1
    mesh.user_dict["CellGroup"] = {"foo": 0, "bar": 1}
    cell_groups = pvg.get_cell_group(mesh)
    group_map = {v: k for k, v in mesh.user_dict["CellGroup"].items()}
    assert cell_groups.tolist() == [group_map[i] for i in mesh.cell_data["CellGroup"]]
