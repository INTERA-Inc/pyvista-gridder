from collections.abc import Sequence
import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh_fixture, flatten",
    [
        # Simple unstructured grid
        pytest.param("simple_unstructured_grid", False, id="simple_ugrid_nested"),
        pytest.param("simple_unstructured_grid", True, id="simple_ugrid_flat"),

        # Other 3D cell types
        pytest.param("tetrahedron_grid", False, id="tetra_nested"),
        pytest.param("tetrahedron_grid", True, id="tetra_flat"),  # 4 vertices + 1 for cell size
        pytest.param("pyramid_grid", False, id="pyramid_nested"),
        pytest.param("pyramid_grid", True, id="pyramid_flat"),  # 5 vertices + 1 for cell size

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
        reconstructed = pv.UnstructuredGrid(result, actual_mesh.celltypes, actual_mesh.points)
        assert reconstructed.n_cells == actual_mesh.n_cells
        assert np.allclose(reconstructed.compute_cell_sizes()["Volume"], actual_mesh.compute_cell_sizes()["Volume"])


@pytest.mark.parametrize(
    "mesh_fixture, expected_dimension",
    [
        # Structured grids with different dimensions
        pytest.param("structured_grid_1d", 1, id="structured_1d"),
        pytest.param("structured_grid_2d", 2, id="structured_2d"),
        pytest.param("structured_grid_3d", 3, id="structured_3d"),
        
        # Explicit structured grid
        pytest.param("explicit_structured_grid", 3, id="explicit_structured"),
        
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
    "invalid_mesh",
    [
        pytest.param(pv.PolyData(), id="polydata"),
        pytest.param(pv.RectilinearGrid(), id="rectilinear_grid"),
    ],
)
def test_get_dimension_invalid_mesh(invalid_mesh):
    """Test that get_dimension raises an appropriate error for unsupported mesh types."""
    with pytest.raises(TypeError):
        pvg.get_dimension(invalid_mesh)
