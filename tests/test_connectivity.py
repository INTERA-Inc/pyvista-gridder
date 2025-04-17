import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh_fixture",
    [
        pytest.param("anticline_2d", id="anticline-2d"),
        pytest.param("anticline_3d", id="anticline-3d"),
        pytest.param("topographic_terrain", id="topographic-terrain"),
        pytest.param("well_2d", id="well-2d"),
        pytest.param("well_2d_voronoi", id="well-2d-voronoi"),
        pytest.param("well_3d", id="well-3d"),
        pytest.param("well_3d_voronoi", id="well-3d-voronoi"),
    ],
)
def test_get_neighborhood(mesh_fixture, request):
    """Test neighborhood extraction with different mesh types."""
    # Get the mesh from the fixture
    mesh = request.getfixturevalue(mesh_fixture)
    
    ndim = pvg.get_dimension(mesh)
    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=True)
    neighbors_ref = [
        mesh.cell_neighbors(i, "edges" if ndim == 2 else "faces")
        for i in range(mesh.n_cells)
    ]

    for neighbor, neighbor_ref in zip(neighbors, neighbors_ref):
        assert set(neighbor) == set(neighbor_ref)


@pytest.mark.parametrize(
    "mesh_fixture, cell_ids, empty_cell_ids",
    [
        pytest.param(
            "anticline_2d",
            [387, 388, 389, 390, 391],
            [[428], [429], [430], [431], [432]],
            id="anticline-2d-empty-cells"
        ),
        pytest.param(
            "anticline_3d",
            [3708, 3709, 3710, 3711, 3712],
            [[4118], [4119], [4120], [4121], [4122]],
            id="anticline-3d-empty-cells"
        ),
        pytest.param(
            "simple_2d_grid",
            [0, 1],
            [[1], [0]],
            id="simple-2d-grid-empty-cells"
        ),
        pytest.param(
            "simple_3d_grid",
            [0, 1],
            [[1], [0]],
            id="simple-3d-grid-empty-cells"
        ),
    ],
)
def test_get_neighborhood_empty_cells(mesh_fixture, cell_ids, empty_cell_ids, request):
    """Test whether empty cells are kept or removed from neighborhood."""
    # Get the mesh from the fixture
    mesh = request.getfixturevalue(mesh_fixture)
    
    assert len(cell_ids) == len(empty_cell_ids)

    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=False)
    for cell_id, empty_cell_id in zip(cell_ids, empty_cell_ids):
        for cid in empty_cell_id:
            assert cid in neighbors[cell_id]

    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=True)
    for cell_id, empty_cell_id in zip(cell_ids, empty_cell_ids):
        for cid in empty_cell_id:
            assert cid not in neighbors[cell_id]


@pytest.mark.parametrize(
    "mesh_fixture, test_custom_centers",
    [
        pytest.param("anticline_2d", False, id="anticline-2d-no-custom"),
        pytest.param("anticline_3d", False, id="anticline-3d-no-custom"),
        pytest.param("well_2d", True, id="well-2d-with-custom"),
        pytest.param("well_3d", True, id="well-3d-with-custom"),
        pytest.param("small_quad_grid", True, id="small-quad-grid-with-custom"),
    ],
)
def test_get_connectivity(mesh_fixture, test_custom_centers, request):
    """Test connectivity extraction with different meshes."""
    # Get the mesh from the fixture
    mesh = request.getfixturevalue(mesh_fixture)
    
    # Test with default parameters
    connectivity = pvg.get_connectivity(mesh)
    
    # Basic checks
    assert isinstance(connectivity, pv.PolyData)
    assert connectivity.n_points == mesh.n_cells
    assert connectivity.n_lines > 0
    
    # Test with custom cell centers only for specified meshes
    if test_custom_centers:
        cell_centers = mesh.cell_centers().points
        connectivity_custom = pvg.get_connectivity(mesh, cell_centers=cell_centers)
        
        assert isinstance(connectivity_custom, pv.PolyData)
        assert connectivity_custom.n_points == mesh.n_cells
        assert connectivity_custom.n_lines > 0
    
    # Test with remove_empty_cells=False
    connectivity_with_empty = pvg.get_connectivity(mesh, remove_empty_cells=False)
    assert isinstance(connectivity_with_empty, pv.PolyData)


def test_get_connectivity_invalid_centers(anticline_2d):
    """Test get_connectivity with invalid cell centers."""
    # Test with invalid cell centers shape
    with pytest.raises(ValueError, match="invalid cell centers"):
        pvg.get_connectivity(anticline_2d, cell_centers=np.ones((anticline_2d.n_cells, 2)))
