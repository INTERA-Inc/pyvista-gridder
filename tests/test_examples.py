import pytest


@pytest.mark.parametrize(
    "mesh, n_points_ref, n_cells_ref",
    [
        pytest.param("freyberg_delaunay", 3242, 6312, id="delaunay"),
        pytest.param("freyberg_delaunay_voronoi", 6579, 3242, id="delaunay_voronoi"),
        pytest.param("freyberg_quadtree", 1922, 1677, id="quadtree"),
        pytest.param("freyberg_quadtree_voronoi", 1963, 1677, id="quadtree_voronoi"),
        pytest.param(
            "freyberg_quadtree_nonconformal", 1922, 1677, id="quadtree_nonconformal"
        ),
    ],
)
def test_freyberg(mesh, n_points_ref, n_cells_ref, request):
    """Test the Freyberg example meshes."""
    mesh = request.getfixturevalue(mesh)
    assert mesh.n_points == n_points_ref
    assert mesh.n_cells == n_cells_ref
