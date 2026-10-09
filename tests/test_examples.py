import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh, n_cells_ref",
    [
        pytest.param("freyberg", 3429, id="conformal"),
        pytest.param("freyberg_nonconformal", 2789, id="nonconformal")
    ]
)
def test_freyberg(mesh, n_cells_ref, request):
    mesh = request.getfixturevalue(mesh)
    cell_groups = pvg.get_cell_group(mesh)
    connectivity = pvg.get_connectivity(mesh)

    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.n_cells == 1677
    assert mesh.extract_cells(cell_groups == "River").n_cells == 320
    assert mesh.extract_cells(cell_groups == "Well").n_cells == 24
    assert connectivity.n_cells == n_cells_ref
