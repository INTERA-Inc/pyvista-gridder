import numpy as np
import pytest

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesha, meshb",
    [
        pytest.param("structured_grid_3d", "structured_grid_3d", id="structured-grids"),
        pytest.param(
            "tetrahedron_grid",
            "pyramid_grid",
            id="unstructured-grids",
        ),
        pytest.param(
            "structured_grid_3d",
            "tetrahedron_grid",
            id="mixed-grids",
        ),
    ],
)
def test_mesh_merge(request, mesha, meshb):
    mesha = request.getfixturevalue(mesha)
    meshb = request.getfixturevalue(meshb).translate((0.0, 1.0, 2.0))
    meshc = mesha.translate((0.0, -1.0, -2.0))

    mesha.cell_data["CellGroup"] = np.zeros(mesha.n_cells, dtype=int)
    meshb.cell_data["CellGroup"] = np.zeros(meshb.n_cells, dtype=int)
    mesha.user_dict["CellGroup"] = {"A": 0}
    meshb.user_dict["CellGroup"] = {"B": 0}

    mesh = (
        pvg.MeshMerge(default_group="C")
        .add(mesha)
        .add(meshb)
        .add(meshc)
        .generate_mesh()
    )
    assert mesh.n_cells == mesha.n_cells + meshb.n_cells + meshc.n_cells
    assert mesh.user_dict["CellGroup"] == {"A": 0, "B": 1, "C": 2}
