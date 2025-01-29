import numpy as np
import pytest
import pyvista as pv
import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh",
    [
        pvg.examples.load_anticline_2d(),
        pvg.examples.load_anticline_3d(),
        pvg.examples.load_topographic_terrain(),
        pvg.examples.load_well_2d(),
        pvg.examples.load_well_2d(voronoi=True),
        pvg.examples.load_well_3d(),
        pvg.examples.load_well_3d(voronoi=True),
    ]
)
def test_get_neighborhood(mesh):
    ndim = pvg.get_dimension(mesh)
    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=True)
    neighbors_ref = [mesh.cell_neighbors(i, "edges" if ndim == 2 else "faces") for i in range(mesh.n_cells)]

    for neighbor, neighbor_ref in zip(neighbors, neighbors_ref):
        assert set(neighbor) == set(neighbor_ref)


@pytest.mark.parametrize(
    "mesh, cell_ids, empty_cell_ids",
    [
        (
            pvg.examples.load_anticline_2d(),
            [387, 388, 389, 390, 391],
            [[428], [429], [430], [431], [432]],
        ),
        (
            pvg.examples.load_anticline_3d(),
            [3708, 3709, 3710, 3711, 3712],
            [[4118], [4119], [4120], [4121], [4122]],
        ),
        (
            pv.UnstructuredGrid(
                {
                    pv.CellType.QUAD: np.array(
                        [
                            [0, 1, 2, 3],
                            [1, 4, 5, 2],
                            [4, 6, 7, 5],
                        ]
                    )
                },
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.5, 0.0],
                        [1.0, 0.5, 0.0],
                        [0.0, 1.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 1.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [3.0, 1.0, 0.0],
                    ]
                ),
            ),
            [0, 1],
            [[1], [0]],
        ),
        (
            pv.UnstructuredGrid(
                {
                    pv.CellType.HEXAHEDRON: np.array(
                        [
                            [0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 8, 9, 2, 5, 10, 11, 6],
                            [8, 12, 13, 9, 10, 14, 15, 11],
                        ]
                    )
                },
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.5],
                        [1.0, 1.0, 0.5],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.5],
                        [1.0, 1.0, 0.5],
                        [0.0, 1.0, 1.0],
                        [2.0, 0.0, 0.0],
                        [2.0, 1.0, 0.0],
                        [2.0, 0.0, 1.0],
                        [2.0, 1.0, 1.0],
                        [3.0, 0.0, 0.0],
                        [3.0, 1.0, 0.0],
                        [3.0, 0.0, 1.0],
                        [3.0, 1.0, 1.0],
                    ]
                )
            ),
            [0, 1],
            [[1], [0]],
        ),
    ]
)
def test_get_neighborhood_empty_cells(mesh, cell_ids, empty_cell_ids):
    """Test whether empty cells are kept or removed from neighborhood."""
    assert len(cell_ids) == len(empty_cell_ids)

    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=False)
    for cell_id, empty_cell_id in zip(cell_ids, empty_cell_ids):
        for cid in empty_cell_id:
            assert cid in neighbors[cell_id]

    neighbors = pvg.get_neighborhood(mesh, remove_empty_cells=True)
    for cell_id, empty_cell_id in zip(cell_ids, empty_cell_ids):
        for cid in empty_cell_id:
            assert cid not in neighbors[cell_id]
