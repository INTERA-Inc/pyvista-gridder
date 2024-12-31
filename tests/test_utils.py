import pytest
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
def test_extract_cell_geometry(mesh):
    ndim = pvg.get_dimension(mesh)

    if "vtkGhostType" in mesh.cell_data:
        mesh = mesh.extract_cells(mesh["vtkGhostType"] == 0)
        
    neighbors = pvg.get_neighborhood(mesh)
    neighbors_ref = [mesh.cell_neighbors(i, "edges" if ndim == 2 else "faces") for i in range(mesh.n_cells)]

    for neighbor, neighbor_ref in zip(neighbors, neighbors_ref):
        assert set(neighbor) == set(neighbor_ref)
