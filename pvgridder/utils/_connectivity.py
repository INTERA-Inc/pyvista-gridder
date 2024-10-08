from __future__ import annotations

import pyvista as pv


def get_neighborhood(
    mesh: pv.UnstructuredGrid,
) -> list:
    neighbors = []
    
    for i in range(mesh.n_cells):
        cell_neighbors = mesh.cell_neighbors(i, connections="edges")
        neighbors.append(cell_neighbors)

    return neighbors


def get_connectivity(
    mesh: pv.UnstructuredGrid,
    return_polydata: bool = False,
) -> list | pv.PolyData:
    neighborhood = get_neighborhood(mesh)
    connectivity = set()

    for i, cell_neighbors in enumerate(neighborhood):
        for j in cell_neighbors:
            connectivity.add((min(i, j), max(i, j)))

    connectivity = sorted(connectivity)

    if return_polydata:
        poly = pv.PolyData()
        centers = mesh.cell_centers().points

        for i, j in connectivity:
            poly += pv.Line(centers[i], centers[j])

        return poly

    else:
        return connectivity
