"""Pytest fixtures for pyvista-gridder tests."""
from collections.abc import Sequence

import numpy as np
import packaging.version
import pytest
import pyvista as pv

import pvgridder as pvg


# Check PyVista version for merge_points compatibility
PYVISTA_VERSION = packaging.version.parse(pv.__version__)
MERGE_POINTS_COMPATIBLE = PYVISTA_VERSION <= packaging.version.parse("0.45")

# Common fixtures for meshes
@pytest.fixture
def simple_polydata_with_duplicates():
    """Create a simple polydata with duplicate points."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # duplicate
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([3, 0, 1, 3])  # triangle

    return pv.PolyData(points, faces=faces)


@pytest.fixture
def simple_polydata_with_close_points():
    """Create a simple polydata with points that are very close."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0001, 0.0, 0.0],  # very close to previous point
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([4, 0, 1, 2, 3])  # quad

    return pv.PolyData(points, faces=faces)


@pytest.fixture
def simple_line():
    """Create a simple line with redundant points."""
    return pv.Line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], resolution=5)


@pytest.fixture
def sinusoidal_line():
    """Create a sinusoidal line with many points."""
    # Create a more robust sinusoidal line for testing
    x = np.linspace(0.0, 2.0 * np.pi, 100)
    y = np.sin(x)
    z = np.zeros_like(x)
    points = np.column_stack((x, y, z)).astype(np.float64)

    # Create polydata with lines
    line_indices = np.arange(len(points))
    lines = np.array([len(line_indices)] + line_indices.tolist())

    return pv.PolyData(points, lines=lines)


@pytest.fixture
def multiple_lines_polydata():
    """Create a polydata with multiple lines."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # First line
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],  # Second line
        ],
        dtype=np.float64,
    )
    lines = np.array([3, 0, 1, 2, 3, 3, 4, 5])  # Two lines with 3 points each

    return pv.PolyData(points, lines=lines)


@pytest.fixture
def square_points():
    """Create a square as points."""
    return np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


@pytest.fixture
def square_polydata(square_points):
    """Create a square as polydata."""
    faces = np.array([4, 0, 1, 2, 3])

    return pv.PolyData(square_points, faces=faces)


@pytest.fixture
def mixed_dimension_grid():
    """Create a mixed-dimension unstructured grid."""
    return pv.UnstructuredGrid(
        {
            pv.CellType.HEXAHEDRON: np.array([[0, 1, 3, 2, 4, 5, 7, 6]]),
            pv.CellType.QUAD: np.array([[8, 9, 11, 10]]),
        },
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


@pytest.fixture
def mesh_with_categorical_data():
    """Create a mesh with categorical data."""
    grid = pv.ImageData(dimensions=(3, 3, 3))
    grid.cell_data["category"] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])[
        : grid.n_cells
    ]
    grid.user_dict["category"] = {"A": 0, "B": 1, "C": 2}

    return grid


# Example data fixtures
@pytest.fixture
def anticline_2d():
    """Fixture for anticline 2D example mesh."""
    return pvg.examples.load_anticline_2d()


@pytest.fixture
def anticline_3d():
    """Fixture for anticline 3D example mesh."""
    return pvg.examples.load_anticline_3d()


@pytest.fixture
def well_2d():
    """Fixture for well 2D example mesh."""
    return pvg.examples.load_well_2d()


@pytest.fixture
def well_2d_voronoi():
    """Fixture for well 2D voronoi example mesh."""
    return pvg.examples.load_well_2d(voronoi=True)


@pytest.fixture
def well_3d():
    """Fixture for well 3D example mesh."""
    return pvg.examples.load_well_3d()


@pytest.fixture
def well_3d_voronoi():
    """Fixture for well 3D voronoi example mesh."""
    return pvg.examples.load_well_3d(voronoi=True)


@pytest.fixture
def topographic_terrain():
    """Fixture for topographic terrain example mesh."""
    return pvg.examples.load_topographic_terrain()


# Grid fixtures for connectivity testing
@pytest.fixture
def simple_2d_grid():
    """Fixture for a simple 2D quad grid."""
    return pv.UnstructuredGrid(
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
        )
    )


@pytest.fixture
def simple_3d_grid():
    """Fixture for a simple 3D hexahedral grid."""
    return pv.UnstructuredGrid(
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
    )


@pytest.fixture
def small_quad_grid():
    """Fixture for a small quad grid with two cells."""
    return pv.UnstructuredGrid(
        {
            pv.CellType.QUAD: np.array(
                [
                    [0, 1, 2, 3],
                    [1, 4, 5, 2],
                ]
            )
        },
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
    )


# Property test fixtures
@pytest.fixture
def simple_unstructured_grid():
    """Create a simple unstructured grid with hexahedron and quad cells."""
    return pv.examples.cells.Tetrahedron() + pv.examples.cells.Hexahedron()


@pytest.fixture
def tetrahedron_grid():
    """Create an unstructured grid with a tetrahedron cell."""
    return pv.examples.cells.Tetrahedron()


@pytest.fixture
def pyramid_grid():
    """Create an unstructured grid with a pyramid cell."""
    return pv.examples.cells.Pyramid()


@pytest.fixture
def explicit_structured_grid():
    """Create a simple explicit structured grid."""
    return pv.ExplicitStructuredGrid(pv.RectilinearGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))


@pytest.fixture
def structured_grid_1d():
    """Create a 1D structured grid."""
    return pv.RectilinearGrid([0.0, 1.0, 2.0], [0.0], [0.0]).cast_to_structured_grid()


@pytest.fixture
def structured_grid_2d():
    """Create a 2D structured grid."""
    return pv.RectilinearGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0]).cast_to_structured_grid()


@pytest.fixture
def structured_grid_3d():
    """Create a 3D structured grid."""
    return pv.RectilinearGrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]).cast_to_structured_grid()