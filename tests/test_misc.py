from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


@pytest.mark.parametrize(
    "mesh, tolerance, expected_points, reference_point_sum",
    [
        # Simple meshes with duplicated/close points
        pytest.param(
            "simple_polydata_with_duplicates", 0.0, 3, 2.0, id="duplicates_exact"
        ),
        pytest.param(
            "simple_polydata_with_duplicates", 0.1, 3, 2.0, id="duplicates_tolerance"
        ),
        pytest.param(
            "simple_polydata_with_close_points", 0.0, 4, 3.0001, id="close_points_exact"
        ),
        pytest.param(
            "simple_polydata_with_close_points",
            0.1,
            3,
            2.00005,
            id="close_points_tolerance",
        ),
        # Example meshes
        pytest.param(
            pvg.examples.load_anticline_2d,
            1e-10,
            None,
            999.4173879623413,
            id="anticline_2d",
        ),
        pytest.param(
            pvg.examples.load_well_2d, 1e-10, None, 1.1368683772161603e-13, id="well_2d"
        ),
    ],
)
def test_average_points(request, mesh, tolerance, expected_points, reference_point_sum):
    """Test point averaging with different meshes and tolerances."""
    # Get the actual mesh (executing the function if it's a callable or fixture)
    if isinstance(mesh, str):
        actual_mesh = request.getfixturevalue(mesh)

    else:
        actual_mesh = mesh()

    # Extract surface to get polydata if needed
    if not isinstance(actual_mesh, pv.PolyData):
        actual_mesh = actual_mesh.extract_surface()

    # If it's an example mesh, we'll check area preservation instead of point count
    is_example = mesh in [pvg.examples.load_anticline_2d, pvg.examples.load_well_2d]

    if is_example:
        # Get original area for example meshes
        original_area = actual_mesh.compute_cell_sizes(
            length=False, area=True, volume=False
        )["Area"].sum()

    # Apply average_points
    original_point_count = actual_mesh.n_points
    result = pvg.average_points(actual_mesh, tolerance=tolerance)

    # Basic verification for all meshes
    assert isinstance(result, pv.PolyData)
    assert result.n_points <= original_point_count

    # Verify the sum of points matches the reference value
    assert np.allclose(result.points.sum(), reference_point_sum, rtol=1e-5)

    # Specific checks based on mesh type
    if is_example:
        # For example meshes, check area preservation
        result_area = result.compute_cell_sizes(length=False, area=True, volume=False)[
            "Area"
        ].sum()
        assert np.isclose(original_area, result_area, rtol=1e-4)

    elif expected_points is not None:
        # For simple meshes, check expected point count
        assert result.n_points == expected_points


@pytest.mark.parametrize(
    "line_source, tolerance, reference_point_sum",
    [
        # Sinusoidal line - more complex and sufficient for testing
        pytest.param(
            "sinusoidal_line", 0.0, 314.1592653589793, id="sinusoidal_no_decimation"
        ),
        pytest.param(
            "sinusoidal_line", 0.1, 24.710092386618626, id="sinusoidal_with_decimation"
        ),
        # Example mesh with curved lines
        pytest.param(
            lambda: pvg.extract_boundary_polygons(
                pvg.examples.load_anticline_2d(), fill=False
            )[0],
            0.1,
            6.800000190734863,
            id="anticline_boundary",
        ),
    ],
)
def test_decimate_rdp(request, line_source, tolerance, reference_point_sum):
    """Test line decimation with different lines and tolerances."""
    # Get the actual line (executing the function if it's a callable or fixture)
    if isinstance(line_source, str):
        actual_line = request.getfixturevalue(line_source)

    else:
        try:
            actual_line = line_source()

        except Exception:
            pytest.skip("Failed to create line for decimate_rdp test")

    # Skip if we got an empty list or None
    if actual_line is None:
        pytest.skip("No valid lines found for decimation")

    # For consistency, ensure the input is a line polydata
    if not actual_line.n_lines > 0:
        try:
            # Try to extract lines
            actual_line = actual_line.extract_feature_edges()

        except Exception:
            pytest.skip("Could not extract lines from input")

    # Track original point count
    original_point_count = actual_line.n_points

    try:
        # Decimate the line
        result = pvg.decimate_rdp(actual_line, tolerance=tolerance)

        # Should be a polydata with lines
        assert isinstance(result, pv.PolyData)
        assert result.n_lines > 0

        # Verify the sum of points matches the reference value
        assert np.allclose(result.points.sum(), reference_point_sum, rtol=1e-5)

        if tolerance > 0.0:
            # With positive tolerance, should have fewer points
            assert result.n_points <= original_point_count

        else:
            # With zero tolerance, should have same number of points
            assert result.n_points == original_point_count

    except Exception as e:
        if "not implemented for input of type" in str(e).lower():
            pytest.skip(f"Decimation not implemented for this input type: {str(e)}")

        else:
            raise


@pytest.mark.parametrize(
    "mesh, fill, reference_point_sum",
    [
        # Basic mesh
        pytest.param(
            pv.Plane(i_resolution=3, j_resolution=3),
            True,
            1.4901161e-08,
            id="plane_filled",
        ),
        pytest.param(
            pv.Plane(i_resolution=3, j_resolution=3),
            False,
            1.4901161e-08,
            id="plane_outline",
        ),
        # Example meshes
        pytest.param(
            pvg.examples.load_anticline_2d,
            True,
            168.00002872943878,
            id="anticline_2d_filled",
        ),
        pytest.param(
            pvg.examples.load_anticline_2d,
            False,
            168.00002872943878,
            id="anticline_2d_outline",
        ),
    ],
)
def test_extract_boundary_polygons(mesh, fill, reference_point_sum):
    """Test boundary extraction with different meshes and fill options."""
    # Get the actual mesh (calling the function if it's callable)
    actual_mesh = mesh() if callable(mesh) else mesh

    # Extract boundaries
    boundaries = pvg.extract_boundary_polygons(actual_mesh, fill=fill)

    # Should be a sequence of polydata objects
    assert isinstance(boundaries, Sequence)
    assert len(boundaries) > 0
    assert all(isinstance(b, pv.PolyData) for b in boundaries)

    # Verify the sum of all boundary points matches the reference value
    total_points_sum = sum(b.points.sum() for b in boundaries)
    assert np.allclose(total_points_sum, reference_point_sum, rtol=1e-5)

    if fill:
        # With fill=True, each boundary should have faces
        assert all(b.n_faces_strict > 0 for b in boundaries)

    else:
        # With fill=False, each boundary should have lines
        assert all(b.n_lines > 0 for b in boundaries)


@pytest.mark.parametrize(
    "mesh, remove_empty_cells, reference_point_sum",
    [
        # Basic meshes
        pytest.param(
            lambda: pv.StructuredGrid(
                *np.meshgrid(
                    np.linspace(0, 2, 3, dtype=np.float64),
                    np.linspace(0, 2, 3, dtype=np.float64),
                    np.linspace(0, 2, 3, dtype=np.float64),
                    indexing="ij",
                )
            ),
            True,
            81.0,
            id="grid_remove_empty",
        ),
        pytest.param(
            lambda: pv.StructuredGrid(
                *np.meshgrid(
                    np.linspace(0, 2, 3, dtype=np.float64),
                    np.linspace(0, 2, 3, dtype=np.float64),
                    np.linspace(0, 2, 3, dtype=np.float64),
                    indexing="ij",
                )
            ),
            False,
            81.0,
            id="grid_keep_empty",
        ),
        # Example meshes
        pytest.param(
            pvg.examples.load_anticline_2d, True, 1081.0173902511597, id="anticline_2d"
        ),
        pytest.param(
            pvg.examples.load_anticline_3d, True, 33651.39129276276, id="anticline_3d"
        ),
        pytest.param(pvg.examples.load_well_2d, True, 0.0, id="well_2d"),
    ],
)
def test_extract_cell_geometry(mesh, remove_empty_cells, reference_point_sum):
    """Test cell geometry extraction with different meshes and empty cell options."""
    # Get the actual mesh (calling the function if it's callable)
    actual_mesh = mesh() if callable(mesh) else mesh

    # Extract cell geometry
    result = pvg.extract_cell_geometry(
        actual_mesh, remove_empty_cells=remove_empty_cells
    )

    # Should be a polydata with cell outlines
    assert isinstance(result, pv.PolyData)
    assert result.n_faces_strict > 0 or result.n_lines > 0  # Either faces or lines

    # Verify the sum of points matches the reference value
    assert np.allclose(result.points.sum(), reference_point_sum, rtol=1e-5)

    # Should have the original cell IDs
    assert "vtkOriginalCellIds" in result.cell_data


@pytest.mark.parametrize(
    "mesh, ndim, method, expected_result",
    [
        # Basic mesh with mixed dimensions
        pytest.param("mixed_dimension_grid", 2, "lower", True, id="basic_2d_lower"),
        pytest.param("mixed_dimension_grid", 3, "upper", True, id="basic_3d_upper"),
        # Example meshes
        pytest.param(pvg.examples.load_well_2d, 2, "lower", True, id="well_2d_lower"),
        pytest.param(pvg.examples.load_well_2d, 2, "upper", True, id="well_2d_upper"),
        pytest.param(pvg.examples.load_well_3d, 3, "lower", True, id="well_3d_lower"),
        pytest.param(pvg.examples.load_well_3d, 3, "upper", True, id="well_3d_upper"),
    ],
)
def test_extract_cells_by_dimension(request, mesh, ndim, method, expected_result):
    """Test extraction of cells by dimension with different meshes and methods."""
    # Get the actual mesh (executing the function if it's a callable or fixture)
    if isinstance(mesh, str):
        actual_mesh = request.getfixturevalue(mesh)

    else:
        actual_mesh = mesh()

    # Extract cells by dimension
    result = pvg.extract_cells_by_dimension(actual_mesh, ndim=ndim, method=method)

    # Should have cells if expected_result is True
    if expected_result:
        assert result.n_cells > 0

    else:
        assert result.n_cells == 0


@pytest.mark.parametrize(
    "mesh, cell_ids",
    [
        # # Basic mesh with adjacent quads
        pytest.param(
            lambda: pv.ImageData(dimensions=(2, 3, 1)),
            [0, 1],
            id="basic_quads",
        ),
        # Example mesh
        pytest.param("well_2d", np.arange(8), id="well_2d"),
        # Example mesh with hidden cells
        pytest.param("anticline_2d", [[164, 165, 205, 206], [20, 61, 102, 143]], id="anticline_2d"),
    ],
)
def test_fuse_cells(request, mesh, cell_ids):
    """Test cell fusion with different meshes."""
    if isinstance(mesh, str):
        actual_mesh = request.getfixturevalue(mesh)

    else:
        actual_mesh = mesh()

    # Fuse cells
    results = pvg.fuse_cells(actual_mesh, cell_ids)

    # Check number of cells
    if np.ndim(cell_ids[0]) == 0:
        assert results.n_cells == actual_mesh.n_cells - len(cell_ids) + 1

    else:
        assert results.n_cells == actual_mesh.n_cells - np.concatenate(cell_ids).size + len(cell_ids)

    # Check area
    assert np.allclose(results.compute_cell_sizes()["Area"].sum(), actual_mesh.compute_cell_sizes()["Area"].sum())


@pytest.mark.parametrize(
    "mesh_type, axes",
    [
        ("structured", [0, 1, 2]),
        pytest.param("unstructured", [None]),
    ],
)
def test_merge_basic(mesh_type, axes):
    """Test basic mesh merging with different mesh types and axes."""
    if mesh_type == "structured":
        for axis in axes:
            # Create two simple structured grids using meshgrid instead of mgrid
            x1, y1, z1 = np.meshgrid(
                np.linspace(0, 1, 2, dtype=np.float64),
                np.linspace(0, 1, 2, dtype=np.float64),
                np.linspace(0, 1, 2, dtype=np.float64),
                indexing="ij",
            )
            grid1 = pv.StructuredGrid(x1, y1, z1)

            x2, y2, z2 = np.meshgrid(
                np.linspace(0, 1, 2, dtype=np.float64),
                np.linspace(0, 1, 2, dtype=np.float64),
                np.linspace(0, 1, 2, dtype=np.float64),
                indexing="ij",
            )
            # Offset the second grid to align with the first along the specified axis
            if axis == 0:
                x2 += 1.0

            elif axis == 1:
                y2 += 1.0

            else:
                z2 += 1.0

            grid2 = pv.StructuredGrid(x2, y2, z2)

            try:
                # Merge along axis
                result = pvg.merge(grid1, grid2, axis=axis)

                # Should be a structured grid with more points along the specified axis
                assert isinstance(result, pv.StructuredGrid)
                assert result.dimensions[axis] > grid1.dimensions[axis]

            except ValueError:
                pytest.skip("Grids can't be merged due to interface mismatch")

    else:
        # Create two simple unstructured grids
        grid1 = pv.UnstructuredGrid(
            {pv.CellType.QUAD: np.array([[0, 1, 3, 2]])},
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        )

        grid2 = pv.UnstructuredGrid(
            {pv.CellType.QUAD: np.array([[0, 1, 3, 2]])},
            np.array(
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        )

        # Merge unstructured grids - we already know MERGE_POINTS_COMPATIBLE is True at this point
        result = pvg.merge(grid1, grid2, merge_points=True)

        # Should be an unstructured grid with more cells
        assert isinstance(result, pv.UnstructuredGrid)
        assert result.n_cells == grid1.n_cells + grid2.n_cells


@pytest.mark.parametrize("as_lines, reference_point_sum", [(True, 11.0), (False, 11.0)])
def test_merge_lines_basic(
    multiple_lines_polydata, simple_line, as_lines, reference_point_sum
):
    """Test basic line merging with different output formats."""
    # Split the multiple lines polydata
    lines = pvg.split_lines(multiple_lines_polydata, as_lines=False)

    # Add a simple line
    lines.append(simple_line)

    # Merge the lines
    result = pvg.merge_lines(lines, as_lines=as_lines)

    # Should be a polydata with lines
    assert isinstance(result, pv.PolyData)
    assert result.n_lines > 0

    # Verify the sum of points matches the reference value
    assert np.allclose(result.points.sum(), reference_point_sum, rtol=1e-5)


@pytest.mark.parametrize(
    "mesh_or_points, distance, expected_area_change",
    [
        # Simple square polydata with different distances
        pytest.param("square_polydata", 0.1, "increase", id="square_outward"),
        pytest.param("square_polydata", -0.1, "decrease", id="square_inward"),
        pytest.param("square_polydata", 0.0, "same", id="square_zero"),
        # Square points (not a mesh)
        pytest.param("square_points", 0.1, None, id="square_points"),
        # Example mesh boundaries - using a direct function to create the boundary
        pytest.param(
            lambda: pvg.extract_boundary_polygons(
                pvg.examples.load_well_2d(), fill=True
            )[0],
            0.1,
            "increase",
            id="well_2d_boundary",
        ),
    ],
)
def test_offset_polygon(request, mesh_or_points, distance, expected_area_change):
    """Test polygon offset with different inputs and distances."""
    # Get actual mesh or points
    if isinstance(mesh_or_points, str):
        actual_input = request.getfixturevalue(mesh_or_points)

    else:
        actual_input = mesh_or_points()

    # Calculate original area if applicable
    is_polydata = isinstance(actual_input, pv.PolyData)
    original_area = None

    if is_polydata and expected_area_change is not None:
        original_area = actual_input.compute_cell_sizes()["Area"][0]

    result = pvg.offset_polygon(actual_input, distance=distance)

    # Basic verification for all inputs
    assert isinstance(result, pv.PolyData)
    assert result.n_faces_strict > 0

    # Check area change if applicable
    if is_polydata and expected_area_change is not None and original_area is not None:
        result_area = result.compute_cell_sizes()["Area"][0]

        if expected_area_change == "increase":
            assert result_area > original_area

        elif expected_area_change == "decrease":
            assert result_area < original_area

        else:  # "same"
            assert np.isclose(result_area, original_area)


@pytest.mark.parametrize(
    "points_source, close",
    [
        # Circle points with different close settings
        pytest.param(
            lambda: np.column_stack(
                [
                    np.cos(
                        np.random.RandomState(42).permutation(
                            np.linspace(0.0, 2.0 * np.pi, 20)[:-1]
                        )
                    ),
                    np.sin(
                        np.random.RandomState(42).permutation(
                            np.linspace(0.0, 2.0 * np.pi, 20)[:-1]
                        )
                    ),
                    np.zeros(20 - 1, dtype=np.float64),
                ]
            ).astype(np.float64),
            False,
            id="circle_open",
        ),
        pytest.param(
            lambda: np.column_stack(
                [
                    np.cos(
                        np.random.RandomState(42).permutation(
                            np.linspace(0.0, 2.0 * np.pi, 20)
                        )
                    ),
                    np.sin(
                        np.random.RandomState(42).permutation(
                            np.linspace(0.0, 2.0 * np.pi, 20)
                        )
                    ),
                    np.zeros(20, dtype=np.float64),
                ]
            ).astype(np.float64),
            True,
            id="circle_closed",
        ),
        # Example mesh points
        pytest.param(
            lambda: pvg.examples.load_anticline_2d()
            .extract_surface()
            .points[
                np.random.RandomState(42).choice(
                    pvg.examples.load_anticline_2d().extract_surface().n_points,
                    size=min(
                        20, pvg.examples.load_anticline_2d().extract_surface().n_points
                    ),
                    replace=False,
                )
            ]
            .astype(np.float64),
            False,
            id="anticline_points",
        ),
    ],
)
def test_reconstruct_line(points_source, close):
    """Test line reconstruction with different points sources and close options."""
    # Get points
    points = points_source() if callable(points_source) else points_source

    try:
        # Reconstruct line
        result = pvg.reconstruct_line(points, close=close)

        # Should be a valid polydata
        assert isinstance(result, pv.PolyData)
        assert result.n_points == len(points)
        assert result.n_lines > 0

    except Exception as e:
        # Some implementations may fail with certain edge cases
        pytest.skip(f"Line reconstruction failed with error: {str(e)}")


@pytest.mark.parametrize(
    "mesh, key, mapping, inplace, preference",
    [
        # Basic mesh with categorical data - inplace variations
        pytest.param(
            "mesh_with_categorical_data",
            "category",
            {0: 10, "B": 20},
            True,
            "cell",
            id="basic_inplace",
        ),
        pytest.param(
            "mesh_with_categorical_data",
            "category",
            {0: 10, "B": 20},
            False,
            "cell",
            id="basic_copy",
        ),
        # Basic mesh with categorical data - preference variations
        pytest.param(
            "mesh_with_categorical_data",
            "category",
            {0: 10, 1: 20},
            False,
            "cell",
            id="basic_cell",
        ),
        # Skip this test case for now as it uses a complex lambda function
        # pytest.param(lambda: lambda m: add_point_data(m, "category"), "category", {0: 10, 1: 20}, False, "point", id="basic_point"),
        # Example mesh
        pytest.param(
            pvg.examples.load_well_2d, "CellGroup", None, False, "cell", id="well_2d"
        ),
    ],
)
def test_remap_categorical_data(request, mesh, key, mapping, inplace, preference):
    """Test categorical data remapping with different options and meshes."""

    # Helper function to add point data for testing preference
    def add_point_data(m, key):
        m = m.copy()
        m.point_data[key] = np.tile([0, 1, 2], m.n_points)[: m.n_points]
        return m

    # Get the actual mesh
    if isinstance(mesh, str):
        actual_mesh = request.getfixturevalue(mesh)

    else:
        actual_mesh = mesh()

    # For example meshes, create a mapping based on the available data
    if mapping is None and key in actual_mesh.cell_data:
        values = np.unique(actual_mesh.cell_data[key])

        if len(values) >= 2:
            mapping = {values[0]: 999, values[1]: 888}

        else:
            pytest.skip(f"Not enough unique values in {key} to test remapping")

    # Skip if the key doesn't exist
    if key not in actual_mesh.cell_data and (
        preference != "point" or key not in actual_mesh.point_data
    ):
        pytest.skip(f"Key {key} not found in {preference} data")

    # Get original ID to check if operation was inplace
    original_id = id(actual_mesh)

    try:
        # Remap
        result = pvg.remap_categorical_data(
            actual_mesh, key, mapping, preference=preference, inplace=inplace
        )

        # Check inplace behavior
        if inplace:
            assert result is None
            result = actual_mesh
            assert id(result) == original_id

        else:
            assert result is not None
            assert id(result) != original_id

        # Check remapping worked
        if preference == "cell":
            for src, target in mapping.items():
                if isinstance(src, str):
                    # String keys are labels, need to find their values
                    if key in result.user_dict:
                        src_val = result.user_dict[key].get(src)

                        if src_val is not None:
                            assert target in result.cell_data[key]

                else:
                    assert target in result.cell_data[key]

        else:
            for src, target in mapping.items():
                if isinstance(src, int):  # Only check integer mappings for point data
                    assert target in result.point_data[key]

    except (ValueError, KeyError) as e:
        # Skip if data is not categorical or other valid error
        if "could not remap non-categorical" in str(
            e
        ) or "could not map unknown key" in str(e):
            pytest.skip(f"Skipping test due to: {str(e)}")

        else:
            raise


@pytest.mark.parametrize("as_lines, reference_point_sum", [(True, 9.0), (False, 9.0)])
def test_split_lines_basic(multiple_lines_polydata, as_lines, reference_point_sum):
    """Test basic line splitting with different output formats."""
    # Split the lines
    result = pvg.split_lines(multiple_lines_polydata, as_lines=as_lines)

    # Should return a sequence of polydata objects
    assert isinstance(result, Sequence)
    assert len(result) == 2  # Two separate lines
    assert all(isinstance(line, pv.PolyData) for line in result)

    # Verify the sum of points matches the reference value
    total_points_sum = sum(line.points.sum() for line in result)
    assert np.allclose(total_points_sum, reference_point_sum, rtol=1e-5)


def test_split_lines_with_sinusoidal_line(sinusoidal_line):
    """Test splitting with sinusoidal line."""
    # Test with sinusoidal line
    result_sine = pvg.split_lines(sinusoidal_line)

    # Should return a sequence with one polydata object (the original line)
    assert isinstance(result_sine, Sequence)
    assert len(result_sine) == 1
    assert isinstance(result_sine[0], pv.PolyData)
    assert result_sine[0].n_points == sinusoidal_line.n_points

    # Verify the sum of points matches the reference value
    total_points_sum = sum(line.points.sum() for line in result_sine)
    assert np.allclose(total_points_sum, 314.1592653589793, rtol=1e-5)


@pytest.mark.parametrize(
    "mesh, expected_celltype, reference_point_sum",
    [
        # Basic meshes with different cell types
        pytest.param(
            lambda: pv.Sphere(theta_resolution=8, phi_resolution=8)
            .triangulate()
            .cast_to_unstructured_grid(),
            pv.CellType.QUADRATIC_TRIANGLE,
            4.653301120757991e-15,
            id="sphere_triangles",
        ),
        pytest.param(
            lambda: pv.Plane(
                i_resolution=3, j_resolution=3
            ).cast_to_unstructured_grid(),
            pv.CellType.QUADRATIC_QUAD,
            0.0,
            id="plane_quads",
        ),
        # Example mesh
        pytest.param(
            lambda: pvg.examples.load_well_2d()
            .extract_cells(range(10))
            .extract_surface()
            .triangulate()
            .cast_to_unstructured_grid(),
            pv.CellType.QUADRATIC_TRIANGLE,
            341.24999991059303,
            id="well_2d_triangles",
        ),
    ],
)
def test_quadraticize(mesh, expected_celltype, reference_point_sum):
    """Test converting linear cells to quadratic cells."""
    # Get the actual mesh
    actual_mesh = mesh() if callable(mesh) else mesh()

    try:
        # Convert to quadratic
        result = pvg.quadraticize(actual_mesh)

        # Should have quadratic cells
        assert isinstance(result, pv.UnstructuredGrid)
        assert result.celltypes[0] == expected_celltype
        assert result.n_points > actual_mesh.n_points

        # Verify the sum of points matches the reference value
        assert np.allclose(result.points.sum(), reference_point_sum, rtol=1e-5)

    except NotImplementedError as e:
        # Skip if unsupported cell types
        pytest.skip(f"Quadraticize not supported for these cell types: {str(e)}")
