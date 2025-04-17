import pytest
import pyvista as pv
import numpy as np
import pvgridder as pvg


def test_interactive_selection():
    # Create a simple mesh
    mesh = pv.Sphere()

    # Mock plotter
    plotter = pv.Plotter(off_screen=True)

    # Call the function
    result = pvg.interactive_selection(
        mesh=mesh,
        plotter=plotter,
        scalars=None,
        view="xy",
        parallel_projection=True,
        preference="cell",
        tolerance=0.0
    )

    # Assert the result is an empty array (no selections made)
    assert isinstance(result, np.ndarray)
    assert result.size == 0
