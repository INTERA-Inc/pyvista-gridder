"""Utility functions."""

from ._connectivity import get_connectivity, get_neighborhood
from ._interactive import interactive_selection
from ._misc import (
    average_points,
    decimate_rdp,
    extract_boundary_polygons,
    extract_cell_geometry,
    extract_cells_by_dimension,
    merge,
    merge_lines,
    quadraticize,
    reconstruct_line,
    remap_categorical_data,
    split_lines,
)
from ._properties import get_dimension
