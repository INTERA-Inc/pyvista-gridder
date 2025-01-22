"""Utility functions."""

from ._connectivity import get_connectivity, get_neighborhood
from ._misc import (
    decimate_rdp,
    extract_boundary_polygons,
    extract_cell_geometry,
    extract_cells_by_dimension,
    merge,
    quadraticize,
    reconstruct_line,
    remap_categorical_data,
    split_lines,
)
from ._properties import get_dimension
