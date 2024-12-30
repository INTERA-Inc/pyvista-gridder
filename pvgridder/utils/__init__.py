"""Utility functions."""

from ._connectivity import get_connectivity, get_neighborhood
from ._misc import (
    decimate_rdp,
    extract_boundary_polygons,
    extract_cell_geometry,
    merge,
    reconstruct_line,
    split_lines,
    quadraticize,
)
from ._properties import get_dimension
