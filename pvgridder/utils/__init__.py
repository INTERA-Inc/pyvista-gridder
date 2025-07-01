"""Utility functions."""

from ._connectivity import get_connectivity, get_neighborhood
from ._interactive import interactive_lasso_selection, interactive_selection
from ._misc import (
    average_points,
    decimate_rdp,
    extract_boundary_polygons,
    extract_cell_geometry,
    extract_cells_by_dimension,
    fuse_cells,
    merge,
    merge_lines,
    offset_polygon,
    quadraticize,
    reconstruct_line,
    remap_categorical_data,
    split_lines,
)
from ._properties import get_cell_connectivity, get_cell_group, get_dimension
