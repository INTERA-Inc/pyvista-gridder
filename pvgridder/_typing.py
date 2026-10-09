from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Tuple, Union

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from shapely import LineString, Polygon
from typing_extensions import TypeAlias


RectilinearLike: TypeAlias = Union[
    pv.ImageData,
    pv.RectilinearGrid,
]
StructuredLike: TypeAlias = Union[
    RectilinearLike,
    pv.StructuredGrid,
]
GridLike: TypeAlias = Union[
    StructuredLike,
    pv.ExplicitStructuredGrid,
    pv.UnstructuredGrid,
]
DataSetLike: TypeAlias = Union[
    GridLike,
    pv.DataSet,
    pv.PolyData,
]
VectorLike: TypeAlias = Union[
    Sequence[Union[int, float]],
    NDArray[Union[np.integer[Any], np.floating[Any]]],
]
MatrixLike: TypeAlias = Union[
    Sequence[Sequence[Union[int, float]]],
    np.ndarray[
        Tuple[int, int],
        np.dtype[Union[np.integer[Any], np.floating[Any]]],
    ],
]
PolygonLike: TypeAlias = Union[
    MatrixLike,
    pv.PolyData,
    Polygon,
]
PolyLineLike: TypeAlias = Union[
    LineString,
    MatrixLike,
    pv.PolyData,
]
