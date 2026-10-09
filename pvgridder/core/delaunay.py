from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from ._base import MeshBase, MeshItem


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Optional

    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from .._typing import DataSetLike, PolygonLike, PolyLineLike, VectorLike


class DelaunayMesh2D(MeshBase):
    """
    2D Delaunay mesh class.

    Parameters
    ----------
    polygon : PolygonLike
        Polygon defining the 2D domain for the Delaunay mesh.
    min_cellsize : scalar, optional
        Minimum cell size for the Delaunay mesh.
    max_cellsize : scalar, optional
        Maximum cell size for the Delaunay mesh.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    def __init__(
        self,
        polygon: PolygonLike,
        min_cellsize: Optional[float] = None,
        max_cellsize: Optional[float] = None,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a 2D Delaunay mesh."""
        super().__init__(default_group, ignore_groups)

        self._polygon = self._convert_polygon(polygon)
        self._min_cellsize = min_cellsize
        self._max_cellsize = max_cellsize

    def add_point(
        self,
        point: VectorLike,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        return self

    def add_polygon(
        self,
        polygon: PolygonLike,
        cellsize: Optional[float] = None,
        hole: bool = False,
        group: Optional[str] = None,
    ) -> Self:
        return self

    def add_polyline(
        self,
        line: PolyLineLike,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        return self

    def generate_mesh(
        self,
        tolerance: float = 1.0e-8,
        engine: Literal["gmsh", "occ"] = "gmsh",
        optimization: Optional[str] = None,
    ) -> pv.UnstructuredGrid:
        mesh = pv.UnstructuredGrid()

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))
