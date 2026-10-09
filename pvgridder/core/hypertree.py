from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import pyvista as pv
import vtk
from shapely import LineString, Polygon, box, contains_xy, get_coordinates

from ._base import MeshBase, MeshItem


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional

    from numpy.typing import NDArray
    from typing_extensions import Self

    from .._typing import (
        DataSetLike,
        MatrixLike,
        PolygonLike,
        PolyLineLike,
        StructuredLike,
        VectorLike,
    )


class QuadNode:
    """
    Quadtree node.

    Parameters
    ----------
    xmin : scalar
        Minimum X coordinate of the node's bounding box.
    xmax : scalar
        Maximum X coordinate of the node's bounding box.
    ymin : scalar
        Minimum Y coordinate of the node's bounding box.
    ymax : scalar
        Maximum Y coordinate of the node's bounding box.
    depth : int, default 0
        Depth of the node in the quadtree.

    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        depth: int = 0,
    ) -> None:
        """Initialize a quadtree node."""
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.depth = depth
        self._children = []

    def contains_point(self, point: VectorLike) -> bool:
        """Return True if the point is within the bounds, allowing for roundoff."""
        x, y = point[:2]
        epsilon = 8.0 * np.finfo(float).eps
        xtol = epsilon * max(abs(self.xmin), abs(self.xmax))
        ytol = epsilon * max(abs(self.ymin), abs(self.ymax))

        return (
            self.xmin - xtol <= x <= self.xmax + xtol
            and self.ymin - ytol <= y <= self.ymax + ytol
        )

    def intersects_polygon(
        self,
        points: MatrixLike | Polygon,
    ) -> bool:
        """Return True if the filled polygon touches or overlaps the node bounds."""
        polygon = points if isinstance(points, Polygon) else Polygon(points)

        return bool(polygon.intersects(box(self.xmin, self.ymin, self.xmax, self.ymax)))

    def intersects_segment(
        self,
        pointa: VectorLike,
        pointb: VectorLike,
    ) -> bool:
        """Return True if the line segment intersects the node's bounding box."""
        x1, y1 = pointa[:2]
        x2, y2 = pointb[:2]

        if (
            max(x1, x2) < self.xmin
            or min(x1, x2) > self.xmax
            or max(y1, y2) < self.ymin
            or min(y1, y2) > self.ymax
        ):
            return False

        # Liang-Barsky algorithm
        u1, u2 = 0.0, 1.0
        dx, dy = x2 - x1, y2 - y1
        iterables = zip(
            [-dx, dx, -dy, dy],
            [x1 - self.xmin, self.xmax - x1, y1 - self.ymin, self.ymax - y1],
        )

        for p, q in iterables:
            if p == 0:
                if q < 0:
                    return False

            else:
                t = q / p

                if p < 0:
                    u1 = max(u1, t)

                else:
                    u2 = min(u2, t)

        return u1 <= u2

    def subdivide(self) -> None:
        """Subdivide rectilinear cell into 4 child quadrants (SW, SE, NW, NE)."""
        if not self.is_leaf:
            return

        xmid = 0.5 * (self.xmin + self.xmax)
        ymid = 0.5 * (self.ymin + self.ymax)
        depth = self.depth + 1
        self._children = [
            QuadNode(self.xmin, xmid, self.ymin, ymid, depth),  # 0: SW
            QuadNode(xmid, self.xmax, self.ymin, ymid, depth),  # 1: SE
            QuadNode(self.xmin, xmid, ymid, self.ymax, depth),  # 2: NW
            QuadNode(xmid, self.xmax, ymid, self.ymax, depth),  # 3: NE
        ]

    @property
    def children(self) -> list[QuadNode]:
        """Return the list of child nodes."""
        return self._children

    @property
    def depth(self) -> int:
        """Get the depth of the node in the quadtree."""
        return self._depth

    @depth.setter
    def depth(self, value: int) -> None:
        """Set the depth of the node in the quadtree."""
        self._depth = value

    @property
    def is_leaf(self) -> bool:
        """Return True if the node is a leaf (i.e., has no children)."""
        return len(self.children) == 0

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate."""
        return self._xmin

    @xmin.setter
    def xmin(self, value: float) -> None:
        """Set the minimum X coordinate."""
        self._xmin = value

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate."""
        return self._xmax

    @xmax.setter
    def xmax(self, value: float) -> None:
        """Set the maximum X coordinate."""
        self._xmax = value

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate."""
        return self._ymin

    @ymin.setter
    def ymin(self, value: float) -> None:
        """Set the minimum Y coordinate."""
        self._ymin = value

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate."""
        return self._ymax

    @ymax.setter
    def ymax(self, value: float) -> None:
        """Set the maximum Y coordinate."""
        self._ymax = value


class QuadTree(MeshBase):
    """
    QuadTree class.

    Parameters
    ----------
    mesh : StructuredLike
        Base mesh for the quadtree.
    max_depth : int, default 5
        Maximum depth of the quadtree.
    min_cellsize : scalar, optional
        Lower bound on refined cell edge lengths. Supersedes *max_depth* if specified.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    def __init__(
        self,
        mesh: StructuredLike,
        *,
        max_depth: int = 5,
        min_cellsize: Optional[float] = None,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a quadtree."""
        super().__init__(default_group, ignore_groups)

        if mesh.dimensions[2] != 1:
            raise ValueError("could not create a quadtree with a 3D mesh")

        if isinstance(mesh, pv.ImageData):
            mesh = mesh.cast_to_rectilinear_grid()

        elif isinstance(mesh, pv.StructuredGrid):
            x = mesh.x[:, 0]
            y = mesh.y[0, :]
            mesh_ = pv.RectilinearGrid(x, y, [0.0])
            mesh_.user_dict.update(mesh.user_dict)

            for k, v in mesh.cell_data.items():
                mesh_.cell_data[k] = v

            mesh = mesh_

        self._mesh = mesh.copy()
        self._max_depth = self._get_depth(max_depth, min_cellsize)
        self._boundary_polygon = None
        self._roots = []

        for y1, y2 in zip(self.y[:-1], self.y[1:]):
            for x1, x2 in zip(self.x[:-1], self.x[1:]):
                self.roots.append(QuadNode(x1, x2, y1, y2, depth=0))

    def add_boundary_polygon(
        self,
        polygon: PolygonLike,
        depth: int = 0,
        cellsize: Optional[float] = None,
    ) -> Self:
        """
        Add a boundary polygon to the quadtree.

        Parameters
        ----------
        polygon : PolygonLike
            Boundary polygon to add.
        depth : int, default 0
            Depth for refinement of the boundary polygon. No refinement by default.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.

        Returns
        -------
        Self
            Self (for daisy chaining).

        Notes
        -----
        Only one boundary polygon can be active at a time.

        """
        if self._boundary_polygon is not None:
            raise ValueError("could not add a second boundary polygon")

        polygon = self._convert_polygon(polygon)
        depth = min(depth, self.max_depth)
        depth = self._get_depth(depth, cellsize)

        if depth > 0:
            self.add_polygon(Polygon(polygon.exterior), boundary_only=True, depth=depth)

            for interior in polygon.interiors:
                self.add_polygon(Polygon(interior), boundary_only=True, depth=depth)

        self._boundary_polygon = polygon

        return self

    def add_circle(
        self,
        radius: float,
        *,
        center: Optional[VectorLike] = None,
        boundary_only: bool = False,
        depth: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cells contained within the circle.

        Parameters
        ----------
        radius : scalar
            Radius of the circle.
        center : VectorLike, optional
            Center of the circle.
        boundary_only : bool, default False
            If True, only refine cells intersecting the boundary.
        depth : int, optional
            Depth for refinement.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        Notes
        -----
        The circle is approximated by a 64-sided polygon for refinement purposes.

        """
        center_ = np.zeros(2) if center is None else np.asanyarray(center[:2])
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        points = center_ + radius * np.column_stack((np.cos(angles), np.sin(angles)))

        return self.add_polygon(
            points, depth=depth, group=group, boundary_only=boundary_only
        )

    def add_point(
        self,
        point: VectorLike,
        *,
        depth: Optional[int] = None,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cell containing point.

        Parameters
        ----------
        point : VectorLike
            Point to refine.
        depth : int, optional
            Depth for refinement.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        depth = min(depth, self.max_depth) if depth is not None else self.max_depth
        depth = self._get_depth(depth, cellsize)

        for root in self.roots:
            self._refine_point(root, point, depth)

        if group:
            mesh = pv.PolyData(np.atleast_2d(np.append(point[:2], 0.0)))
            item = MeshItem(mesh, group=group)
            self.items.append(item)

        return self

    def add_polygon(
        self,
        polygon: PolygonLike,
        *,
        boundary_only: bool = False,
        depth: Optional[int] = None,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cells contained within the polygon.

        Parameters
        ----------
        polygon : PolygonLike
            Polygon to refine within. If a PolyData is provided, the first polygon face
            will be used.
        boundary_only : bool, default False
            If True, only refine cells intersected by the polygon boundary.
        depth : int, optional
            Depth for refinement.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        polygon = self._convert_polygon(polygon)
        depth = self.max_depth if depth is None else min(depth, self.max_depth)
        depth = self._get_depth(depth, cellsize)

        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            raise ValueError("could not create a valid polygon from the given points")

        points = get_coordinates(polygon.exterior)

        if boundary_only:
            for pointa, pointb in zip(points[:-1], points[1:]):
                for root in self.roots:
                    self._refine_segment(root, pointa, pointb, depth)

        else:
            for root in self.roots:
                self._refine_polygon(root, polygon, depth)

        if group:
            mesh = pv.PolyData().from_irregular_faces(
                np.insert(points, 2, 0.0, axis=1),
                [np.arange(len(points))],
            )
            self.items.append(MeshItem(mesh, group=group))

        return self

    def add_polyline(
        self,
        line: PolyLineLike,
        depth: Optional[int] = None,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cells intersected by polyline.

        Parameters
        ----------
        line : PolyLineLike
            Polyline defining the path for refinement.
        depth : int, optional
            Depth for refinement.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import split_lines

        depth = min(depth, self.max_depth) if depth is not None else self.max_depth
        depth = self._get_depth(depth, cellsize)

        if isinstance(line, pv.PolyData):
            if line.n_lines == 0:
                raise ValueError(
                    "could not create a valid polyline from the given PolyData"
                )

            line = split_lines(line, as_lines=False)[0].points[:, :2]

        elif isinstance(line, LineString):
            line = get_coordinates(line)

        for pointa, pointb in zip(line[:-1], line[1:]):
            for root in self.roots:
                self._refine_segment(root, pointa, pointb, depth)

        if group:
            mesh = pv.MultipleLines(np.insert(line, 2, 0.0, axis=1))
            item = MeshItem(mesh, group=group)
            self.items.append(item)

        return self

    def add_rectangle(
        self,
        dx: float,
        dy: float,
        origin: Optional[VectorLike] = None,
        boundary_only: bool = False,
        depth: Optional[int] = None,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cells contained within the rectangle.

        Parameters
        ----------
        dx : scalar
            Width of the rectangle.
        dy : scalar
            Height of the rectangle.
        origin : VectorLike, optional
            Origin of the rectangle.
        boundary_only : bool, optional
            If True, only refine cells intersected by the boundary of the rectangle.
        depth : int, optional
            Depth for refinement.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        origin_ = np.zeros(2) if origin is None else np.asanyarray(origin[:2])
        points = origin_ + [(0.0, 0.0), (dx, 0.0), (dx, dy), (0.0, dy)]

        return self.add_polygon(
            points, depth=depth, cellsize=cellsize, group=group, boundary_only=boundary_only
        )

    def add_square(
        self,
        dx: float,
        origin: Optional[VectorLike] = None,
        boundary_only: bool = False,
        depth: Optional[int] = None,
        cellsize: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Refine cells contained within the square.

        Parameters
        ----------
        dx : scalar
            Side length of the square.
        origin : VectorLike, optional
            Origin of the square.
        boundary_only : bool, optional
            If True, only refine cells intersected by the boundary of the square.
        depth : int, optional
            Depth for refinement.
        cellsize : scalar, optional
            Target cell size for refinement. Supersedes *depth* if specified.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        return self.add_rectangle(
            dx=dx,
            dy=dx,
            origin=origin,
            boundary_only=boundary_only,
            depth=depth,
            cellsize=cellsize,
            group=group,
        )

    def generate_mesh(
        self,
        balance: bool = False,
        conformal: bool = False,
        tolerance: float = 1.0e-8,
    ) -> pv.UnstructuredGrid:
        """
        Generate a QuadTree mesh.

        Parameters
        ----------
        balance : bool, default False
            If True, balance the tree so that adjacent cells differ in depth by at most
            one.
        conformal : bool, default False
            If True, generate a conforming mesh (i.e., polygons with hanging nodes
            instead of quads).
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points.

        Returns
        -------
        pyvista.UnstructuredGrid
            QuadTree mesh.

        """
        from .. import split_lines

        if balance:
            self._balance()

        if not conformal:
            # Build VTK HyperTreeGrid
            nx = self.x.size
            ny = self.y.size

            htg = vtk.vtkHyperTreeGrid()
            htg.Initialize()
            htg.SetDimensions((nx, ny, 1))
            htg.SetBranchFactor(2)

            xValues = vtk.vtkDoubleArray()
            xValues.SetNumberOfValues(nx)
            for i, x in enumerate(self.x):
                xValues.SetValue(i, x)
            htg.SetXCoordinates(xValues)

            yValues = vtk.vtkDoubleArray()
            yValues.SetNumberOfValues(ny)
            for i, y in enumerate(self.y):
                yValues.SetValue(i, y)
            htg.SetYCoordinates(yValues)

            zValues = vtk.vtkDoubleArray()
            zValues.SetNumberOfValues(1)
            zValues.SetValue(0, 0.0)
            htg.SetZCoordinates(zValues)

            # Traversal via non-oriented cursors
            cursor = vtk.vtkHyperTreeGridNonOrientedCursor()
            offset = 0

            for root_idx, root_node in enumerate(self.roots):
                htg.InitializeNonOrientedCursor(cursor, root_idx, True)
                cursor.SetGlobalIndexStart(offset)

                self._build_vtk_tree(root_node, cursor)
                offset += cursor.GetTree().GetNumberOfVertices()

            # Extract mesh
            geometry = vtk.vtkHyperTreeGridGeometry()
            geometry.SetInputData(htg)
            geometry.Update()
            mesh = pv.wrap(geometry.GetOutput()).cast_to_unstructured_grid()

        else:
            # Collect all leaf nodes to determine the maximum depth for scaling
            raw_leaves: list[QuadNode] = []

            for root in self.roots:
                self._get_leaves(root, raw_leaves)

            scale = 1 << max((node.depth for node in raw_leaves), default=0)

            # Determine the bounds of each leaf node in the scaled coordinate system
            nx = self.x.size - 1
            leaves: list[tuple[QuadNode, int, int, int, int]] = []

            for root_idx, root in enumerate(self.roots):
                root_x = root_idx % nx
                root_y = root_idx // nx
                self._get_leaf_bounds(
                    root,
                    root_x * scale,
                    (root_x + 1) * scale,
                    root_y * scale,
                    (root_y + 1) * scale,
                    leaves,
                )

            # Organize vertices by their coordinates
            vertical_vertices: dict[int, set[int]] = defaultdict(set)
            horizontal_vertices: dict[int, set[int]] = defaultdict(set)

            for _, xmin, xmax, ymin, ymax in leaves:
                vertical_vertices[xmin].update((ymin, ymax))
                vertical_vertices[xmax].update((ymin, ymax))
                horizontal_vertices[ymin].update((xmin, xmax))
                horizontal_vertices[ymax].update((xmin, xmax))

            # Flag hanging nodes within each leaf's bounds
            vertical_hanging_nodes: set[tuple[int, int]] = set()
            horizontal_hanging_nodes: set[tuple[int, int]] = set()

            for _, xmin, xmax, ymin, ymax in leaves:
                for x in (xmin, xmax):
                    vertical_hanging_nodes.update(
                        (x, y) for y in vertical_vertices[x] if ymin < y < ymax
                    )

                for y in (ymin, ymax):
                    horizontal_hanging_nodes.update(
                        (x, y) for x in horizontal_vertices[y] if xmin < x < xmax
                    )

            # Initialize data structures for mesh construction
            points: list[tuple[float, float, float]] = []
            point_ids: dict[tuple[int, int], int] = {}
            point_hanging_node_types: list[int] = []
            cells: list[int] = []

            def get_point_id(idx: int, idy: int) -> int:
                key = (idx, idy)

                if key in point_ids:
                    return point_ids[key]

                x_cell = min(idx // scale, self.x.size - 2)
                y_cell = min(idy // scale, self.y.size - 2)
                x_fraction = (idx - x_cell * scale) / scale
                y_fraction = (idy - y_cell * scale) / scale
                x_value = self.x[x_cell] + x_fraction * (
                    self.x[x_cell + 1] - self.x[x_cell]
                )
                y_value = self.y[y_cell] + y_fraction * (
                    self.y[y_cell + 1] - self.y[y_cell]
                )
                point_ids[key] = len(points)
                point_hanging_node_types.append(
                    0
                    if key in vertical_hanging_nodes
                    else 1
                    if key in horizontal_hanging_nodes
                    else -1
                )
                points.append((float(x_value), float(y_value), 0.0))

                return point_ids[key]

            # Construct mesh cells from leaf boundaries
            for _, xmin, xmax, ymin, ymax in leaves:
                boundary = [
                    *(
                        (x, ymin)
                        for x in sorted(horizontal_vertices[ymin])
                        if xmin <= x <= xmax
                    ),
                    *(
                        (xmax, y)
                        for y in sorted(vertical_vertices[xmax])
                        if ymin < y <= ymax
                    ),
                    *(
                        (x, ymax)
                        for x in sorted(horizontal_vertices[ymax], reverse=True)
                        if xmin <= x < xmax
                    ),
                    *(
                        (xmin, y)
                        for y in sorted(vertical_vertices[xmin], reverse=True)
                        if ymin < y < ymax
                    ),
                ]
                point_ids_for_cell = [get_point_id(x, y) for x, y in boundary]
                cells.extend((len(point_ids_for_cell), *point_ids_for_cell))

            celltypes = np.full(len(leaves), pv.CellType.POLYGON, dtype=np.uint8)
            mesh = pv.UnstructuredGrid(
                np.asarray(cells, dtype=np.int64),
                celltypes,
                np.asarray(points, dtype=float).reshape(-1, 3),
            )
            mesh.point_data["HangingNode"] = np.array(
                point_hanging_node_types,
                dtype=np.int8,
            )

        centers = mesh.cell_centers().points
        idx = np.searchsorted(self.x, centers[:, 0], side="right") - 1
        idy = np.searchsorted(self.y, centers[:, 1], side="right") - 1
        mesh.cell_data["vtkOriginalCellIds"] = idx + idy * (self.x.size - 1)

        for k, v in self.mesh.cell_data.items():
            mesh.cell_data[k] = v[mesh.cell_data["vtkOriginalCellIds"]]

        # Generate cell groups
        mesh = mesh.clean().cast_to_unstructured_grid()
        groups = dict(self.mesh.user_dict.get("CellGroup", {}))
        group_array = np.asanyarray(
            mesh.cell_data.get(
                "CellGroup",
                self._initialize_group_array(mesh, groups),
            )
        )

        for item in self.items:
            if isinstance(item.mesh, pv.PolyData):
                # Polyline
                if item.mesh.n_lines > 0:
                    for polyline in split_lines(item.mesh, as_lines=True):
                        points_ = polyline.points

                        for pointa, pointb in zip(points_[:-1], points_[1:]):
                            cids = mesh.find_cells_along_line(pointa, pointb)

                            if cids.size > 0:
                                group_array[cids] = self._get_group_number(
                                    item.group, groups
                                )

                # Polygon
                elif item.mesh.n_faces > 0:
                    for face in item.mesh.irregular_faces:
                        polygon = Polygon(item.mesh.points[face, :2])
                        mask = contains_xy(polygon, centers[:, 0], centers[:, 1])

                        if mask.any():
                            group_array[mask] = self._get_group_number(
                                item.group, groups
                            )

                # Point
                else:
                    point = item.mesh.points[0]
                    mask = np.isclose(
                        np.tile(point, reps=(mesh.n_points, 1)),
                        mesh.points,
                    ).all(axis=1)

                    if mask.any():
                        pid = np.flatnonzero(mask)[0]
                        cid = mesh.point_cell_ids(pid)

                    else:
                        cid = mesh.find_containing_cell([point])
                        cid = [cid] if cid >= 0 else []

                    cid = cast(list[int], cid)

                    if len(cid) >= 0:
                        group_array[cid] = self._get_group_number(item.group, groups)

        mesh.cell_data["CellGroup"] = group_array
        mesh.user_dict["CellGroup"] = groups
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        # Handle boundary polygon if it exists
        if self._boundary_polygon is not None:
            mask = contains_xy(self._boundary_polygon, centers[:, 0], centers[:, 1])
            mesh = mesh.extract_cells(mask)

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))

    def _balance(self) -> None:
        """Enforce 2:1 balance rule via hash-bucket edge sweeping."""
        nx = self.x.size - 1

        while True:
            # Gather active leaves to determine current max tree depth
            raw_leaves: list[QuadNode] = []

            for root in self.roots:
                self._get_leaves(root, raw_leaves)

            max_depth = max((node.depth for node in raw_leaves), default=0)
            scale = 1 << max_depth

            # Assign integer coordinates to all leaf bounds
            leaves_bounded: list[tuple[QuadNode, int, int, int, int]] = []

            for root_idx, root in enumerate(self.roots):
                rx = root_idx % nx
                ry = root_idx // nx
                x0, x1 = rx * scale, (rx + 1) * scale
                y0, y1 = ry * scale, (ry + 1) * scale
                self._get_leaf_bounds(root, x0, x1, y0, y1, leaves_bounded)

            # Bucket leaf boundary segments into hashtables
            v_left, v_right = defaultdict(list), defaultdict(list)
            h_bot, h_top = defaultdict(list), defaultdict(list)

            for node, x0, x1, y0, y1 in leaves_bounded:
                v_left[x0].append((y0, y1, node))
                v_right[x1].append((y0, y1, node))
                h_bot[y0].append((x0, x1, node))
                h_top[y1].append((x0, x1, node))

            # Sweep vertical and horizontal boundaries
            subdivided = self._sweep_edges(v_left, v_right)
            subdivided |= self._sweep_edges(h_bot, h_top)

            # Exit when all adjacent neighbor pairs satisfy delta_level <= 1
            if not subdivided:
                break

    def _build_vtk_tree(
        self,
        node: QuadNode,
        cursor: vtk.vtkHyperTreeGridNonOrientedCursor,
    ) -> None:
        """Recursively apply VTK cursor subdivision based on internal tree topology."""
        if not node.is_leaf:
            cursor.SubdivideLeaf()

            for i, node in enumerate(node.children):
                cursor.ToChild(i)
                self._build_vtk_tree(node, cursor)
                cursor.ToParent()

    def _get_depth(self, depth: int, cellsize: float | None) -> int:
        """Get the quadtree depth corresponding to the given cell size."""
        if cellsize is None:
            return depth

        mesh_min_cellsize = min((np.diff(self.x).min(), np.diff(self.y).min()))
        
        return max(0, int(np.floor(np.log2(mesh_min_cellsize) - np.log2(cellsize))))

    def _get_leaves(self, node: QuadNode, leaves: list[QuadNode]) -> None:
        """Recursively collect active leaf nodes."""
        if node.is_leaf:
            leaves.append(node)

        else:
            for child in node.children:
                self._get_leaves(child, leaves)

    def _get_leaf_bounds(
        self,
        node: QuadNode,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
        leaves: list[tuple[QuadNode, int, int, int, int]],
    ) -> None:
        """Collect leaves with integer bounds on the finest-depth root grid."""
        if node.is_leaf:
            leaves.append((node, xmin, xmax, ymin, ymax))
            return

        xmid = (xmin + xmax) // 2
        ymid = (ymin + ymax) // 2

        for child, bounds in zip(
            node.children,
            (
                (xmin, xmid, ymin, ymid),
                (xmid, xmax, ymin, ymid),
                (xmin, xmid, ymid, ymax),
                (xmid, xmax, ymid, ymax),
            ),
        ):
            self._get_leaf_bounds(child, *bounds, leaves)

    def _refine_point(
        self,
        node: QuadNode,
        point: VectorLike,
        depth: int,
    ) -> None:
        """Refine the quadtree at the given point up to the specified depth."""
        if not node.contains_point(point):
            return

        if node.depth < depth:
            if node.is_leaf:
                node.subdivide()

            for child in node.children:
                self._refine_point(child, point, depth)

    def _refine_polygon(
        self,
        node: QuadNode,
        polygon: Polygon,
        depth: int,
    ) -> None:
        """Refine the quadtree within the given polygon up to the specified depth."""
        if node.depth >= depth or not node.intersects_polygon(polygon):
            return

        node.subdivide()

        for child in node.children:
            self._refine_polygon(child, polygon, depth)

    def _refine_segment(
        self,
        node: QuadNode,
        pointa: VectorLike,
        pointb: VectorLike,
        depth: int,
    ) -> None:
        """Refine the quadtree along the given segment up to the specified depth."""
        if not node.intersects_segment(pointa, pointb):
            return

        if node.depth < depth:
            if node.is_leaf:
                node.subdivide()

            for child in node.children:
                self._refine_segment(child, pointa, pointb, depth)

    @staticmethod
    def _sweep_edges(
        left_dict: dict[int, list[tuple[int, int, QuadNode]]],
        right_dict: dict[int, list[tuple[int, int, QuadNode]]],
    ) -> bool:
        """Sweeps coincident 1D boundary segments using two pointers."""
        subdivided_any = False

        # Intersect matching grid line coordinates across the entire domain
        for coord in left_dict.keys() & right_dict.keys():
            left = sorted(left_dict[coord], key=lambda item: item[0])
            right = sorted(right_dict[coord], key=lambda item: item[0])

            i = j = 0
            n_left, n_right = len(left), len(right)

            while i < n_left and j < n_right:
                l_min, l_max, l_node = left[i]
                r_min, r_max, r_node = right[j]

                # Check 1D segment overlap along the shared boundary line
                if l_min < r_max and r_min < l_max:
                    diff = l_node.depth - r_node.depth

                    if diff > 1:
                        r_node.subdivide()
                        subdivided_any = True

                    elif diff < -1:
                        l_node.subdivide()
                        subdivided_any = True

                # Advance two pointers
                if l_max <= r_max:
                    i += 1

                if r_max <= l_max:
                    j += 1

        return subdivided_any

    @property
    def max_depth(self) -> int:
        """Get the maximum depth."""
        return self._max_depth

    @property
    def mesh(self) -> DataSetLike:
        """Get the base mesh."""
        return self._mesh

    @property
    def roots(self) -> list[QuadNode]:
        """Get the root nodes."""
        return self._roots

    @property
    def x(self) -> NDArray:
        """Get the X coordinates of the points."""
        return np.asanyarray(self.mesh.x)

    @property
    def y(self) -> NDArray:
        """Get the Y coordinates of the points."""
        return np.asanyarray(self.mesh.y)
