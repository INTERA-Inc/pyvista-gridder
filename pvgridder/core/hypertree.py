from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
import vtk

from ._base import MeshBase, MeshItem


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Optional

    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


class QuadNode:
    """
    Quadtree node.

    Parameters
    ----------
    xmin : float
        Minimum X coordinate of the node's bounding box.
    xmax : float
        Maximum X coordinate of the node's bounding box.
    ymin : float
        Minimum Y coordinate of the node's bounding box.
    ymax : float
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

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Return True if the point is within the node's bounding box."""
        x, y = point

        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def intersects_segment(
        self,
        pointa: tuple[float, float],
        pointb: tuple[float, float],
    ) -> bool:
        """Return True if the line segment intersects the node's bounding box."""
        x1, y1 = pointa
        x2, y2 = pointb

        if max(x1, x2) < self.xmin or min(x1, x2) > self.xmax or max(y1, y2) < self.ymin or min(y1, y2) > self.ymax:
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
    mesh : pyvista.ImageData | pyvista.RectilinearGrid
        Base mesh for the quadtree.
    max_depth : int, default 4
        Maximum depth of the quadtree.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    def __init__(
        self,
        mesh: pv.ImageData | pv.RectilinearGrid,
        *,
        max_depth: int = 4,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a quadtree."""
        super().__init__(default_group, ignore_groups)

        if isinstance(mesh, pv.ImageData):
            mesh = mesh.cast_to_rectilinear_grid()

        self.max_depth = max_depth
        self._mesh = mesh.copy()
        self._roots = []

        for y1, y2 in zip(self.y[:-1], self.y[1:]):
            for x1, x2 in zip(self.x[:-1], self.x[1:]):
                self.roots.append(QuadNode(x1, x2, y1, y2, depth=0))

    def add_point(
        self,
        point: tuple[float, float],
        depth: Optional[int] = None,
    ) -> Self:
        """Refine cell containing point."""
        depth = min(depth, self.max_depth) if depth else self.max_depth

        for root in self.roots:
            self._refine_point(root, point, depth)

        return self

    def add_polyline(
        self,
        line: list[tuple[float, float]],
        depth: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        """Refine cells intersected by polyline."""
        depth = min(depth, self.max_depth) if depth else self.max_depth

        for pointa, pointb in zip(line[:-1], line[1:]):
            for root in self.roots:
                self._refine_segment(root, pointa, pointb, depth)

        if group:
            line_mesh = pv.MultipleLines(np.insert(line, 2, 0.0, axis=1))
            item = MeshItem(line_mesh, group=group)
            self.items.append(item)

        return self

    def generate_mesh(
        self,
        balance: bool = False,
        conformal: bool = False,
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
            mesh = pv.wrap(geometry.GetOutput()).clean().cast_to_unstructured_grid()

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
                        (x, y)
                        for y in vertical_vertices[x]
                        if ymin < y < ymax
                    )

                for y in (ymin, ymax):
                    horizontal_hanging_nodes.update(
                        (x, y)
                        for x in horizontal_vertices[y]
                        if xmin < x < xmax
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
                    *((x, ymin) for x in sorted(horizontal_vertices[ymin]) if xmin <= x <= xmax),
                    *((xmax, y) for y in sorted(vertical_vertices[xmax]) if ymin < y <= ymax),
                    *((x, ymax) for x in sorted(horizontal_vertices[ymax], reverse=True) if xmin <= x < xmax),
                    *((xmin, y) for y in sorted(vertical_vertices[xmin], reverse=True) if ymin < y < ymax),
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
        groups = dict(self.mesh.user_dict) or {}
        group_array = np.asanyarray(
            mesh.cell_data.get(
                "CellGroup",
                self._initialize_group_array(mesh, groups),
            )
        )

        for item in self.items:
            if isinstance(item.mesh, pv.PolyData):
                if item.mesh.n_lines > 0:
                    for polyline in split_lines(item.mesh, as_lines=True):
                        for pointa, pointb in zip(polyline.points[:-1], polyline.points[1:]):
                            cids = mesh.find_cells_intersecting_line(pointa, pointb)

                            if cids.size > 0:
                                group_array[cids] = self._get_group_number(item.group, groups)

        mesh.cell_data["CellGroup"] = group_array
        mesh.user_dict["CellGroup"] = groups

        return mesh

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
        point: tuple[float, float],
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

    def _refine_segment(
        self,
        node: QuadNode,
        pointa: tuple[float, float],
        pointb: tuple[float, float],
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

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        """Set the maximum depth."""
        self._max_depth = value

    @property
    def mesh(self) -> pv.DataSet:
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
