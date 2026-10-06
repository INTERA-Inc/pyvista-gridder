from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
import vtk


if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import ArrayLike, NDArray


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


class QuadTree:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        max_depth: int = 4,
    ) -> None:
        """Initialize a quadtree."""
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self._roots = []

        for y1, y2 in zip(self.y[:-1], self.y[1:]):
            for x1, x2 in zip(self.x[:-1], self.x[1:]):
                self.roots.append(QuadNode(x1, x2, y1, y2, depth=0))

    def add_point(
        self,
        point: tuple[float, float],
        depth: Optional[int] = None,
    ) -> None:
        """Refine mesh containing point."""
        depth = self.max_depth if depth is None else depth

        for root in self.roots:
            self._refine_point(root, point, depth)

    def add_polyline(
        self,
        line: list[tuple[float, float]],
        depth: Optional[int] = None,
    ) -> None:
        """Refine mesh intersected by polyline."""
        depth = self.max_depth if depth is None else depth

        for pointa, pointb in zip(line[:-1], line[1:]):
            for root in self.roots:
                self._refine_segment(root, pointa, pointb, depth)

    def generate_mesh(self, balance: bool = True) -> pv.UnstructuredGrid:
        """
        Generate a QuadTree mesh.

        Parameters
        ----------
        balance : bool, default True
            If True, balance the tree so that adjacent cells differ in depth by at most
            one.

        Returns
        -------
        pyvista.UnstructuredGrid
            QuadTree mesh.
        
        """
        if balance:
            # Iteratively subdivides coarse cells until max level difference between
            # adjacent leaves is <= 1
            while True:
                leaves: list[QuadNode] = []

                for root in self.roots:
                    self._get_leaves(root, leaves)
    
                subdivided_any = False

                for i in range(len(leaves)):
                    for j in range(i + 1, len(leaves)):
                        a, b = leaves[i], leaves[j]
    
                        # Only check pairs with depth difference >= 2
                        if abs(a.depth - b.depth) <= 1:
                            continue
    
                        if self._are_neighbors(a, b):
                            # Subdivide the coarser leaf
                            coarser = a if a.depth < b.depth else b
                            coarser.subdivide()
                            subdivided_any = True
    
                if not subdivided_any:
                    break

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
        global_offset = 0

        for root_idx, root_node in enumerate(self.roots):
            htg.InitializeNonOrientedCursor(cursor, root_idx, True)
            cursor.SetGlobalIndexStart(global_offset)

            self._build_vtk_tree(root_node, cursor)
            global_offset += cursor.GetTree().GetNumberOfVertices()

        # Extract mesh
        geometry = vtk.vtkHyperTreeGridGeometry()
        geometry.SetInputData(htg)
        geometry.Update()
        mesh = pv.wrap(geometry.GetOutput()).cast_to_unstructured_grid()

        return mesh

    @staticmethod
    def _are_neighbors(a: QuadNode, b: QuadNode) -> bool:
        """Returns True if leaf A and leaf B share a horizontal or vertical edge."""
        x_touch = (abs(a.xmax - b.xmin) < 1.0e-9) or (abs(a.xmin - b.xmax) < 1.0e-9)
        y_overlap = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) > 1.0e-9

        if x_touch and y_overlap:
            return True

        y_touch = (abs(a.ymax - b.ymin) < 1.0e-9) or (abs(a.ymin - b.ymax) < 1.0e-9)
        x_overlap = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) > 1.0e-9

        return y_touch and x_overlap

    def _build_vtk_tree(
        self,
        node: QuadNode,
        cursor: vtk.vtkHyperTreeGridNonOrientedCursor,
    ) -> None:
        """Recursively applies VTK cursor subdivision based on internal tree topology."""
        if not node.is_leaf:
            cursor.SubdivideLeaf()

            for i, node in enumerate(node.children):
                cursor.ToChild(i)
                self._build_vtk_tree(node, cursor)
                cursor.ToParent()

    def _get_leaves(self, node: QuadNode, leaves: list[QuadNode]) -> None:
        """Recursively collects active leaf nodes."""
        if node.is_leaf:
            leaves.append(node)
            
        else:
            for child in node.children:
                self._get_leaves(child, leaves)

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

    @property
    def max_depth(self) -> int:
        """Get the maximum depth."""
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        """Set the maximum depth."""
        self._max_depth = value

    @property
    def roots(self) -> list[QuadNode]:
        """Get the root nodes."""
        return self._roots

    @property
    def x(self) -> NDArray:
        """Get the X coordinates of the points."""
        return self._x

    @x.setter
    def x(self, value: ArrayLike) -> None:
        """Set the X coordinates of the points."""
        self._x = np.asanyarray(value)

    @property
    def y(self) -> NDArray:
        """Get the Y coordinates of the points."""
        return self._y

    @y.setter
    def y(self, value: ArrayLike) -> None:
        """Set the Y coordinates of the points."""
        self._y = np.asanyarray(value)
