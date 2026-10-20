from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import pyvista as pv
from pyrequire import require_package
from shapely import (
    LineString,
    Point,
    Polygon,
    get_coordinates,
    get_parts,
)

from ._base import MeshBase, MeshItem


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal, Optional, Protocol

    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    from .._typing import PolygonLike, PolyLineLike, VectorLike


if TYPE_CHECKING:
    class _GmshGeometryEngine(Protocol):
        """Geometry methods shared by Gmsh's engines."""

        def addPoint(
            self,
            x: float,
            y: float,
            z: float,
            meshSize: float = 0.0,
            tag: int = -1,
        ) -> int: ...

        def addLine(self, startTag: int, endTag: int, tag: int = -1) -> int: ...

        def addCurveLoop(self, curveTags: Sequence[int], tag: int = -1) -> int: ...

        def addPlaneSurface(self, wireTags: Sequence[int], tag: int = -1) -> int: ...


class DelaunayMesh2D(MeshBase):
    """
    2D Delaunay mesh class.

    Parameters
    ----------
    polygon : PolygonLike
        Polygon defining the 2D domain for the Delaunay mesh.
    cellsize : ArrayLike, optional
        Target size along the outer boundary.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    @require_package("gmsh")
    def __init__(
        self,
        polygon: PolygonLike,
        *,
        cellsize: Optional[ArrayLike] = None,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a 2D Delaunay mesh."""
        super().__init__(default_group, ignore_groups)

        validated_polygon = self._validate_polygon(polygon)
        holes = [Polygon(interior) for interior in validated_polygon.interiors]

        self._polygon = Polygon(validated_polygon.exterior)
        self._cellsize = np.asanyarray(cellsize) if cellsize is not None else None

        for hole in holes:
            self.add_polygon(hole, cellsize=cellsize, hole=True)

    def add_circle(
        self,
        radius: float,
        *,
        center: Optional[VectorLike] = None,
        boundary_only: bool = False,
        cellsize: Optional[float] = None,
        growth: Optional[float] = None,
        hole: bool = False,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add a circle to the mesh.

        Parameters
        ----------
        radius : float
            Radius of the circle.
        center : VectorLike, optional
            Center of the circle. Defaults to the origin.
        boundary_only : bool, default False
            If True, apply the group only along the circle boundary.
        cellsize : float, optional
            Target cell size along the circle boundary.
        growth : float, optional
            Geometric cell size growth factor away from the circle.
        hole : bool, default False
            If True, the circle is treated as a hole in the mesh.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        n = max(np.ceil(2.0 * np.pi * radius / cellsize), 3) if cellsize else 64
        center_ = np.zeros(2) if center is None else np.asanyarray(center[:2])
        angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        points = center_ + radius * np.column_stack((np.cos(angles), np.sin(angles)))

        return self.add_polygon(
            points,
            boundary_only=boundary_only,
            cellsize=cellsize,
            growth=growth,
            hole=hole,
            group=group,
        )

    def add_point(
        self,
        point: VectorLike,
        *,
        cellsize: Optional[float] = None,
        constraint: int = 0,
        growth: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Embed a point in the mesh.

        Parameters
        ----------
        point : VectorLike
            Point inside the domain.
        cellsize : scalar, optional
            Target cell size at the point.
        constraint : int, default 0
            Number of constraint points to add evenly around the point.
        growth : float, optional
            Geometric cell size growth factor away from the point.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        points = self._check_point_array(point, axis=2).astype(float, copy=True)
        geometry = Point(points[:2])

        if not self._polygon.covers(geometry):
            raise ValueError("could not embed point outside the domain")

        self.items.append(
            MeshItem(
                pv.PolyData(points[None, :]),
                geometry=geometry,
                cellsize=cellsize,
                group=group,
                hole=False,
                constraint=int(constraint),
                growth=growth,
            )
        )
        return self

    def add_polygon(
        self,
        polygon: PolygonLike,
        *,
        boundary_only: bool = False,
        cellsize: Optional[ArrayLike] = None,
        growth: Optional[float] = None,
        hole: bool = False,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add a polygon to the mesh.

        Parameters
        ----------
        polygon : PolygonLike
            Polygon to add to the mesh.
        boundary_only : bool, default False
            If True, apply the polygon group only along its boundary.
        cellsize : ArrayLike, optional
            Target cell size along the polygon boundary.
        growth : float, optional
            Geometric cell size growth factor away from the polygon boundary.
        hole : bool, default False
            If True, exclude the polygon interior from the domain.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        polygon = self._validate_polygon(polygon)

        if hole:
            polygons = [
                part
                for part in get_parts(polygon.intersection(self._polygon))
                if isinstance(part, Polygon) and part.area > 0.0
            ]

        elif self._polygon.covers(polygon):
            polygons = [polygon]

        else:
            raise ValueError("could not add polygon outside the domain")

        for poly in polygons:
            cell_sizes = self._get_ring_sizes(poly.exterior, cellsize)
            points = get_coordinates(poly.exterior)[:-1]
            points = self._check_point_array(points, axis=2)
            mesh = pv.PolyData().from_irregular_faces(points, [np.arange(len(points))])
            self.items.append(
                MeshItem(
                    mesh,
                    geometry=poly,
                    cellsize=cellsize,
                    cell_sizes=cell_sizes,
                    boundary_only=boundary_only,
                    growth=growth,
                    group=group,
                    hole=hole,
                )
            )

        return self

    def add_polyline(
        self,
        line: PolyLineLike,
        *,
        cellsize: Optional[float] = None,
        growth: Optional[float] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Embed a polyline in the mesh.

        Parameters
        ----------
        line : PolyLineLike
            Polyline to add to the mesh.
        cellsize : scalar, optional
            Target cell size along the polyline.
        growth : float, optional
            Geometric cell size growth factor away from the polyline.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        line = self._validate_polyline(line)
        segments = [
            segment
            for segment in get_parts(line.intersection(self._polygon))
            if isinstance(segment, LineString) and segment.length > 0.0
        ]

        if not segments:
            raise ValueError("could not add polyline outside the domain")

        for segment in segments:
            points = get_coordinates(segment)
            points = np.column_stack((points, np.zeros(len(points))))
            self.items.append(
                MeshItem(
                    pv.MultipleLines(points),
                    geometry=segment,
                    cellsize=cellsize,
                    group=group,
                    hole=False,
                    growth=growth,
                )
            )

        return self

    def generate_mesh(
        self,
        *,
        tolerance: float = 1.0e-8,
        min_cellsize: Optional[float] = None,
        max_cellsize: Optional[float] = None,
        celltype: Literal["quad", "triangle"] = "triangle",
        algorithm: Optional[int] = None,
        optimization: Optional[Literal["Netgen", "Laplace2D", "Relocate2D"]] = None,
        engine: Literal["gmsh", "occ"] = "gmsh",
    ) -> pv.UnstructuredGrid:
        """
        Generate a 2D Delaunay mesh.

        Parameters
        ----------
        tolerance : scalar, default 1.0e-8
            Tolerance for merging duplicate mesh points.
        min_cellsize : scalar, optional
            Minimum cell size for the Delaunay mesh.
        max_cellsize : scalar, optional
            Maximum cell size for the Delaunay mesh.
        celltype : {'quad', 'triangle'}, default 'triangle'
            Preferred cell type.
        algorithm : int, optional
            Gmsh 2D algorithm. Default is 8 if *celltype* is 'quad', 6 otherwise.
        optimization : {'Netgen', 'Laplace2D', 'Relocate2D'}, optional
            Gmsh 2D optimization method.
        engine : {'gmsh', 'occ'}, default 'gmsh'
            Geometry engine:

             - 'gmsh': Gmsh built-in engine
             - 'occ': OpenCascade

        Returns
        -------
        pyvista.UnstructuredGrid
            2D Delaunay mesh.

        """
        import gmsh
        from shapely.geometry import MultiPoint
        from shapely.ops import polygonize, split, unary_union

        if min_cellsize and max_cellsize and min_cellsize > max_cellsize:
            raise ValueError("could not set min_cellsize greater than max_cellsize")

        if engine not in {"gmsh", "occ"}:
            raise ValueError(f"invalid engine '{engine}'")

        if celltype not in {"quad", "triangle"}:
            raise ValueError(f"invalid cell type '{celltype}'")

        if algorithm is None:
            algorithm = 6 if celltype == "quad" else 6

        # Extract holes and define the domain for the Delaunay mesh
        holes = [item.geometry for item in self.items if item.hole]
        domain = (
            self._polygon.difference(unary_union(holes)) if holes else self._polygon
        )

        if domain.is_empty:
            raise ValueError("could not generate a valid domain with the given holes")

        # Determine default cell size for the mesh points
        default_size = (
            max_cellsize
            if max_cellsize is not None
            else np.linalg.norm(
                np.diff(get_coordinates(self._polygon.exterior)[:, :2], axis=0),
                axis=1,
            ).max()
        )

        # Compute the characteristic sizes for the mesh points
        sizes = self._get_ring_sizes(self._polygon.exterior, self._cellsize)
        constraints, items = [domain.boundary], []
        growth_items = []

        # Process each item to determine constraints and cell sizes
        for item in self.items:
            geometry = item.geometry
            growth = getattr(item, "growth", None)

            if growth is not None and growth > 1.0:
                growth_items.append(item)

            if isinstance(geometry, Point):
                if not domain.covers(geometry):
                    raise ValueError("could not place point inside a hole")

                items.append(item)
                constraint = getattr(item, "constraint", 0)

                if constraint:
                    radius = (
                        item.cellsize if item.cellsize is not None else default_size
                    )
                    radius = min(radius, 0.95 * geometry.distance(domain.boundary))

                    if radius <= 0.0:
                        raise ValueError(
                            "could not place constraint points due to insufficient clearance from the domain boundary"
                        )

                    angles = 2.0 * np.pi * np.arange(constraint) / constraint

                    for angle in angles:
                        constraint_point = Point(
                            geometry.x + radius * np.cos(angle),
                            geometry.y + radius * np.sin(angle),
                        )
                        items.append(
                            MeshItem(
                                pv.PolyData(
                                    [[constraint_point.x, constraint_point.y, 0.0]]
                                ),
                                geometry=constraint_point,
                                cellsize=item.cellsize,
                                group=None,
                                hole=False,
                            )
                        )

            elif not item.hole:
                constraints.append(
                    geometry.boundary if isinstance(geometry, Polygon) else geometry
                )

            if isinstance(geometry, Polygon):
                for k, v in getattr(item, "cell_sizes", {}).items():
                    sizes[k] = min(sizes.get(k, v), v)

            elif item.cellsize is not None:
                for coordinate in get_coordinates(geometry):
                    key = (float(coordinate[0]), float(coordinate[1]))
                    sizes[key] = min(sizes.get(key, item.cellsize), item.cellsize)

        # Construct the network of constraints and split it with the items
        network = unary_union(constraints).intersection(domain)

        if items:
            network = split(network, MultiPoint([item.geometry for item in items]))

        # Extract lines from the network to form the polygonized faces
        def _extract_lines(geom):
            if isinstance(geom, LineString):
                yield geom

            elif hasattr(geom, "geoms"):
                for part in geom.geoms:
                    yield from _extract_lines(part)

        network_lines = list(_extract_lines(network))
        faces = [
            face
            for face in polygonize(network_lines)
            if domain.covers(face.representative_point())
        ]

        # Initialize the Gmsh engine and set mesh options
        try:
            gmsh.initialize()
            gmsh.option.setNumber("Mesh.Algorithm", algorithm)
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_cellsize or 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_cellsize or 1.0e20)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.RecombineAll", 0)

            engine_ = gmsh.model.geo if engine == "gmsh" else gmsh.model.occ
            point_tags, curve_tags = {}, {}

            # Add network lines to the Gmsh model
            for line in network_lines:
                points = get_coordinates(line)

                for pointa, pointb in zip(points[:-1], points[1:]):
                    self._add_curve(
                        engine_, pointa, pointb, curve_tags, point_tags, sizes,
                        default_size,
                    )

            network_points = set(point_tags)

            # Collect network points and initialize boundary curves and surfaces
            boundary_curves = set()
            surfaces = [
                self._add_plane_surface(
                    engine_, face, curve_tags, point_tags, sizes, default_size,
                    boundary_curves,
                )
                for face in faces
            ]

            # Add isolated item points to the Gmsh model
            for item in items:
                self._add_point(
                    engine_, item.geometry.coords[0], point_tags, sizes, default_size
                )

            # Synchronize the Gmsh model
            engine_.synchronize()

            # Build geometric growth fields
            growth_fields = []

            for item in growth_items:
                geometry = item.geometry

                if item.cellsize is not None:
                    cell_sizes = np.asarray(item.cellsize, dtype=float)
                    lc = float(
                        cell_sizes if cell_sizes.ndim == 0 else cell_sizes.min()
                    )

                elif isinstance(geometry, Polygon) and getattr(
                    item, "cell_sizes", None
                ):
                    lc = float(min(item.cell_sizes.values()))

                else:
                    lc = float(default_size)

                distance_field = gmsh.model.mesh.field.add("Distance")

                if isinstance(geometry, Point):
                    point_key = (float(geometry.x), float(geometry.y))
                    gmsh.model.mesh.field.setNumbers(
                        distance_field, "PointsList", [point_tags[point_key]]
                    )

                else:
                    feature_boundary = (
                        geometry
                        if isinstance(geometry, LineString)
                        else geometry.boundary
                    )
                    feature_curves = [
                        abs(tag)
                        for edge, tag in curve_tags.items()
                        if feature_boundary.covers(LineString(edge))
                    ]
                    
                    if not feature_curves:
                        gmsh.model.mesh.field.remove(distance_field)
                        continue

                    gmsh.model.mesh.field.setNumbers(
                        distance_field, "CurvesList", sorted(set(feature_curves))
                    )

                growth_field = gmsh.model.mesh.field.add("MathEval")
                gradient = float(item.growth) - 1.0
                expression = (
                    f"{lc:.16g} + {gradient:.16g} * F{distance_field}"
                )
                gmsh.model.mesh.field.setString(growth_field, "F", expression)
                growth_fields.append(growth_field)

            if len(growth_fields) == 1:
                gmsh.model.mesh.field.setAsBackgroundMesh(growth_fields[0])

            elif growth_fields:
                growth_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(
                    growth_field, "FieldsList", growth_fields
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(growth_field)

            # Setup physical groups
            point_groups = dict.fromkeys(point_tags.values(), self.default_group)
            curve_groups = {}
            endpoint_groups = {}
            surface_groups = defaultdict(list)

            for item in self.items:
                has_group = item.group and item.group not in self.ignore_groups

                if not has_group:
                    continue

                for point in get_coordinates(item.geometry):
                    key = (float(point[0]), float(point[1]))
                    point_tag = point_tags.get(key)

                    if point_tag is not None:
                        point_groups[point_tag] = item.group

                if isinstance(item.geometry, (LineString, Polygon)):
                    grouped_boundary = (
                        item.geometry
                        if isinstance(item.geometry, LineString)
                        else item.geometry.boundary
                    )

                    for k, v in curve_tags.items():
                        if grouped_boundary.covers(LineString(k)):
                            curve_groups[abs(v)] = item.group

                            for endpoint in k:
                                endpoint_groups[point_tags[endpoint]] = item.group

                if (
                    isinstance(item.geometry, Polygon)
                    and not item.hole
                    and not getattr(item, "boundary_only", False)
                ):
                    surface_groups[item.group] += [
                        surface
                        for face, surface in zip(faces, surfaces)
                        if item.geometry.covers(face.representative_point())
                    ]

            groups = {}
            physical_points = defaultdict(list)
            physical_curves = defaultdict(list)
            curve_physical_tags = {}
            surface_physical_tags = {}

            for k, v in point_groups.items():
                physical_points[v].append(k)

            for k, v in physical_points.items():
                physical_tag = gmsh.model.addPhysicalGroup(0, v)
                gmsh.model.setPhysicalName(0, physical_tag, k)
                groups[k] = physical_tag

            for k, v in curve_groups.items():
                physical_curves[v].append(k)

            for k, v in physical_curves.items():
                tag = gmsh.model.addPhysicalGroup(1, sorted(set(v)))
                gmsh.model.setPhysicalName(1, tag, k)
                curve_physical_tags[k] = tag

            for k, v in surface_groups.items():
                tag = gmsh.model.addPhysicalGroup(2, sorted(set(v)))
                gmsh.model.setPhysicalName(2, tag, k)
                surface_physical_tags[k] = tag

            # Precalculate internal curve midpoints for fast embedding checks
            internal_curves = [
                (
                    Point((k[0][0] + k[1][0]) * 0.5, (k[0][1] + k[1][1]) * 0.5),
                    abs(tag),
                )
                for k, tag in curve_tags.items()
                if abs(tag) not in boundary_curves
            ]

            embedded_item_points = [
                (
                    item.geometry,
                    point_tags[
                        (
                            float(item.geometry.coords[0][0]),
                            float(item.geometry.coords[0][1]),
                        )
                    ],
                )
                for item in items
                if (
                    float(item.geometry.coords[0][0]),
                    float(item.geometry.coords[0][1]),
                )
                not in network_points
            ]

            # Embed internal curves and points into the mesh surfaces
            for face, surface in zip(faces, surfaces):
                if celltype == "quad":
                    gmsh.model.mesh.setRecombine(2, surface)

                embedded_curves = [
                    tag for mid_pt, tag in internal_curves if face.covers(mid_pt)
                ]
                embedded_pts = [
                    tag
                    for pt_geom, tag in embedded_item_points
                    if face.contains(pt_geom)
                ]

                if embedded_curves:
                    gmsh.model.mesh.embed(1, embedded_curves, 2, surface)

                if embedded_pts:
                    gmsh.model.mesh.embed(0, embedded_pts, 2, surface)

            # Generate and optimize the 2D mesh
            gmsh.model.mesh.generate(2)

            if optimization:
                gmsh.model.mesh.optimize(optimization, force=True)

            # Convert to PyVista
            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()

            group_ids = {self.default_group: 0}
            max_element_tag = max(
                (int(np.max(tags, initial=0)) for tags in element_tags), default=0
            )
            element_groups = np.full(max_element_tag + 1, -1, dtype=int)

            for dim, tag in gmsh.model.getPhysicalGroups():
                group = gmsh.model.getPhysicalName(dim, tag)
                gid = group_ids.setdefault(group, len(group_ids))

                if dim < 2 and group == self.default_group:
                    continue

                for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    _, entity_elements, _ = gmsh.model.mesh.getElements(dim, entity)

                    for tags in entity_elements:
                        element_groups[tags] = gid

            max_tag = int(np.max(node_tags) if len(node_tags) > 0 else 0)
            tag_to_idx = np.empty(max_tag + 1, dtype=np.int64)
            tag_to_idx[node_tags] = np.arange(len(node_tags))

            # Map cell groups to cell array
            cells_dict, cell_offset = {}, 0
            cell_array = np.empty(sum(len(tags) for tags in element_tags), dtype=int)

            for type_, tags, nodes in zip(element_types, element_tags, element_nodes):
                vtk_type = _gmsh_to_pyvista_type.get(type_)
                
                if vtk_type:
                    n = gmsh.model.mesh.getElementProperties(type_)[3]
                    mapped_nodes = tag_to_idx[np.reshape(nodes, (-1, n))]
                    cell_groups = element_groups[tags]

                    if type_ in {1, 15}:
                        keep = cell_groups >= 0
                        mapped_nodes = mapped_nodes[keep]
                        cell_groups = cell_groups[keep]

                    if not len(mapped_nodes):
                        continue

                    cells_dict.setdefault(vtk_type, []).append(mapped_nodes)
                    cell_end = cell_offset + len(mapped_nodes)
                    np.maximum(cell_groups, 0, out=cell_array[cell_offset:cell_end])
                    cell_offset = cell_end

            cells = {k: np.vstack(v) for k, v in cells_dict.items()}
            mesh = pv.UnstructuredGrid(cells, np.reshape(coordinates, (-1, 3)))
            mesh.cell_data["CellGroup"] = cell_array[:cell_offset]
            mesh.user_dict["CellGroup"] = group_ids.copy()

            # Map point groups to point data array
            default_id = group_ids.get(self.default_group, 0)
            tag_to_gid = {v: group_ids[k] for k, v in groups.items()}
            point_array = np.full(len(node_tags), default_id, dtype=int)

            for tag in groups.values():
                nodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(0, tag)

                if len(nodes) > 0:
                    point_array[tag_to_idx[nodes]] = tag_to_gid[tag]

            for group, tag in curve_physical_tags.items():
                nodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, tag)

                if len(nodes) > 0:
                    point_array[tag_to_idx[nodes]] = group_ids[group]

            for group, tag in surface_physical_tags.items():
                nodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, tag)

                if len(nodes) > 0:
                    point_array[tag_to_idx[nodes]] = group_ids[group]

            for point_tag, group in endpoint_groups.items():
                nodes, _, _ = gmsh.model.mesh.getNodes(0, point_tag)

                if len(nodes) > 0:
                    point_array[tag_to_idx[nodes]] = group_ids[group]

            # Explicitly grouped point constraints take precedence over curves
            for point_tag, group in point_groups.items():
                if group != self.default_group:
                    nodes, _, _ = gmsh.model.mesh.getNodes(0, point_tag)

                    if len(nodes) > 0:
                        point_array[tag_to_idx[nodes]] = group_ids[group]

            mesh.point_data["PointGroup"] = point_array
            mesh.user_dict["PointGroup"] = group_ids
            mesh.set_active_scalars("CellGroup", preference="cell")

        finally:
            gmsh.clear()
            gmsh.finalize()

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))

    @staticmethod
    def _add_point(
        engine: _GmshGeometryEngine,
        point: Sequence[float] | NDArray,
        point_tags: dict[tuple[float, float], int],
        sizes: dict[tuple[float, float], float],
        default_size: float,
    ) -> int:
        """Add a 2D point to a Gmsh geometry engine."""
        key = (float(point[0]), float(point[1]))

        if key not in point_tags:
            point_tags[key] = engine.addPoint(
                key[0], key[1], 0.0, sizes.get(key, default_size), -1
            )

        return point_tags[key]

    @staticmethod
    def _add_curve(
        engine: _GmshGeometryEngine,
        pointa: Sequence[float] | NDArray,
        pointb: Sequence[float] | NDArray,
        curve_tags: dict[tuple[tuple[float, float], tuple[float, float]], int],
        point_tags: dict[tuple[float, float], int],
        sizes: dict[tuple[float, float], float],
        default_size: float,
    ) -> int:
        """Add an oriented Gmsh line."""
        key_a = (float(pointa[0]), float(pointa[1]))
        key_b = (float(pointb[0]), float(pointb[1]))
        key = (key_a, key_b) if key_a < key_b else (key_b, key_a)

        if key not in curve_tags:
            curve_tags[key] = engine.addLine(
                DelaunayMesh2D._add_point(
                    engine, key[0], point_tags, sizes, default_size
                ),
                DelaunayMesh2D._add_point(
                    engine, key[1], point_tags, sizes, default_size
                ),
            )

        return curve_tags[key] if key_a == key[0] else -curve_tags[key]

    @staticmethod
    def _add_plane_surface(
        engine: _GmshGeometryEngine,
        polygon: Polygon,
        curve_tags: dict[tuple[tuple[float, float], tuple[float, float]], int],
        point_tags: dict[tuple[float, float], int],
        sizes: dict[tuple[float, float], float],
        default_size: float,
        boundary_curves: set[int],
    ) -> int:
        """Add a Gmsh plane surface from exterior and interior polygon loops."""
        loops = []

        for ring in (polygon.exterior, *polygon.interiors):
            points = get_coordinates(ring)
            tags = [
                DelaunayMesh2D._add_curve(
                    engine, pointa, pointb, curve_tags, point_tags, sizes, default_size
                )
                for pointa, pointb in zip(points[:-1], points[1:])
            ]
            boundary_curves.update(abs(tag) for tag in tags)
            loops.append(engine.addCurveLoop(tags))

        return engine.addPlaneSurface(loops)


_gmsh_to_pyvista_type = {
    1: pv.CellType.LINE,
    2: pv.CellType.TRIANGLE,
    3: pv.CellType.QUAD,
    15: pv.CellType.VERTEX,
}
