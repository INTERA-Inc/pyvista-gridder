from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union, cast

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from pyrequire import require_package
from scipy.spatial import Voronoi

from ._base import MeshBase, MeshItem
from ._helpers import generate_surface_from_two_lines, resolution_to_perc


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Optional

    from numpy.typing import ArrayLike
    from typing_extensions import Self


@require_package("shapely>=2.0")
class VoronoiMesh2D(MeshBase):
    """
    2D Voronoi mesh class.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Background mesh.
    axis : int, default 2
        Background mesh axis to discard.
    preference : {'cell', 'point'}, default 'cell'
        Determine which data to use for background mesh.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "VoronoiMesh2D"
    __qualname__: str = "pvgridder.VoronoiMesh2D"

    def __init__(
        self,
        mesh: pv.DataSet,
        axis: int = 2,
        preference: Literal["cell", "point"] = "cell",
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a 2D Voronoi mesh."""
        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self._preference = preference
        self._fuse_cells = []
        self.mesh.points[:, self.axis] = 0.0  # type: ignore

    def add(
        self,
        mesh_or_points: pv.DataSet | ArrayLike,
        priority: int = 0,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points to Voronoi diagram.

        Parameters
        ----------
        mesh_or_points : pyvista.DataSet | ArrayLike
            Dataset or coordinates of points.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        if not isinstance(mesh_or_points, pv.DataSet):
            mesh_or_points = self._check_point_array(mesh_or_points)
            mesh = pv.PolyData(mesh_or_points)

        else:
            mesh = mesh_or_points.copy()
            mesh = cast(
                Union[pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid], mesh
            )

        mesh.points[:, self.axis] = 0.0  # type: ignore
        item = MeshItem(mesh, group=group, priority=priority)
        self.items.append(item)

        return self

    def add_circle(
        self,
        radius: float,
        constraint_radius: Optional[float] = None,
        resolution: Optional[int | ArrayLike] = None,
        center: Optional[ArrayLike] = None,
        plain: bool = False,
        priority: int = 0,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points from a circle to Voronoi diagram.

        Parameters
        ----------
        radius : scalar
            Circle radius.
        constraint_radius : scalar, optional
            Constraint circle radius. If None, default to 1.5 times *radius*.
        resolution : int | ArrayLike, optional
            Number of subdivisions along the azimuthal axis or relative position of
            subdivisions (in percentage) with respect to the starting angle (0 degree).
        center : ArrayLike, optional
            Center of the circle.
        plain : bool, default False
            If True, fuse all cells within the circle into a single cell.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import Annulus, Circle, MeshMerge

        constraint_radius = (
            constraint_radius if constraint_radius is not None else 1.5 * radius
        )
        dr = constraint_radius - radius

        if dr > 0.0:
            self.add(
                (
                    MeshMerge()
                    .add(Circle(radius - dr, resolution, center=center))
                    .add(Annulus(radius - dr, radius, 1, resolution, center=center))
                    .generate_mesh()
                ),
                priority=priority,
                group=group,
            )
            self.add(
                Annulus(radius, radius + dr, 1, resolution, center=center),
                priority=0,
            )

        elif dr == 0.0:
            self.add(
                Circle(radius, resolution, center=center),
                priority=priority,
                group=group,
            )

        else:
            raise ValueError("invalid constraint radius")

        if plain:
            center = np.zeros(3) if center is None else np.asanyarray(center)
            center = np.insert(center, self.axis, 0.0) if len(center) == 2 else center
            self.fuse_cells.append(
                lambda x: np.linalg.norm(x - center, axis=1) < radius
            )

        return self

    def add_polyline(
        self,
        mesh_or_points: ArrayLike | pv.PolyData,
        width: float,
        preference: Literal["cell", "point"] = "cell",
        padding: Optional[float] = None,
        constraint: int | tuple[int, int] = 1,
        resolution: Optional[int | ArrayLike] = None,
        priority: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points from a polyline to Voronoi diagram.

        Parameters
        ----------
        mesh_or_points : ArrayLike | pyvista.PolyData
            Dataset or coordinates of points.
        width : scalar
            Width of polyline.
        preference : {'cell', 'point'}, default 'cell'
            Determine which coordinates to add:

             - if 'cell', add cell centers of polyline.
             - if 'point', add polyline point coordinates.

        padding : scalar, optional
            Distance between cell centers of first and last points (if
            *preference* = 'cell') and start and end of the polyline, respectively.
            Default is half of *width*.
        constraint : int | tuple[int, int], default 1
            Number of constraint points added at the start and the end of the polyline.
        resolution : int | ArrayLike, optional
            Number of subdivisions along the line or relative position of subdivisions
            (in percentage) with respect to the starting point.
        priority : int, default 0
            Priority of item. Points enclosed in a cell with (strictly) higher
            priority are discarded.
        group : str, optional
            Group name.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        from .. import extract_cells, split_lines

        if not isinstance(mesh_or_points, pv.PolyData):
            mesh = pv.MultipleLines(np.asanyarray(mesh_or_points))

        else:
            mesh = mesh_or_points.copy()

        if isinstance(constraint, int):
            constraint_start = constraint
            constraint_end = constraint

        else:
            constraint_start, constraint_end = constraint

        perc = resolution_to_perc(resolution)
        perc = [2.0 * perc[0] - perc[1], *perc.tolist(), 2.0 * perc[-1] - perc[-2]]

        # Loop over polylines
        for polyline in split_lines(mesh):
            # Remove axis from points
            points = np.delete(polyline.points, self.axis, axis=1)

            # Calculate new point coordinates if cell centers
            if preference == "cell":
                padding = padding if padding is not None else 0.5 * width
                points = np.vstack(
                    (
                        points[0]
                        + padding
                        * (points[0] - points[1])
                        / np.linalg.norm(points[0] - points[1]),
                        0.5 * (points[:-1] + points[1:]),
                        points[-1]
                        + padding
                        * (points[-1] - points[-2])
                        / np.linalg.norm(points[-1] - points[-2]),
                    )
                )

            # Calculate forward direction vectors
            fdvec = np.diff(points, axis=0)
            fdvec = np.vstack((fdvec, fdvec[-1]))

            # Calculate backward direction vectors
            bdvec = np.diff(points[::-1], axis=0)[::-1]
            bdvec = np.vstack((bdvec[0], bdvec))

            # Append constraint points at the start and at the end of the polyline
            for _ in range(constraint_start):
                points = np.vstack((points[0] - fdvec[0], points))
                fdvec = np.vstack((fdvec[0], fdvec))
                bdvec = np.vstack((bdvec[0], bdvec))

            for _ in range(constraint_end):
                points = np.vstack((points, points[-1] - bdvec[-1]))
                fdvec = np.vstack((fdvec, fdvec[-1]))
                bdvec = np.vstack((bdvec, bdvec[-1]))

            # Calculate normal vectors
            fnorm = np.column_stack((-fdvec[:, 1], fdvec[:, 0]))
            bnorm = np.column_stack((bdvec[:, 1], -bdvec[:, 0]))
            normals = 0.5 * (fnorm + bnorm)
            normals /= np.linalg.norm(normals, axis=1)[:, None]

            # Generate structured grid with constraint cells
            points = np.insert(points, self.axis, 0.0, axis=1)
            normals = np.insert(normals, self.axis, 0.0, axis=1)

            tvec = 0.5 * width * normals
            line_a = points - tvec
            line_b = points + tvec
            plane = "yz" if self.axis == 0 else "xz" if self.axis == 1 else "xy"
            mesh = generate_surface_from_two_lines(line_a, line_b, plane, perc)

            # Identify constraint cells
            shape = [n - 1 for n in mesh.dimensions if n != 1]
            constraint_ = np.ones(shape, dtype=bool)
            constraint_[constraint_start : shape[0] - constraint_end, 1:-1] = False
            constraint_ = constraint_.ravel(order="F")

            # Add to items
            item = MeshItem(
                extract_cells(mesh, ~constraint_),
                group=group,
                priority=priority if priority else 0,
            )
            self.items.append(item)

            item = MeshItem(extract_cells(mesh, constraint_), group=None, priority=0)
            self.items.append(item)

        return self

    def generate_mesh(
        self,
        infinity: Optional[float] = None,
        min_length: float = 1.0e-4,
        tolerance: float = 1.0e-8,
        qhull_options: Optional[str] = None,
    ) -> pv.UnstructuredGrid:
        """
        Generate 2D Voronoi mesh.

        Parameters
        ----------
        infinity : scalar, optional
            Value used for points at infinity.
        min_length : scalar, default 1.0e-4
            Set the minimum length of polygons' edges.
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points.
        qhull_options: str, optional
            Additional options to pass to Qhull performing the Voronoi tessellation.
            See <http://www.qhull.org/html/qh-optq.htm#qhull> for more details.

        Returns
        -------
        pyvista.UnstructuredGrid
            2D Voronoi mesh.

        """
        from shapely import Polygon, get_coordinates

        from .. import (
            average_points,
            decimate_rdp,
            extract_boundary_polygons,
            extract_cells,
            fuse_cells,
            get_cell_centers,
        )

        groups = {}
        items = sorted(self.items, key=lambda item: abs(item.priority))

        if self.preference == "cell":
            points = get_cell_centers(self.mesh).tolist()
            group_array = self._initialize_group_array(self.mesh, groups)
            priority_array = np.full(self.mesh.n_cells, -np.inf)

        elif self.preference == "point":
            points = self.mesh.points.tolist()
            group_array = self._initialize_group_array(self.mesh.n_points, groups)
            priority_array = np.full(self.mesh.n_points, -np.inf)

        active = np.ones(len(points), dtype=bool)

        for i, item in enumerate(items):
            mesh_a = item.mesh
            points_ = get_cell_centers(mesh_a)

            # Remove out of bound points from item mesh
            mask = self.mesh.find_containing_cell(get_cell_centers(mesh_a)) != -1
            mask = cast(NDArray, mask)

            if mask.any():
                mesh_a = extract_cells(mesh_a, mask)
                points_ = points_[mask]

            # Initialize item arrays
            item_group_array = self._initialize_group_array(mesh_a, groups, item.group)
            item_priority_array = np.full(mesh_a.n_cells, abs(item.priority))

            # Disable existing points contained by item mesh and with lower (or equal) priority
            if not isinstance(mesh_a, pv.PolyData):
                idx = mesh_a.find_containing_cell(points)
                mask = np.logical_and(
                    idx != -1,
                    (
                        priority_array <= item_priority_array[idx]
                        if item.priority >= 0
                        else priority_array < item_priority_array[idx]
                    ),
                )
                active[mask] = False
                group_array[mask] = False

            # Append points to point list
            points += points_.tolist()
            active = np.concatenate((active, np.ones(len(points_), dtype=bool)))
            group_array = np.concatenate((group_array, item_group_array))
            priority_array = np.concatenate((priority_array, item_priority_array))

        points = np.delete(points, self.axis, axis=1)
        voronoi_points = points[active]
        regions, vertices = self._generate_voronoi_tesselation(voronoi_points, infinity, qhull_options)

        # Average points within minimum distance
        if min_length > 0.0:
            poly = average_points(
                pv.PolyData().from_irregular_faces(
                    np.insert(vertices, 2, 0.0, axis=-1), regions
                ),
                tolerance=min_length,
            )
            regions = poly.irregular_faces
            vertices = poly.points
            mask = np.isin(
                np.arange(len(voronoi_points)),
                poly.cell_data["vtkOriginalCellIds"],
                assume_unique=True,
                invert=True,
            )

            if mask.any():
                idx = np.arange(len(active))[active]
                active[idx[mask]] = False
                voronoi_points = voronoi_points[~mask]

        # Generate boundary polygon
        boundary_polygons = extract_boundary_polygons(
            self.mesh, fill=False, with_holes=True
        )

        if boundary_polygons is None or len(boundary_polygons) == 0:
            raise ValueError(
                "could not extract boundary polygons for the background mesh"
            )

        boundary_polygon = boundary_polygons[0]
        boundary = [
            np.delete(decimate_rdp(polygon).points, self.axis, axis=1)
            for polygon in boundary_polygon
        ]
        boundary = Polygon(boundary[0], boundary[1:])

        # Generate polygonal mesh
        points, cells = [], []
        n_points = 0

        for i, region in enumerate(regions):
            polygon = Polygon(vertices[region])

            if not polygon.is_valid:
                raise ValueError(f"region {i} is not a valid polygon")

            polygon = boundary.intersection(polygon)
            points_ = list(get_coordinates(polygon)[:-1])
            cells += [len(points_), *(np.arange(len(points_)) + n_points)]

            points += points_
            n_points += len(points_)

        points = self._check_point_array(points)
        mesh = pv.PolyData(points, faces=cells)
        mesh = mesh.cast_to_unstructured_grid()

        mesh.cell_data["CellGroup"] = group_array[active]
        mesh.user_dict["CellGroup"] = groups
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        # Add coordinates of Voronoi points
        voronoi_points = np.insert(voronoi_points, self.axis, 0.0, axis=1)
        mesh.cell_data["X"] = voronoi_points[:, 0]
        mesh.cell_data["Y"] = voronoi_points[:, 1]
        mesh.cell_data["Z"] = voronoi_points[:, 2]

        # Fuse cells, if any
        if self.fuse_cells:
            points = get_cell_centers(mesh)
            indices = [func(points) for func in self.fuse_cells]
            mesh = fuse_cells(mesh, indices)

        return cast(pv.UnstructuredGrid, self._clean(mesh, tolerance))

    def _generate_voronoi_tesselation(
        self,
        points: ArrayLike,
        infinity: Optional[float] = None,
        qhull_options: Optional[str] = None,
    ) -> tuple[list[list[NDArray]], NDArray]:
        """
        Generate Voronoi tessalation.

        Note
        ----
        See <https://stackoverflow.com/a/43023639>.

        """
        voronoi = Voronoi(points, qhull_options=qhull_options)

        # Construct a map containing all ridges for a given point
        ridges = {}

        for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        center = voronoi.points.mean(axis=0)
        radius = infinity if infinity else np.ptp(self.mesh.points).max() * 1.0e3
        new_vertices = voronoi.vertices.tolist()
        new_regions = []

        for p1, region in enumerate(voronoi.point_region):
            vertices = voronoi.regions[region]

            if -1 not in vertices:
                new_regions.append(vertices)

            else:
                ridge = ridges[p1]
                new_region = [v for v in vertices if v >= 0]

                for p2, v1, v2 in ridge:
                    if v2 < 0:
                        v1, v2 = v2, v1

                    if v1 >= 0:
                        continue

                    t = voronoi.points[p2] - voronoi.points[p1]
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])

                    midpoint = voronoi.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = voronoi.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                # Sort region counterclockwise
                vs = np.array([new_vertices[v] for v in new_region])
                c = vs.mean(axis=0)
                angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                new_regions.append([new_region[i] for i in np.argsort(angles)])

        return new_regions, np.array(new_vertices)

    @property
    def mesh(self) -> pv.DataSet:
        """Get background mesh."""
        return self._mesh

    @property
    def axis(self) -> int:
        """Get discarded axis."""
        return self._axis

    @property
    def preference(self) -> Literal["cell", "point"]:
        """Get preference."""
        return cast(Literal["cell", "point"], self._preference)

    @property
    def fuse_cells(self) -> list[Callable]:
        """Get list of cells to fuse."""
        return self._fuse_cells
