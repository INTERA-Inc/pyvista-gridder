from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from scipy.spatial import Voronoi
from typing_extensions import Self

from ._base import MeshBase, MeshItem
from ._helpers import generate_surface_from_two_lines, resolution_to_perc
from .._common import require_package


@require_package("shapely", (2, 0))
class VoronoiMesh2D(MeshBase):
    """
    2D Voronoi mesh class.

    Parameters
    ----------
    mesh : :class:`pyvista.PolyData` | :class:`pyvista.StructuredGrid` | :class:`pyvista.UnstructuredGrid`
        Background mesh.
    axis : int, default 2
        Background mesh axis to discard.
    default_group : str, optional
        Default group name.
    ignore_groups : sequence of str, optional
        List of groups to ignore.

    """

    __name__: str = "VoronoiMesh2D"
    __qualname__: str = "pvgridder.VoronoiMesh2D"

    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        """Initialize a 2D Voronoi mesh."""
        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self.mesh.points[:, self.axis] = 0.0

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
        mesh_or_points : :class:`pyvista.DataSet` | ArrayLike
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

        item = MeshItem(mesh, group=group, priority=priority)
        self.items.append(item)

        return self

    def add_polyline(
        self,
        mesh_or_points: ArrayLike | pv.PolyData,
        width: float,
        preference: Optional[Literal["cell", "point"]] = "cell",
        padding: Optional[float] = None,
        constrain_start: bool = True,
        constrain_end: bool = True,
        resolution: Optional[int | ArrayLike] = None,
        priority: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add points from a polyline to Voronoi diagram.

        Parameters
        ----------
        mesh_or_points : ArrayLike | :class:`pyvista.PolyData`
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
        constrain_start : bool, default True
            If True, add a constraint point at the start of the polyline.
        constrain_end : bool, default True
            If True, add a constraint point at the end of the polyline.
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
        from .. import split_lines

        if not isinstance(mesh_or_points, pv.PolyData):
            mesh = pv.MultipleLines(mesh_or_points)

        else:
            mesh = mesh_or_points.copy()

        perc = resolution_to_perc(resolution)
        perc = [2.0 * perc[0] - perc[1], *perc.tolist(), 2.0 * perc[-1] - perc[-2]]

        # Loop over polylines
        for polyline in split_lines(mesh):
            # Remove axis from points
            points = np.delete(polyline.points, self.axis, axis=1)

            # Calculate new point coordinates if cell centers
            if preference == "cell":
                padding = padding if padding is not None else 0.5 * width
                points = np.row_stack(
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
            fdvec = np.row_stack((fdvec, fdvec[-1]))

            # Calculate backward direction vectors
            bdvec = np.diff(points[::-1], axis=0)[::-1]
            bdvec = np.row_stack((bdvec[0], bdvec))

            # Append constraint points at the start and at the end of the polyline
            if constrain_start:
                points = np.row_stack((points[0] - fdvec[0], points))
                fdvec = np.row_stack((fdvec[0], fdvec))
                bdvec = np.row_stack((bdvec[0], bdvec))

            if constrain_end:
                points = np.row_stack((points, points[-1] - bdvec[-1]))
                fdvec = np.row_stack((fdvec, fdvec[-1]))
                bdvec = np.row_stack((bdvec, bdvec[-1]))

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
            constraint = np.ones(shape, dtype=bool)
            constraint[int(constrain_start) : shape[0] - int(constrain_end), 1:-1] = (
                False
            )
            constraint = constraint.ravel(order="F")

            # Add to items
            item = MeshItem(
                mesh.extract_cells(~constraint),
                group=group,
                priority=priority if priority else 0,
            )
            self.items.append(item)

            item = MeshItem(mesh.extract_cells(constraint), group=None, priority=0)
            self.items.append(item)

        return self

    def generate_mesh(
        self,
        infinity: Optional[float] = None,
        tolerance: float = 1.0e-4,
    ) -> pv.UnstructuredGrid:
        """
        Generate 2D Voronoi mesh.

        Parameters
        ----------
        infinity : scalar, optional
            Value used for points at infinity.
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points.

        Returns
        -------
        :class:`pyvista.UnstructuredGrid`
            2D Voronoi mesh.

        """
        from shapely import Polygon

        from .. import decimate_rdp, extract_boundary_polygons

        points = self.mesh.cell_centers().points.tolist()
        active = np.ones(len(points), dtype=bool)

        groups = {}
        group_array = self._initialize_group_array(self.mesh, groups)
        priority_array = np.full(self.mesh.n_cells, -np.inf)
        items = sorted(self.items, key=lambda item: abs(item.priority))

        for i, item in enumerate(items):
            mesh_a = item.mesh
            group = item.group if item.group else self.default_group
            points_ = mesh_a.cell_centers().points
            item_group_array = self._initialize_group_array(mesh_a, groups, item.group)
            item_priority_array = np.full(mesh_a.n_cells, abs(item.priority))

            # Remove out of bound points from item mesh
            mask = self.mesh.find_containing_cell(points_) != -1
            points_ = points_[mask]

            # Disable existing points contained by item mesh and with lower (or equal) priority
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
        regions, vertices = self._generate_voronoi_tesselation(voronoi_points, infinity)

        # Generate boundary polygon
        boundary = extract_boundary_polygons(self.mesh)[0]
        boundary = decimate_rdp(boundary)
        boundary = Polygon(np.delete(boundary.points, self.axis, axis=1))

        # Generate polygonal mesh
        points, cells = [], []
        n_points = 0

        for i, region in enumerate(regions):
            polygon = Polygon(vertices[region])

            if not polygon.is_valid:
                raise ValueError(f"region {i} is not a valid polygon")

            polygon = boundary.intersection(polygon)
            points_ = np.array(polygon.exterior.coords)
            mask = np.linalg.norm(np.diff(points_, axis=0), axis=1) > tolerance
            points_ = points_[:-1][mask].tolist()
            cells += [len(points_), *(np.arange(len(points_)) + n_points)]

            points += points_
            n_points += len(points_)

        points = self._check_point_array(points)
        mesh = pv.PolyData(points, faces=cells)
        mesh = mesh.cast_to_unstructured_grid()

        mesh.cell_data["Group"] = group_array[active]
        mesh.user_dict["Group"] = groups
        mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        # Add coordinates of Voronoi points
        voronoi_points = np.insert(voronoi_points, self.axis, 0.0, axis=1)
        mesh.cell_data["X"] = voronoi_points[:, 0]
        mesh.cell_data["Y"] = voronoi_points[:, 1]
        mesh.cell_data["Z"] = voronoi_points[:, 2]

        return mesh

    def _generate_voronoi_tesselation(
        self,
        points: ArrayLike,
        infinity: Optional[float] = None,
    ) -> tuple[list[ArrayLike], ArrayLike, ArrayLike]:
        """
        Generate Voronoi tessalation.

        Note
        ----
        See <https://stackoverflow.com/a/43023639>.

        """
        voronoi = Voronoi(points, qhull_options="Qbb Qc")

        # Construct a map containing all ridges for a given point
        ridges = {}

        for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        center = voronoi.points.mean(axis=0)
        radius = infinity if infinity else self.mesh.points.ptp().max() * 1.0e3
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
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Return background mesh."""
        return self._mesh

    @property
    def axis(self) -> int:
        """Return discarded axis."""
        return self._axis
