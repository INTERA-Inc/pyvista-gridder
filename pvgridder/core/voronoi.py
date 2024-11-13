from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Optional
from typing_extensions import Self

import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi

from ._base import MeshBase, MeshItem
from ._helpers import generate_plane_surface_from_two_lines
from .._common import require_package


@require_package("shapely", (2, 0))
class VoronoiMesh2D(MeshBase):
    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self.mesh.points[:, self.axis] = 0.0

    def add(
        self,
        mesh_or_points: ArrayLike | pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        zorder: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        if not isinstance(mesh_or_points, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh_or_points = self._check_point_array(mesh_or_points)
            mesh = pv.PolyData(mesh_or_points)

        else:
            mesh = mesh_or_points.copy()
        
        item = MeshItem(mesh, group=group, zorder=zorder, type_="mesh")
        self.items.append(item)

        return self

    def add_polyline(
        self,
        mesh_or_points: ArrayLike | pv.PolyData,
        width: float,
        zorder: Optional[int] = None,
        group: Optional[str] = None,
    ) -> Self:
        from .. import split_lines

        if not isinstance(mesh_or_points, pv.PolyData):
            mesh = pv.MultipleLines(mesh_or_points)

        else:
            mesh = mesh_or_points.copy()

        mesh.points[:, self.axis] = 0.0
        polylines = split_lines(mesh)

        # Loop over polylines
        for polyline in polylines:
            # Remove axis from points
            points = np.delete(polyline.points, self.axis, axis=1)

            # Calculate forward direction vectors
            fdvec = np.diff(points, axis=0)
            fdvec = np.row_stack((fdvec, fdvec[-1]))
            
            # Calculate backward direction vectors
            bdvec = np.diff(points[::-1], axis=0)[::-1]
            bdvec = np.row_stack((bdvec[0], bdvec))

            # Append constraint points at the beginning and at the end of the polyline
            points = np.row_stack((points[0] - fdvec[0], points, points[-1] - bdvec[-1]))

            # Calculate normal vectors
            fnorm = np.column_stack((-fdvec[:, 1], fdvec[:, 0]))
            bnorm = np.column_stack((bdvec[:, 1], -bdvec[:, 0]))
            normals = 0.5 * (fnorm + bnorm)
            normals = np.row_stack((normals[0], normals, normals[-1]))
            normals /= np.linalg.norm(normals, axis=1)[:, None]

            # Generate structured grid with constraint cells
            points = np.insert(points, self.axis, 0.0, axis=1)
            normals = np.insert(normals, self.axis, 0.0, axis=1)

            line_a = points - 1.5 * width * normals
            line_b = points + 1.5 * width * normals
            mesh = generate_plane_surface_from_two_lines(line_a, line_b, 3, axis=self.axis)

            # Identify constraint cells
            shape = [n - 1 for n in mesh.dimensions if n != 1]
            nx = shape[0]

            constraint = np.ones(mesh.n_cells, dtype=bool)
            constraint[nx + 1 : -(nx + 1)] = False
            mesh.cell_data["constraint"] = constraint

            # Add to items
            item = MeshItem(mesh, group=group, zorder=zorder, type_="line")
            self.items.append(item)

        return self

    def generate_mesh(
        self,
        infinity: Optional[float] = None,
        tolerance: float = 1.0e-4,
    ) -> pv.UnstructuredGrid:
        from shapely import Polygon
        from .. import extract_boundary_polygons

        points = self.mesh.cell_centers().points.tolist()
        active = np.ones(len(points), dtype=bool)

        groups = {}
        group_array = self._initialize_group_array(self.mesh, groups)
        zorder_array = np.zeros(self.mesh.n_cells)

        for i, item in enumerate(self.items):
            mesh_a = item.mesh
            zorder = item.zorder if item.zorder else 0
            group = item.group if item.group else self.default_group
            points_ = mesh_a.cell_centers().points

            if item.type_ == "line":
                item_group_array = self._initialize_group_array(mesh_a, groups)
                item_group_array = np.where(
                    mesh_a.cell_data["constraint"],
                    self._get_group_number(self.default_group, groups),
                    self._get_group_number(group, groups),
                )
                item_zorder_array = np.where(mesh_a.cell_data["constraint"], 0, zorder)

            else:
                item_group_array = self._initialize_group_array(mesh_a, groups, item.group)
                item_zorder_array = np.full(mesh_a.n_cells, zorder)

            # Remove out of bound points from item mesh
            mask = self.mesh.find_containing_cell(points_) != -1
            points_ = points_[mask]

            # Disable points from point list contained by item mesh and with lower priority
            idx = mesh_a.find_containing_cell(points)
            mask = np.logical_and(
                idx != -1,
                zorder_array <= item_zorder_array[idx],
            )
            active[mask] = False
            group_array[mask] = False

            # Append points to point list
            points += points_.tolist()
            active = np.concatenate((active, np.ones(len(points_), dtype=bool)))
            group_array = np.concatenate((group_array, item_group_array))
            zorder_array = np.concatenate((zorder_array, item_zorder_array))

        points = np.delete(points, self.axis, axis=1)
        voronoi_points = points[active]
        regions, vertices = self._generate_voronoi_tesselation(voronoi_points, infinity)

        # Generate boundary polygon
        boundary = extract_boundary_polygons(self.mesh)
        boundary = Polygon(np.delete(boundary[0].points, self.axis, axis=1))

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
            points_ = points_[np.insert(mask, 0, True)].tolist()
            cells += [len(points_), *(np.arange(len(points_)) + n_points).tolist()]

            points += points_
            n_points += len(points_)

        points = self._check_point_array(points)
        mesh = pv.PolyData(points, faces=cells)
        mesh = mesh.cast_to_unstructured_grid()

        mesh.cell_data["group"] = group_array[active]
        mesh.user_dict["group"] = groups

        return mesh.clean(tolerance=tolerance, produce_merge_map=False)

    def _generate_voronoi_tesselation(
        self,
        points: ArrayLike,
        infinity: Optional[float] = None,
    ) -> tuple[list[ArrayLike], ArrayLike, ArrayLike]:
        """
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
        return self._mesh

    @property
    def axis(self) -> int:
        return self._axis
