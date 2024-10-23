from __future__ import annotations
from numpy.typing import ArrayLike
from typing import Literal, Optional

import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi

from ._base import MeshBase, MeshItem


class VoronoiMesh2D(MeshBase):
    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[list[str]] = None,
    ) -> None:
        try:
            import shapely
        
        except ModuleNotFoundError:
            raise ModuleNotFoundError("VoronoiMesh2D requires shapely")

        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self.mesh.points[:, self.axis] = 0.0

    def add(
        self,
        mesh_or_points: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid | ArrayLike,
        preference: Literal["point", "cell"] = "cell",
        group: Optional[str] = None,
    ) -> None:
        if not isinstance(mesh_or_points, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh_or_points = np.asarray(mesh_or_points)

            if mesh_or_points.shape[1] == 2:
                mesh_or_points = np.insert(
                    mesh_or_points,
                    self.axis,
                    np.zeros(len(mesh_or_points)),
                    axis=-1,
                )

            if mesh_or_points.shape[2] != 3:
                raise ValueError(f"could not add points with shape {mesh_or_points.shape}")

            mesh = pv.PolyData(mesh_or_points)

        else:
            mesh = mesh_or_points
        
        item = MeshItem(mesh, preference=preference, group=group)
        self.items.append(item)

    def generate_mesh(
        self,
        infinity: Optional[float] = None,
        tolerance: float = 1.0e-4,
    ) -> pv.UnstructuredGrid:
        from shapely import Polygon
        from .. import extract_boundary_polygons

        points = self.mesh.cell_centers().points.tolist()
        active = np.ones(len(points), dtype=bool)

        for i, item in enumerate(self.items):
            mesh = item.mesh

            if item.preference == "cell":
                points_ = mesh.cell_centers().points
                idx = mesh.find_containing_cell(points)
                active[idx != -1] = False

            else:
                points_ = mesh.points

            idx = self.mesh.find_containing_cell(points_)
            points_ = points_[idx != -1]

            points += points_.tolist()
            active = np.concatenate((active, np.ones(len(points_), dtype=bool)))

        idx = [i for i in range(3) if i != self.axis]
        points = np.array(points)[:, idx]
        voronoi_points = points[active]
        regions, vertices = self._generate_voronoi_tesselation(voronoi_points, infinity)

        # Generate boundary polygon
        boundary = extract_boundary_polygons(self.mesh)
        boundary = Polygon(boundary[0].points[:, idx])

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

        points = np.array(points)
        points = np.insert(
            points,
            self.axis,
            np.zeros(len(points)),
            axis=-1,
        )
        mesh = pv.PolyData(points, faces=cells)
        mesh = mesh.cast_to_unstructured_grid()

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
