from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing_extensions import Self


class MeshItem:
    """
    Mesh item.

    Parameters
    ----------
    mesh : pyvista.PolyData | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Input mesh.

    """

    __name__: str = "MeshItem"
    __qualname__: str = "pvgridder.MeshItem"

    def __init__(
        self, mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid, **kwargs
    ) -> None:
        """Initialize a new mesh item."""
        self._mesh = mesh

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Return mesh."""
        return self._mesh


class MeshBase(ABC):
    """
    Base mesh class.

    Parameters
    ----------
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.
    items : Sequence[MeshItem], optional
        Initial list of mesh items.

    """

    __name__: str = "MeshBase"
    __qualname__: str = "pvgridder.MeshBase"

    def __init__(
        self,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
        items: Optional[Sequence[MeshItem]] = None,
    ) -> None:
        """Initialize a new mesh."""
        self._default_group = default_group if default_group else "default"
        self._ignore_groups = list(ignore_groups) if ignore_groups else []
        self._items = items if items else []

    def _check_point_array(
        self, points: ArrayLike, axis: Optional[int] = None
    ) -> ArrayLike:
        """Check the validity of a point array."""
        points = np.asarray(points)
        axis = axis if axis is not None else self.axis if hasattr(self, "axis") else 2

        if points.ndim == 1:
            points = np.insert(points, axis, 0.0) if points.size == 2 else points

            if points.shape != (3,):
                raise ValueError(
                    f"invalid 1D point array (expected shape (2,) or (3,), got {points.shape})"
                )

        elif points.ndim == 2:
            points = (
                np.insert(points, axis, np.zeros(len(points)), axis=1)
                if points.shape[1] == 2
                else points
            )

            if points.shape[1] != 3:
                raise ValueError(
                    f"invalid 2D point array (expected size 2 or 3 along axis 1, got {points.shape[1]})"
                )

        else:
            raise ValueError(
                f"invalid point array (expected 1D or 2D array, got {points.ndim}D array)"
            )

        return points

    def _initialize_group_array(
        self,
        mesh_or_size: pv.StructuredGrid | pv.UnstructuredGrid | int,
        groups: dict,
        group: Optional[str | dict[str, Callable]] = None,
        default_group: Optional[str] = None,
    ) -> ArrayLike:
        """Initialize group array."""
        if isinstance(mesh_or_size, pv.DataSet):
            mesh = mesh_or_size
            size = mesh.n_cells

        else:
            mesh = None
            size = mesh_or_size

        arr = np.full(size, -1, dtype=int)

        if (
            mesh is not None
            and "CellGroup" in mesh.cell_data
            and "CellGroup" in mesh.user_dict
        ):
            for k, v in mesh.user_dict["CellGroup"].items():
                if k in self.ignore_groups:
                    continue

                arr[mesh.cell_data["CellGroup"] == v] = self._get_group_number(
                    k, groups
                )

        if mesh is not None and group:
            if isinstance(group, str):
                group = {group: np.ones(size, dtype=bool)}

            for k, v in group.items():
                if hasattr(v, "__call__"):
                    mask = v(mesh)

                elif isinstance(v, str):
                    mask = (
                        mesh.cell_data["CellGroup"] != groups[v[1:]]
                        if v.startswith("~")
                        else mesh.cell_data["CellGroup"] == groups[v]
                    )

                elif isinstance(v, (list, tuple, np.ndarray)) and all(
                    isinstance(x, str) for x in v
                ):
                    mask = np.zeros(size, dtype=bool)

                    for cid in v:
                        mask |= (
                            mesh.cell_data["CellGroup"] != groups[cid[1:]]
                            if cid.startswith("~")
                            else mesh.cell_data["CellGroup"] == groups[cid]
                        )

                else:
                    mask = np.asanyarray(v)

                    if mask.dtype.kind.startswith("i"):
                        mask_ = np.zeros(size, dtype=bool)
                        mask_[mask] = True
                        mask = mask_

                    elif not mask.dtype.kind.startswith("b"):
                        raise ValueError("invalid mask array")

                if k.startswith("~"):
                    k = k[1:]
                    mask = ~mask

                arr[mask] = self._get_group_number(k, groups)

        if (arr == -1).any():
            default_group = default_group if default_group else self.default_group
            arr[arr == -1] = self._get_group_number(default_group, groups)

        return arr

    @staticmethod
    def _clean(mesh: pv.DataSet, tolerance: Optional[float] = None) -> pv.DataSet:
        """Clean generated mesh."""
        from .. import remap_categorical_data

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        if "vtkGhostType" in mesh.cell_data:
            if (mesh.cell_data["vtkGhostType"] == 0).all():
                mesh.cell_data.pop("vtkGhostType", None)

        # Remove unused cell groups
        if "CellGroup" in mesh.cell_data and "CellGroup" in mesh.user_dict:
            values = list(mesh.user_dict["CellGroup"].values())
            mask = np.isin(values, mesh.cell_data["CellGroup"])

            if not mask.all():
                keys = [
                    k for k, mask_ in zip(mesh.user_dict["CellGroup"], mask) if mask_
                ]
                mapping = {k: v for v, k in enumerate(keys)}
                remap_categorical_data(
                    mesh,
                    key="CellGroup",
                    mapping=mapping,
                    preference="cell",
                    inplace=True,
                )

        return mesh

    @staticmethod
    def _get_group_number(group: str, groups: dict) -> int:
        """Get group number."""
        return groups.setdefault(group, len(groups))

    @abstractmethod
    def generate_mesh(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Generate mesh."""
        pass

    @property
    def default_group(self) -> str:
        """Return default group name."""
        return self._default_group

    @property
    def ignore_groups(self) -> list[str]:
        """Return list of groups to ignore."""
        return self._ignore_groups

    @property
    def items(self) -> list[MeshItem]:
        """Return list of mesh items."""
        return self._items


class MeshStackBase(MeshBase):
    """
    Base mesh stack class.

    Parameters
    ----------
    mesh : pyvista.ImageData | pyvista.PolyData | pyvista.RectilinearGrid | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Base mesh.
    axis : int, default 2
        Stacking axis.
    bottom_up : bool, default True
        If True, assume items are stacked from bottom to top.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshStackBase"
    __qualname__: str = "pvgridder.MeshStackBase"

    def __init__(
        self,
        mesh: pv.ImageData
        | pv.PolyData
        | pv.RectilinearGrid
        | pv.StructuredGrid
        | pv.UnstructuredGrid,
        axis: int = 2,
        bottom_up: bool = True,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new mesh stack."""
        if axis not in {0, 1, 2}:
            raise ValueError(f"invalid axis {axis} (expected {{0, 1, 2}}, got {axis})")

        if isinstance(mesh, (pv.ImageData, pv.RectilinearGrid)):
            mesh = mesh.cast_to_structured_grid()

        if isinstance(mesh, pv.StructuredGrid) and mesh.dimensions[axis] != 1:
            raise ValueError(
                f"invalid mesh or axis, dimension along axis {axis} should be 1 (got {mesh.dimensions[axis]})"
            )

        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis
        self._bottom_up = bottom_up
        self._transition_flag = False

    def add(
        self,
        arg: float | ArrayLike | Callable | pv.DataSet,
        resolution: Optional[int | ArrayLike] = None,
        method: Optional[Literal["constant", "log", "log_r"]] = None,
        priority: int = 0,
        thickness: float = 0.0,
        extrapolation: Optional[Literal["nearest"]] = None,
        group: Optional[str | dict] = None,
    ) -> Self:
        """
        Add a new item to stack.

        Parameters
        ----------
        arg : scalar | Callable | pyvista.DataSet
            New item to add to stack:

             - if scalar, all points of the previous items are translated by *abs(arg)*
               along the stacking axis in the direction given by *bottom_up*. If it's
               the first item of the stack, set the coordinates of the points of the
               base mesh to *arg* along stacking axis.

             - if Callable, must be in the form ``f(x, y, z) -> xyz`` where ``x``,
               ``y``, ``z`` are the coordinates of the points of the base mesh, and
               ``xyz`` is an array of the output coordinates along the stacking axis.

             - if :class:`pyvista.DataSet`, the coordinates of the points along the
               stacking axis are obtained by linear interpolation of the coordinates of
               the points in the dataset.

        resolution : int | ArrayLike, optional
            Number of subdivisions along the stacking axis or relative position of
            subdivisions (in percentage) with respect to the previous item. Ignored if
            first item of stack or transition item.
        method : {'constant', 'log', 'log_r'}, optional
            Subdivision method if *resolution* is an integer:

             - if 'constant', subdivisions are equally spaced.
             - if 'log', subdivisions are logarithmically spaced (from small to large).
             - if 'log_r', subdivisions are logarithmically spaced (from large to small).

            Ignored if first item of stack or transition item.

        priority : int, default 0
            Priority of item. If two consecutive items have the same priority, the last
            one takes priority. Ignored if first item of stack or transition item.
        thickness : scalar, default 0.0
            Minimum thickness of item. Ignored if first item of stack or transition
            item.
        extrapolation : {'nearest'}, optional
            Extrapolation method for points outside of the convex hull.
        group : str | dict, optional
            Group name or group mapping as a dictionary where key is the group name and:

             - if value is a string or a sequence of strings, group or list of groups
               in the base mesh to replace by the group. The selection is inverted if
               the string starts with a tilde (~).

             - if value is a Callable, must be in the form ``f(mesh) -> ind_or_mask``
               where ``mesh`` is the base mesh, and ``ind_or_mask`` are the indices of
               cells or a boolean array of the same size.

             - if value is an ArrayLike, indices of cells or mask array to assign to the
               group.

            Ignored if first item of stack.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        if isinstance(arg, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh = self._interpolate(arg.points, extrapolation)

        elif hasattr(arg, "__call__"):
            mesh = self.mesh.copy()
            mesh.points[:, self.axis] = arg(*mesh.points.T)

        else:
            if np.ndim(arg) == 0:
                if not self.items:
                    mesh = self.mesh.copy()
                    mesh.points[:, self.axis] = arg

                else:
                    arg = abs(arg)
                    arg *= 1.0 if self.bottom_up else -1.0
                    mesh = self.items[-1].mesh.copy()
                    mesh.points[:, self.axis] += arg

            else:
                arg = np.asarray(arg)

                if arg.ndim == 2:
                    if arg.shape[1] != 3:
                        raise ValueError("invalid 2D array")

                    mesh = self._interpolate(arg, extrapolation)

                else:
                    raise ValueError(f"could not add {arg.ndim}D array to stack")

        item = (
            MeshItem(
                mesh,
                resolution=resolution,
                method=method,
                priority=priority,
                thickness=thickness,
                group=group,
                transition=self._transition_flag,
            )
            if self.items
            else MeshItem(mesh, priority=priority)
        )
        self.items.append(item)
        self._transition_flag = False

        return self

    def generate_mesh(
        self,
        tolerance: float = 1.0e-8,
    ) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """
        Generate mesh by stacking all items.

        Parameters
        ----------
        tolerance : scalar, default 1.0e-8
            Set merging tolerance of duplicate points (for unstructured grids).

        Returns
        -------
        pyvista.StructuredGrid | pyvista.UnstructuredGrid
            Stacked mesh.

        """
        from .. import merge

        if len(self.items) <= 1:
            raise ValueError("not enough items to stack")

        groups = {}

        # Cut intersecting meshes w.r.t. priority
        for item1, item2 in zip(self.items[:-1], self.items[1:]):
            if item2.transition:
                continue

            shift = (
                item2.mesh.points[:, self.axis]
                - item1.mesh.points[:, self.axis]
                - item2.thickness
            )

            if not self.bottom_up:
                shift *= -1.0

            if item2.priority < item1.priority:
                item2.mesh.points[:, self.axis] = np.where(
                    shift < 0.0,
                    item2.mesh.points[:, self.axis] - shift,
                    item2.mesh.points[:, self.axis],
                )

            else:
                item1.mesh.points[:, self.axis] = np.where(
                    shift < 0.0,
                    item1.mesh.points[:, self.axis] + shift,
                    item1.mesh.points[:, self.axis],
                )

        # Generate submeshes
        meshes = []
        n_layers = 0

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1.mesh.copy()
            mesh_a.cell_data["CellGroup"] = self._initialize_group_array(
                mesh_a, groups, item2.group
            )

            if item2.transition:
                mesh_b = self._transition(mesh_a, item2.mesh, groups, item2.group)
                nsub, repeats = 1, mesh_b.n_cells
                mesh_b.cell_data["ColumnId"] = np.full(mesh_b.n_cells, -1)

            else:
                mesh_b = self._extrude(
                    mesh_a, item2.mesh, item2.resolution, item2.method
                )
                nsub, repeats = mesh_b.n_cells // mesh_a.n_cells, mesh_a.n_cells
                mesh_b.cell_data["ColumnId"] = np.tile(
                    np.arange(mesh_a.n_cells), nsub
                ).copy()

            mesh_b.cell_data["LayerId"] = np.repeat(
                np.arange(nsub) + n_layers, repeats
            ).copy()
            mesh_b.cell_data["StackItem"] = np.full(mesh_b.n_cells, i)
            meshes.append(mesh_b)
            n_layers += nsub

        # Merge submeshes
        mesh = merge(meshes, axis=self.axis, merge_points=False)
        mesh.user_dict["CellGroup"] = groups
        _ = mesh.set_active_scalars("CellGroup", preference="cell")

        return self._clean(mesh, tolerance)

    @abstractmethod
    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Extrude a line or surface mesh."""
        pass

    @abstractmethod
    def _transition(self, *args, **kwargs) -> pv.UnstructuredGrid:
        """Generate a transition mesh."""
        pass

    def _interpolate(
        self,
        points: ArrayLike,
        extrapolation: Optional[Literal["nearest"]] = None,
    ) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Interpolate new point coordinates."""
        mesh = self.mesh.copy()
        idx = [
            i for i in range(3) if i != self.axis and np.unique(points[:, i]).size > 1
        ]

        if len(idx) > 1:
            interp = LinearNDInterpolator(points[:, idx], points[:, self.axis])
            tmp = interp(mesh.points[:, idx])
            mask = np.isnan(tmp)

            if mask.any():
                if not extrapolation:
                    raise ValueError(
                        "could not interpolate from points not fully enclosing base mesh"
                    )

                elif extrapolation == "nearest":
                    interp = NearestNDInterpolator(points[:, idx], points[:, self.axis])
                    tmp[mask] = interp(mesh.points[mask][:, idx])

                else:
                    raise ValueError(f"invalid extrapolation method '{extrapolation}'")

        else:
            idx = idx[0]
            x = mesh.points[:, idx]
            xp = points[:, idx]

            if (
                not (xp[0] <= x[0] <= xp[-1] and xp[0] <= x[-1] <= xp[-1])
                and not extrapolation
            ):
                raise ValueError(
                    "could not interpolate from points not fully enclosing base mesh"
                )

            tmp = np.interp(x, xp, points[:, self.axis])

        mesh.points[:, self.axis] = tmp

        return mesh

    @property
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Return base mesh."""
        return self._mesh

    @property
    def axis(self) -> int:
        """Return stacking axis."""
        return self._axis

    @property
    def bottom_up(self) -> bool:
        """Return whether the stacking is from bottom to top."""
        return self._bottom_up
