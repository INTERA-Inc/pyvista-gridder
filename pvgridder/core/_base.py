from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from scipy.interpolate import LinearNDInterpolator
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
        mesh: pv.StructuredGrid | pv.UnstructuredGrid,
        groups: dict,
        group: Optional[str | dict[str, Callable]] = None,
        default_group: Optional[str] = None,
    ) -> ArrayLike:
        """Initialize group array."""
        arr = np.full(mesh.n_cells, -1, dtype=int)

        if "CellGroup" in mesh.cell_data and "CellGroup" in mesh.user_dict:
            for k, v in mesh.user_dict["CellGroup"].items():
                if k in self.ignore_groups:
                    continue

                arr[mesh.cell_data["CellGroup"] == v] = self._get_group_number(k, groups)

        if group:
            if isinstance(group, str):
                group = {group: lambda x: np.ones(x.n_cells, dtype=bool)}

            for k, v in group.items():
                arr[v(mesh)] = self._get_group_number(k, groups)

        if (arr == -1).any():
            default_group = default_group if default_group else self.default_group
            arr[arr == -1] = self._get_group_number(default_group, groups)

        return arr

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
    mesh : pyvista.PolyData | pyvista.StructuredGrid | pyvista.UnstructuredGrid
        Base mesh.
    axis : int, default 2
        Stacking axis.
    default_group : str, optional
        Default group name.
    ignore_groups : Sequence[str], optional
        List of groups to ignore.

    """

    __name__: str = "MeshStackBase"
    __qualname__: str = "pvgridder.MeshStackBase"

    def __init__(
        self,
        mesh: pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid,
        axis: int = 2,
        default_group: Optional[str] = None,
        ignore_groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize a new mesh stack."""
        if axis not in {0, 1, 2}:
            raise ValueError(f"invalid axis {axis} (expected {{0, 1, 2}}, got {axis})")

        if isinstance(mesh, pv.StructuredGrid) and mesh.dimensions[axis] != 1:
            raise ValueError(
                f"invalid mesh or axis, dimension along axis {axis} should be 1 (got {mesh.dimensions[axis]})"
            )

        super().__init__(default_group, ignore_groups)
        self._mesh = mesh.copy()
        self._axis = axis

    def add(
        self,
        arg: float | ArrayLike | Callable | pv.DataSet,
        resolution: Optional[int | ArrayLike] = None,
        method: Optional[Literal["constant", "log", "log_r"]] = None,
        priority: int = 0,
        group: Optional[str] = None,
    ) -> Self:
        """
        Add a new item to stack.

        Parameters
        ----------
        arg : scalar | Callable | pyvista.DataSet
            New item to add to stack:

             - if scalar, all points of the previous items are translated by *arg* along
               the stacking axis. If it's the first item of the stack, set the
               coordinates of the points of the base mesh to *arg* along stacking axis.

             - if Callable, must be in the form ``f(x, y, z) -> xyz`` where ``x``,
               ``y``, ``z`` are the coordinates of the points of the base mesh, and
               ``xyz`` is an array of the output coordinates along the stacking axis.

             - if :class:`pyvista.DataSet`, the coordinates of the points along the
               stacking axis are obtained by linear interpolation of the coordinates of
               the points in the dataset.

        resolution : int | ArrayLike, optional
            Number of subdivisions along the stacking axis or relative position of
            subdivisions (in percentage) with respect to the previous item. Ignored if
            first item of stack.
        method : {'constant', 'log', 'log_r'}, optional
            Subdivision method if *resolution* is an integer:

             - if 'constant', subdivisions are equally spaced.
             - if 'log', subdivisions are logarithmically spaced (from small to large).
             - if 'log_r', subdivisions are logarithmically spaced (from large to small).

            Ignored if first item of stack.

        priority : int, default 0
            Priority of item. If two consecutive items have the same priority, the last
            one takes priority. Ignored if first item of stack.
        group : str, optional
            Group name. Ignored if first item of stack.

        Returns
        -------
        Self
            Self (for daisy chaining).

        """
        if isinstance(arg, (pv.PolyData, pv.StructuredGrid, pv.UnstructuredGrid)):
            mesh = self._interpolate(arg.points)

        elif hasattr(arg, "__call__"):
            mesh = self.mesh.copy()
            mesh.points[:, self.axis] = arg(*mesh.points.T)

        else:
            if np.ndim(arg) == 0:
                if not self.items:
                    mesh = self.mesh.copy()
                    mesh.points[:, self.axis] = arg

                else:
                    mesh = self.items[-1].mesh.copy()
                    mesh.points[:, self.axis] += arg

            else:
                arg = np.asarray(arg)

                if arg.ndim == 2:
                    if arg.shape[1] != 3:
                        raise ValueError("invalid 2D array")

                    mesh = self._interpolate(arg)

                else:
                    raise ValueError(f"could not add {arg.ndim}D array to stack")

        item = (
            MeshItem(
                mesh,
                resolution=resolution,
                method=method,
                priority=priority,
                group=group,
            )
            if self.items
            else MeshItem(mesh, priority=priority)
        )
        self.items.append(item)

        return self

    def generate_mesh(
        self, tolerance: float = 1.0e-8
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
            shift = item2.mesh.points[:, self.axis] - item1.mesh.points[:, self.axis]

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

        for i, (item1, item2) in enumerate(zip(self.items[:-1], self.items[1:])):
            mesh_a = item1.mesh.copy()
            mesh_a.cell_data["CellGroup"] = self._initialize_group_array(
                mesh_a, groups, item2.group
            )
            mesh_b = self._extrude(mesh_a, item2.mesh, item2.resolution, item2.method)

            if i > 0:
                mesh = merge(mesh, mesh_b, self.axis)

            else:
                mesh = mesh_b

        mesh.user_dict["CellGroup"] = groups

        if isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.clean(tolerance=tolerance, produce_merge_map=False)

        # Flag zero area/volume cells as inactive
        self._set_active(mesh)

        return mesh

    @abstractmethod
    def _extrude(self, *args, **kwargs) -> pv.StructuredGrid | pv.UnstructuredGrid:
        """Extrude a line or surface mesh."""
        pass

    @abstractmethod
    def _set_active(self, *args) -> None:
        """Set active cell data."""
        pass

    def _interpolate(
        self, points: ArrayLike
    ) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Interpolate new point coordinates."""
        mesh = self.mesh.copy()
        idx = [
            i for i in range(3) if i != self.axis and np.unique(points[:, i]).size > 1
        ]

        if len(idx) > 1:
            interp = LinearNDInterpolator(points[:, idx], points[:, self.axis])
            tmp = interp(mesh.points[:, idx])

            if np.isnan(tmp).any():
                raise ValueError(
                    "could not interpolate from points not fully enclosing base mesh"
                )

            mesh.points[:, self.axis] = tmp

        else:
            idx = idx[0]
            x = mesh.points[:, idx]
            xp = points[:, idx]

            if not (xp[0] <= x[0] <= xp[-1] and xp[0] <= x[-1] <= xp[-1]):
                raise ValueError(
                    "could not interpolate from points not fully enclosing base mesh"
                )

            mesh.points[:, self.axis] = np.interp(x, xp, points[:, self.axis])

        return mesh

    @property
    def mesh(self) -> pv.PolyData | pv.StructuredGrid | pv.UnstructuredGrid:
        """Return base mesh."""
        return self._mesh

    @property
    def axis(self) -> int:
        """Return stacking axis."""
        return self._axis
