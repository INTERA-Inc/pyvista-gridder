from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike
from shapely import Polygon, contains_xy


def interactive_lasso_selection(
    mesh: pv.DataSet,
    plotter: Optional[pv.Plotter] = None,
    scalars: Optional[str | ArrayLike] = None,
    view: Literal["xy", "xz", "yz"] = "xy",
    preference: Literal["cell", "point"] = "cell",
    return_polygon: bool = False,
    **kwargs,
) -> ArrayLike | tuple[ArrayLike, Polygon]:
    """
    Select cell(s) or point(s) interactively using a lasso selection.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    plotter : pyvista.Plotter, optional
        PyVista plotter.
    scalars : str | ArrayLike, optional
        Scalars used to “color” the mesh.
    view : {'xy', 'xz', 'yz}, optional
        Isometric view.
    preference : {'cell', 'point'}, default 'cell'
        Picking mode.
    return_polygon : bool, default False
        If True, return the polygon used for selection.
    **kwargs : dict, optional
        Additional keyword arguments if *plotter* is None. See ``pyvista.Plotter`` for more details.

    Returns
    -------
    ArrayLike
        Indice(s) of selected cell(s) or point(s).
    Polygon
        Polygon used for selection if *return_polygon* is True.

    """
    p = plotter if plotter is not None else pv.Plotter(**kwargs)
    points = pv.PolyData()
    polygon = pv.PolyData()

    def callback(point: tuple[float, float, float]) -> None:
        points.points = (
            np.vstack((points.points, [point]))
            if points.n_points > 0
            else np.array([point])
        )

        if points.n_points == 1:
            polygon.points = points.points
            p.add_mesh(
                points,
                render_points_as_spheres=True,
                point_size=10,
                color="red",
                style="points_gaussian",
            )
            p.add_mesh(
                polygon,
                line_width=3,
                color="green",
                style="wireframe",
            )

        else:
            polygon_ = pv.MultipleLines(points.points)
            polygon.points = points.points
            polygon.lines = polygon_.lines

        p.update()

    p.add_mesh(mesh, scalars=scalars, show_edges=True)
    p.track_click_position(callback, side="right", double=False)

    negative = view.startswith("-")
    view = view[1:] if negative else view

    try:
        getattr(p, f"view_{view}")(negative=negative)

    except AttributeError:
        raise ValueError(f"invalid view '{view}'")

    p.enable_parallel_projection()
    p.enable_2d_style()
    p.add_axes()
    p.show()

    # Select cells or points within the lasso polygon
    if "x" not in view:
        points = np.delete(points.points, 0, axis=1)

    elif "y" not in view:
        points = np.delete(points.points, 1, axis=1)

    else:
        points = np.delete(points.points, 2, axis=1)

    polygon = Polygon(points)

    if preference == "cell":
        centers = mesh.cell_centers().points
        mask = contains_xy(polygon, centers)

    else:
        mask = contains_xy(polygon, mesh.points)

    ind = np.flatnonzero(mask)

    if return_polygon:
        return ind, polygon

    else:
        return ind


def interactive_selection(
    mesh: pv.DataSet,
    plotter: Optional[pv.Plotter] = None,
    view: Optional[str] = None,
    parallel_projection: bool = False,
    preference: Literal["cell", "point"] = "cell",
    picker: Optional[Literal["cell", "hardware", "point"]] = None,
    tolerance: float = 0.0,
    **kwargs,
) -> ArrayLike:
    """
    Select cell(s) or point(s) interactively.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    plotter : pyvista.Plotter, optional
        PyVista plotter.
    view : str, optional
        Isometric view.
    parallel_projection : bool, default False
        If True, enable parallel projection.
    preference : {'cell', 'point'}, default 'cell'
        Picking mode.
    picker : {'cell', 'hardware', 'point'}, optional
        Picker type.
    tolerance : float, default 0.0
        Picking tolerance. Ignored if *picker* is 'hardware'.
    **kwargs : dict, optional
        Additional keyword arguments. See ``pyvista.Plotter.add_mesh()`` for more details.

    Returns
    -------
    ArrayLike
        Indice(s) of selected cell(s) or point(s).

    """
    kwargs_ = {
        "scalar_bar_args": {
            "vertical": True,
            "position_y": 0.1,
            "height": 0.8,
        },
    }
    kwargs_.update(kwargs)
    kwargs_["show_edges"] = True
    kwargs_["pickable"] = True

    p = plotter if plotter is not None else pv.Plotter()
    p.theme.allow_empty_mesh = True  # Hide warning
    actors = {}

    if picker is None:
        picker = "cell" if preference == "cell" else "point"

    def callback(mesh: pv.DataSet) -> None:
        id_ = (
            mesh.cell_data["vtkOriginalCellIds"][0]
            if preference == "cell"
            else mesh.point_data["vtkOriginalPointIds"][0]
        )

        if id_ not in actors:
            actors[id_] = p.add_mesh(mesh, style="wireframe", color="red", line_width=3)
            p.update()

        else:
            actor = actors.pop(id_)
            p.remove_actor(actor, reset_camera=False, render=True)

    p.add_mesh(mesh, **kwargs_)
    p.enable_element_picking(
        mode=preference,
        callback=callback,
        show_message=False,
        picker=picker,
        tolerance=tolerance,
    )

    if view:
        negative = view.startswith("-")
        view = view[1:] if negative else view

        try:
            getattr(p, f"view_{view}")(negative=negative)

        except AttributeError:
            raise ValueError(f"invalid view '{view}'")

    if parallel_projection:
        p.enable_parallel_projection()

    p.show()

    return np.array(list(actors))
