from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pyvista as pv
from numpy.typing import ArrayLike


def interactive_selection(
    mesh: pv.DataSet,
    plotter: Optional[pv.Plotter] = None,
    scalars: Optional[str | ArrayLike] = None,
    view: Optional[str] = None,
    parallel_projection: bool = False,
    preference: Literal["cell", "point"] = "cell",
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
    scalars : str | ArrayLike, optional
        Scalars used to “color” the mesh.
    view : str, optional
        Isometric view.
    parallel_projection : bool, default False
        If True, enable parallel projection.
    preference : {'cell', 'point'}, default 'cell'
        Picking mode.
    tolerance : float, default 0.0
        Picking tolerance.
    **kwargs : dict, optional
        Additional keyword arguments if *plotter* is None. See ``pyvista.Plotter`` for more details.

    Returns
    -------
    ArrayLike
        Indice(s) of selected cell(s) or point(s).

    """
    p = plotter if plotter is not None else pv.Plotter(**kwargs)
    actors = {}

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

    p.add_mesh(mesh, scalars=scalars, show_edges=True)
    p.enable_element_picking(
        mode=preference,
        callback=callback,
        show_message=False,
        picker=preference,
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
