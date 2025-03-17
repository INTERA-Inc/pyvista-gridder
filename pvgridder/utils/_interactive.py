from __future__ import annotations
from typing import Literal, Optional
from numpy.typing import ArrayLike

import numpy as np
import pyvista as pv


def interactive_selection(
    mesh: pv.DataSet,
    scalars: Optional[str | ArrayLike] = None,
    parallel_projection: bool = False,
    preference: Literal["cell", "point"] = "cell",
    tolerance: float = 0.0,
    message: Optional[str] = None,
    **kwargs,
) -> ArrayLike:
    """
    Select cell(s) or point(s) interactively.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Input mesh.
    scalars : str | ArrayLike
        Scalars used to “color” the mesh.
    parallel_projection : bool, default False
        If True, enable parallel projection.
    preference : {'cell', 'point'}, default 'cell'
        Picking mode.
    tolerance : float, default 0.0
        Picking tolerance.
    message : str, optional
        Text to display.
    
    Returns
    -------
    ArrayLike
        Indice(s) of selected cell(s) or point(s).

    """
    p = pv.Plotter(**kwargs)
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

    if parallel_projection:
        p.enable_parallel_projection()

    if message:
        p.add_text(message, "upper_left")

    p.add_axes()
    p.show()

    return np.array(list(actors))
