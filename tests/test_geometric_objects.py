import numpy as np
import pytest
import pyvista as pv

import pvgridder as pvg


def test_annular_sector():
    """Test annular sector geometric object."""
    mesh = pvg.AnnularSector(0.42, 8.0, -42.0, 42.0, 8, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 304.0763618052006)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 46.523708495598974)


def test_annulus():
    """Test annulus geometric object."""
    mesh = pvg.Annulus(0.42, 8.0, 8, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 37.88999953866005)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 180.52040205003146)


def test_circle():
    """Test circle geometric object."""
    mesh = pvg.Circle(0.42, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 0.41999998688697804)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 0.498934534707864)


def test_curved_line():
    """Test curved line geometric object."""
    mesh = pvg.CurvedLine([1.0, 2.0, 3.0], 42.8, [0.0, 0.0, -1.0], [0.5, 0.5, 0.0], 16)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points[0], [1.0, 2.0, 3.0])
    assert np.allclose(mesh.points.sum(), 54.660583)
    assert np.allclose(mesh.cell_data["Length"].sum(), 42.8)


def test_cylindrical_shell():
    """Test cylindrical shell geometric object."""
    mesh = pvg.CylindricalShell(0.42, 8.0, 8.0, 8, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 75.7799990773201)
    assert np.allclose(np.abs(mesh.cell_data["Volume"]).sum(), 1444.1632164002517)


def test_cylindrical_shell_sector():
    """Test cylindrical shell sector geometric object."""
    mesh = pvg.CylindricalShellSector(0.42, 8.0, -42.0, 42.0, 8.0, 8, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 608.1527236104012)
    assert np.allclose(np.abs(mesh.cell_data["Volume"]).sum(), 372.1896679647918)


@pytest.mark.parametrize(
    "shell, holes, engine, ref_area",
    [
        (pv.Polygon(radius=8.0, n_sides=42), None, "gmsh", 200.3128038269316),
        (pv.Polygon(radius=8.0, n_sides=42), None, "occ", 200.3128038269316),
        (
            pv.Polygon(radius=8.0, n_sides=42),
            [pv.Polygon(radius=4.0, n_sides=42)],
            "gmsh",
            150.2346028701987,
        ),
        (
            pv.Polygon(radius=8.0, n_sides=42),
            [pv.Polygon(radius=4.0, n_sides=42)],
            "occ",
            150.2346028701987,
        ),
        (
            pv.Polygon(radius=8.0, n_sides=42),
            [
                pv.Polygon(radius=4.0, n_sides=21, center=(-5.0, 0.0, 0.0)),
                pv.Polygon(radius=4.0, n_sides=21, center=(5.0, 0.0, 0.0)),
            ],
            "occ",
            110.45831838912522,
        ),
    ],
)
def test_polygon(shell, holes, engine, ref_area):
    """Test polygon geometric object."""
    for celltype in ("polygon", "triangle", "quad"):
        if celltype == "polygon" and holes:
            continue

        mesh = pvg.Polygon(
            shell.points[:, :2], holes, celltype, optimization="Netgen", engine=engine
        )
        mesh = mesh.compute_cell_sizes()
        assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), ref_area)
        assert getattr(pv.CellType, celltype.upper()) in mesh.celltypes


def test_quadrilateral():
    """Test quadrilateral geometric object."""
    points = [
        [0.0, 0.0, 0.0],
        [42.0, 0.0, 0.0],
        [42.0, 88.0, 0.0],
        [0.0, 88.0, 0.0],
    ]
    mesh = pvg.Quadrilateral(points, 4, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 2925.0)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 42.0 * 88.0)


def test_rectangle():
    """Test rectangle geometric object."""
    mesh = pvg.Rectangle(42.0, 88.0, 4, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 2925.0)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 42.0 * 88.0)


def test_regular_line():
    """Test regular line geometric object."""
    mesh = pvg.RegularLine(pv.Polygon(radius=8.0, n_sides=42).points, 21)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 3.700252)
    assert np.allclose(np.abs(mesh.cell_data["Length"]).sum(), 48.80356979568666)


def test_sector():
    """Test sector geometric object."""
    mesh = pvg.Sector(0.42, -42.0, 42.0)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 0.6242416501045227)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 0.08771683144610964)


def test_sector_rectangle():
    """Test sector rectangle geometric object."""
    mesh = pvg.SectorRectangle(0.42, 4.0, 8.0, 8, 4)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 372.07949244976044)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 31.862344196032552)


def test_structured_surface():
    """Test structured surface geometric object."""
    x = np.linspace(0.0, 8.0, 16)
    line_a = np.column_stack((x, np.cos(x), np.zeros_like(x)))
    line_b = pv.MultipleLines(np.column_stack((x, np.sin(x) + 2.0, np.zeros_like(x))))
    mesh = pvg.StructuredSurface(line_a, line_b)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 164.82945439034813)
    assert np.allclose(np.abs(mesh.cell_data["Area"]).sum(), 16.152423046293965)


def test_volume():
    """Test volume geometric object."""
    x = np.linspace(0.0, 42.0, 16)
    y = np.linspace(0.0, 21.0, 32)
    surface_a = pv.RectilinearGrid(x, y, [0.0]).cast_to_structured_grid()
    surface_b = surface_a.translate((0.0, 0.0, 8.0))

    mesh = pvg.Volume(surface_a, surface_b, 8)
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 163584.0)
    assert np.allclose(np.abs(mesh.cell_data["Volume"]).sum(), 7056.0)

    mesh = pvg.Volume(
        surface_a.cast_to_unstructured_grid(), surface_b.cast_to_unstructured_grid(), 8
    )
    mesh = mesh.compute_cell_sizes()
    assert np.allclose(mesh.points.sum(), 163584.0)
    assert np.allclose(np.abs(mesh.cell_data["Volume"]).sum(), 7056.0)
