"""Golden ray-coordinate generator for test_trace_ray_coords_vs_reference.

This script reproduces the reference arrays used in Phase 1 tests by tracing
through equivalent systems using the Optiland reference oracle.  Run it
whenever the test golden values need to be regenerated (e.g., after changing
the test system prescription).

Usage
-----
From the repository root with Optiland available on the Python path::

    python golden/generate_ray_coords.py

The script prints copy-pasteable Python arrays that go into
``tests/test_kernels.py::TestTrace::test_trace_ray_coords_vs_reference``.

Pinned Optiland commit
----------------------
Generated from Optiland @ git HEAD (optiland/optiland.git), cloned
2026-06-02.  The commit hash of the clone used:

    git -C /tmp/optiland rev-parse HEAD

Re-run the clone if the hash no longer matches the upstream HEAD before
regenerating.

System description
------------------
Plano-convex singlet (two surfaces):
  Surface 1: StandardGeometry  R = +50 mm,   k = 0,  z = 0 mm
  Surface 2: Plane             R = inf,       k = 0,  z = 5 mm
  n_glass = 1.5  (surfaces 1-back to 2-front)

Object: collimated rays parallel to z-axis (on-axis field),
starting at z = -200 mm, pupil positions y ∈ {-0.9, -0.6, -0.3, 0, +0.3, +0.6, +0.9} mm.

Measurement plane: z = 101.6667 mm  (near paraxial focus of the singlet).
"""

from __future__ import annotations

import sys

import numpy as np

# --- make Optiland importable -------------------------------------------------
# Adjust this path if the clone is elsewhere on your machine.
OPTILAND_PATH = "/tmp/optiland"
sys.path.insert(0, OPTILAND_PATH)

try:
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries import Plane, StandardGeometry
    from optiland.rays import RealRays
except ImportError as exc:
    clone_cmd = (
        f"git clone --depth=1 https://github.com/optiland/optiland.git {OPTILAND_PATH}"
    )
    raise SystemExit(
        f"Could not import Optiland from {OPTILAND_PATH!r}.\n"
        f"Clone it with:\n    {clone_cmd}\n"
        f"Original error: {exc}"
    ) from exc


def trace_plano_convex(
    pupil_ys: np.ndarray, image_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Trace collimated on-axis rays through the plano-convex singlet.

    Parameters
    ----------
    pupil_ys : np.ndarray
        1-D array of y positions (mm) at the entrance pupil.
    image_z : float
        z-position (mm) of the measurement plane.

    Returns
    -------
    x_img, y_img : np.ndarray
        Hit positions at z = image_z.
    """
    n = len(pupil_ys)
    rays = RealRays(
        np.zeros(n),
        pupil_ys.copy(),
        np.full(n, -200.0),
        np.zeros(n),
        np.zeros(n),
        np.ones(n),
        np.ones(n),
        np.full(n, 0.55),
    )

    # --- Surface 1: R = 50, k = 0, at z = 0 ---
    cs1 = CoordinateSystem(0, 0, 0)
    g1 = StandardGeometry(cs1, radius=50.0, conic=0.0)
    cs1.localize(rays)
    t = g1.distance(rays)
    rays.x += t * rays.L
    rays.y += t * rays.M
    rays.z += t * rays.N
    nx, ny, nz = g1.surface_normal(rays)
    rays.refract(nx, ny, nz, n1=1.0, n2=1.5)
    cs1.globalize(rays)

    # --- Surface 2: plane at z = 5 ---
    cs2 = CoordinateSystem(0, 0, 5)
    g2 = Plane(cs2)
    cs2.localize(rays)
    t = g2.distance(rays)
    rays.x += t * rays.L
    rays.y += t * rays.M
    rays.z += t * rays.N
    nx, ny, nz = g2.surface_normal(rays)
    rays.refract(nx, ny, nz, n1=1.5, n2=1.0)
    cs2.globalize(rays)

    # --- Free-space propagation to image plane ---
    z_np = np.asarray(rays.z)
    N_np = np.asarray(rays.N)
    M_np = np.asarray(rays.M)
    L_np = np.asarray(rays.L)
    x_np = np.asarray(rays.x)
    y_np = np.asarray(rays.y)

    t_prop = (image_z - z_np) / N_np
    x_img = x_np + t_prop * L_np
    y_img = y_np + t_prop * M_np
    return x_img, y_img


def trace_4f_chief(obj_y: float, image_z: float = 310.0) -> float:
    """Trace a chief ray through the biconvex 4f system.

    System: R = 100 mm, n = 1.5, d = 10 mm; two lenses at z = 0-10 and
    z = 200-210.

    Parameters
    ----------
    obj_y : float
        Object height (mm); ray aimed at lens-1 centre.
    image_z : float
        Measurement z-plane (mm).

    Returns
    -------
    float
        Ray y-position at image_z.
    """
    M_init = -obj_y / 100.0
    N_init = np.sqrt(1 - M_init**2)
    rays = RealRays(
        np.array([0.0]),
        np.array([obj_y]),
        np.array([-100.0]),
        np.array([0.0]),
        np.array([M_init]),
        np.array([N_init]),
        np.array([1.0]),
        np.array([0.55]),
    )
    surfaces = [
        (0.0, 100.0, 1.0, 1.5),
        (10.0, -100.0, 1.5, 1.0),
        (200.0, 100.0, 1.0, 1.5),
        (210.0, -100.0, 1.5, 1.0),
    ]
    for z_s, radius, n1, n2 in surfaces:
        cs = CoordinateSystem(0, 0, z_s)
        g = StandardGeometry(cs, radius=radius, conic=0.0)
        cs.localize(rays)
        t = g.distance(rays)
        rays.x += t * rays.L
        rays.y += t * rays.M
        rays.z += t * rays.N
        nx, ny, nz = g.surface_normal(rays)
        rays.refract(nx, ny, nz, n1=n1, n2=n2)
        cs.globalize(rays)

    z_np = float(np.asarray(rays.z)[0])
    y_np = float(np.asarray(rays.y)[0])
    M_np = float(np.asarray(rays.M)[0])
    N_np = float(np.asarray(rays.N)[0])
    dt = (image_z - z_np) / N_np
    return y_np + dt * M_np


if __name__ == "__main__":
    # --- Plano-convex singlet golden ---
    pupil_ys = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    image_z = 101.6667
    x_golden, y_golden = trace_plano_convex(pupil_ys, image_z)

    print("# --- Plano-convex singlet (copy-array golden) ---")
    print(f"# System: R=50, flat back, n=1.5, d=5mm; image_z={image_z}")
    print(f"# Pupil: {list(pupil_ys)}")
    print(f"x_golden = {x_golden!r}")
    print(f"y_golden = {y_golden!r}")
    print()

    # --- 4f chief-ray golden ---
    y_4f = trace_4f_chief(obj_y=1.0, image_z=310.0)
    print("# --- 4f biconvex chief ray (copy-value golden) ---")
    print("# System: R=100, n=1.5, d=10mm; obj_y=1.0; image_z=310")
    print(f"y_4f_golden = {y_4f:.16f}")
