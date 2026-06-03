"""Phase 1 tests — Core kernels, Rays, and trace (TEST_PLAN §1, §11).

Golden values are sourced as follows (see TEST_PLAN §0.1 modes):

* **copy-value**: literal pasted from an Optiland test, provenance noted.
* **copy-math**: reference computed inline (physics invariant).
* **copy-array**: regenerated from the Optiland clone via golden/generate_ray_coords.py.
* **model-pattern**: structure from Optiland tests, numbers from our own system.

Cross-backend parity tests (Tier B) are authored fresh.
"""

from __future__ import annotations

import math

import numpy as np

from optisketch.backends import NumpyBackend
from optisketch.kernels import (
    _intersect,
    _normal,
    _reflect,
    _refract,
    _trace_surfaces,
)
from optisketch.rays import Rays, _Params, _Structure

# ---------------------------------------------------------------------------
# Assertion helpers (TEST_PLAN tolerance tiers)
# ---------------------------------------------------------------------------


def _ac(a, b, rtol, atol, be):
    a_np = be.to_numpy(a) if be is not None else np.asarray(a)
    b_np = be.to_numpy(b) if be is not None else np.asarray(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def assert_tier_a(a, b, be=None):
    _ac(a, b, 1e-5, 1e-7, be)  # reference/golden


def assert_tier_b(a, b, be=None):
    _ac(a, b, 1e-11, 1e-12, be)  # cross-backend parity


def assert_tier_d(a, b, be=None):
    _ac(a, b, 0.0, 1e-12, be)  # exact invariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rays(
    be,
    *,
    x=0.0,
    y=0.0,
    z=0.0,
    L=0.0,
    M=0.0,
    N=1.0,
    intensity=1.0,
    wavelength=0.55,
    opd=0.0,
    valid=True,
):
    """Construct a single-ray Rays bundle."""
    return Rays(
        x=be.array([x]),
        y=be.array([y]),
        z=be.array([z]),
        L=be.array([L]),
        M=be.array([M]),
        N=be.array([N]),
        i=be.array([intensity]),
        w=be.array([wavelength]),
        opd=be.array([opd]),
        valid=be.array([valid]),
    )


def make_rays_batch(be, *, ys, x=0.0, z=0.0, L=0.0, M=0.0, N=1.0, wavelength=0.55):
    """Construct a multi-ray Rays bundle varying y."""
    n = len(ys)
    ys_arr = np.asarray(ys, dtype=np.float64)
    return Rays(
        x=be.array(np.full(n, x)),
        y=be.array(ys_arr),
        z=be.array(np.full(n, z)),
        L=be.array(np.full(n, L)),
        M=be.array(np.full(n, M)),
        N=be.array(np.full(n, N)),
        i=be.array(np.ones(n)),
        w=be.array(np.full(n, wavelength)),
        opd=be.array(np.zeros(n)),
        valid=be.array(np.ones(n, dtype=bool)),
    )


def _singlet_setup(be):
    """Plano-convex singlet: R=50, flat back, n=1.5, d=5mm.

    Object at z=-200 (collimated, on-axis).
    """
    structure = _Structure(
        n_surfaces=2,
        is_plane=(False, True),
        reflective=(False, False),
    )
    params = _Params(
        z=be.array([0.0, 5.0]),
        radii=be.array([50.0, 1.0]),  # R unused for plane
        conics=be.array([0.0, 0.0]),
        n1=be.array([1.0, 1.5]),
        n2=be.array([1.5, 1.0]),
        semi_aperture=be.array([math.inf, math.inf]),
    )
    return structure, params


def _4f_setup(be):
    """4-surface biconvex 4f system: R=100, n=1.5, d=10mm.

    Object at z=-100, image near z=310.
    """
    structure = _Structure(
        n_surfaces=4,
        is_plane=(False, False, False, False),
        reflective=(False, False, False, False),
    )
    params = _Params(
        z=be.array([0.0, 10.0, 200.0, 210.0]),
        radii=be.array([100.0, -100.0, 100.0, -100.0]),
        conics=be.array([0.0, 0.0, 0.0, 0.0]),
        n1=be.array([1.0, 1.5, 1.0, 1.5]),
        n2=be.array([1.5, 1.0, 1.5, 1.0]),
        semi_aperture=be.array([math.inf, math.inf, math.inf, math.inf]),
    )
    return structure, params


# ---------------------------------------------------------------------------
# §1.1  _intersect
# ---------------------------------------------------------------------------

# copy-value goldens from optiland tests/test_geometries.py


def test_intersect_sphere_single(backend):
    """t=2.7888 -- TestStandardGeometry::test_distance"""
    be = backend
    x, y, z = be.array([1.0]), be.array([2.0]), be.array([-3.0])
    L, M, N = be.array([0.0]), be.array([0.0]), be.array([1.0])
    R, k = be.array(-12.0), be.array(0.5)
    t = _intersect(x, y, z, L, M, N, R, k, is_plane=False, be=be)
    assert_tier_a(t, 2.7888809636986154, be=be)


def test_intersect_sphere_tilted(backend):
    """t=10.201 -- TestStandardGeometry::test_distance"""
    be = backend
    L_val = 0.359
    M_val = -0.229
    N_val = math.sqrt(1 - L_val**2 - M_val**2)
    x, y, z = be.array([1.0]), be.array([2.0]), be.array([-10.2])
    L = be.array([L_val])
    M = be.array([M_val])
    N = be.array([N_val])
    R, k = be.array(-12.0), be.array(0.5)
    t = _intersect(x, y, z, L, M, N, R, k, is_plane=False, be=be)
    assert_tier_a(t, 10.201933401020467, be=be)


def test_intersect_plane_axial(backend):
    """Plane at z=0, ray at z=-5 going +z: t=5.0 (copy-value, analytic)."""
    be = backend
    x, y, z = be.array([0.0]), be.array([0.0]), be.array([-5.0])
    L, M, N = be.array([0.0]), be.array([0.0]), be.array([1.0])
    R, k = be.array(0.0), be.array(0.0)  # R unused for plane
    t = _intersect(x, y, z, L, M, N, R, k, is_plane=True, be=be)
    assert_tier_d(t, 5.0, be=be)


def test_intersect_batch_equals_stacked(backend):
    """Batch result matches stacked per-ray results — Tier B."""
    be = backend
    xs = np.array([1.0, 2.0])
    ys = np.array([2.0, 3.0])
    zs = np.array([-3.0, -4.0])
    R, k = be.array(-12.0), be.array(0.5)
    L = be.array(np.zeros(2))
    M = be.array(np.zeros(2))
    N = be.array(np.ones(2))
    t_batch = _intersect(
        be.array(xs), be.array(ys), be.array(zs), L, M, N, R, k, False, be
    )
    t0 = _intersect(
        be.array([xs[0]]),
        be.array([ys[0]]),
        be.array([zs[0]]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        R,
        k,
        False,
        be,
    )
    t1 = _intersect(
        be.array([xs[1]]),
        be.array([ys[1]]),
        be.array([zs[1]]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        R,
        k,
        False,
        be,
    )
    assert_tier_b(t_batch, np.array([be.to_numpy(t0)[0], be.to_numpy(t1)[0]]), be=be)


def test_intersect_parity(jax_backend):
    """Numpy vs JAX: Tier-B parity for _intersect."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    rng = np.random.default_rng(0)
    x_v = rng.uniform(-2, 2, 5)
    y_v = rng.uniform(-2, 2, 5)
    z_v = rng.uniform(-5, -1, 5)
    L_v = rng.uniform(-0.1, 0.1, 5)
    M_v = rng.uniform(-0.1, 0.1, 5)
    N_v = np.sqrt(1 - L_v**2 - M_v**2)
    R_v = -12.0
    k_v = 0.5

    t_np = _intersect(
        x_v, y_v, z_v, L_v, M_v, N_v, np.float64(R_v), np.float64(k_v), False, be_np
    )
    t_jax = _intersect(
        be_jax.array(x_v),
        be_jax.array(y_v),
        be_jax.array(z_v),
        be_jax.array(L_v),
        be_jax.array(M_v),
        be_jax.array(N_v),
        be_jax.array(R_v),
        be_jax.array(k_v),
        False,
        be_jax,
    )
    assert_tier_b(t_np, be_jax.to_numpy(t_jax))


# ---------------------------------------------------------------------------
# §1.2  _normal
# ---------------------------------------------------------------------------

# copy-value from optiland tests/test_geometries.py::TestStandardGeometry


def test_normal_conic_values(backend):
    """R=10, k=0.5, x=1, y=2 -> (0.10127, 0.20254, -0.97402)."""
    be = backend
    x, y = be.array([1.0]), be.array([2.0])
    R, k = be.array(10.0), be.array(0.5)
    nx, ny, nz = _normal(x, y, R, k, is_plane=False, be=be)
    assert_tier_a(nx, 0.10127393670836665, be=be)
    assert_tier_a(ny, 0.2025478734167333, be=be)
    assert_tier_a(nz, -0.9740215340114144, be=be)


def test_normal_unit_length(backend):
    """||n|| == 1 for arbitrary conic — copy-math, Tier D."""
    be = backend
    x = be.array([1.0, -0.5, 0.0])
    y = be.array([2.0, 1.2, 0.0])
    R, k = be.array(10.0), be.array(0.5)
    nx, ny, nz = _normal(x, y, R, k, is_plane=False, be=be)
    mag2 = be.to_numpy(nx) ** 2 + be.to_numpy(ny) ** 2 + be.to_numpy(nz) ** 2
    np.testing.assert_allclose(mag2, 1.0, atol=1e-12)


def test_normal_plane(backend):
    """Plane normal is (0, 0, 1)."""
    be = backend
    x = be.array([1.0, -1.0])
    y = be.array([2.0, 3.0])
    nx, ny, nz = _normal(x, y, None, None, is_plane=True, be=be)
    np.testing.assert_array_equal(be.to_numpy(nx), [0.0, 0.0])
    np.testing.assert_array_equal(be.to_numpy(ny), [0.0, 0.0])
    np.testing.assert_array_equal(be.to_numpy(nz), [1.0, 1.0])


def test_normal_parity(jax_backend):
    """Numpy vs JAX: Tier-B parity for _normal."""
    be_np = NumpyBackend()
    be_jax = jax_backend
    x, y = np.array([1.0, -0.5, 0.3]), np.array([2.0, 1.2, -0.8])
    R, k = 10.0, 0.5

    nx_np, ny_np, nz_np = _normal(x, y, R, k, False, be_np)
    nx_j, ny_j, nz_j = _normal(
        be_jax.array(x),
        be_jax.array(y),
        be_jax.array(R),
        be_jax.array(k),
        False,
        be_jax,
    )
    for a, b in [(nx_np, nx_j), (ny_np, ny_j), (nz_np, nz_j)]:
        assert_tier_b(a, be_jax.to_numpy(b))


# ---------------------------------------------------------------------------
# §1.3  _refract
# ---------------------------------------------------------------------------


def test_refract_flat_30deg(backend):
    """Flat surface (normal +z), 30 deg incidence, n1=1 -> n2=1.5.

    copy-value: M_out=1/3, N_out=2*sqrt(2)/3 ~= 0.94280904
    (Snell's law: sin(t2) = sin(30)/1.5 = 1/3)
    """
    be = backend
    sin30 = 0.5
    cos30 = math.sqrt(3.0) / 2.0
    L = be.array([0.0])
    M = be.array([sin30])
    N = be.array([cos30])
    nx = be.array([0.0])
    ny = be.array([0.0])
    nz = be.array([1.0])

    Lout, Mout, Nout, tir = _refract(L, M, N, nx, ny, nz, 1.0, 1.5, be)

    assert_tier_a(Lout, 0.0, be=be)
    assert_tier_a(Mout, 1.0 / 3.0, be=be)
    assert_tier_a(Nout, 2.0 * math.sqrt(2.0) / 3.0, be=be)
    assert not be.to_numpy(tir)[0], "No TIR expected here"


def test_refract_snell_residual(backend):
    """n1*sin(t1) = n2*sin(t2) for several angles — copy-math."""
    be = backend
    n1, n2 = 1.0, 1.5
    for angle_deg in [10.0, 20.0, 30.0]:
        theta1 = math.radians(angle_deg)
        M_in = math.sin(theta1)
        N_in = math.cos(theta1)
        nx = be.array([0.0])
        ny = be.array([0.0])
        nz = be.array([1.0])
        _, Mout, Nout, tir = _refract(
            be.array([0.0]), be.array([M_in]), be.array([N_in]), nx, ny, nz, n1, n2, be
        )
        assert not be.to_numpy(tir)[0]
        M_np = be.to_numpy(Mout)[0]
        N_np = be.to_numpy(Nout)[0]
        theta2 = math.asin(M_np / math.sqrt(M_np**2 + N_np**2))
        residual = abs(n1 * math.sin(theta1) - n2 * math.sin(theta2))
        assert residual < 1e-12, f"Snell residual {residual} at {angle_deg} deg"


def test_refract_unit_direction(backend):
    """||d_out|| == 1 for arbitrary incidence — copy-math."""
    be = backend
    theta1 = math.radians(25.0)
    L, M, N = 0.0, math.sin(theta1), math.cos(theta1)
    nx, ny, nz = 0.0, 0.0, 1.0
    Lout, Mout, Nout, _ = _refract(
        be.array([L]),
        be.array([M]),
        be.array([N]),
        be.array([nx]),
        be.array([ny]),
        be.array([nz]),
        1.0,
        1.5,
        be,
    )
    mag2 = (
        be.to_numpy(Lout)[0] ** 2
        + be.to_numpy(Mout)[0] ** 2
        + be.to_numpy(Nout)[0] ** 2
    )
    assert abs(mag2 - 1.0) < 1e-12


def test_refract_tir_sets_flag(backend):
    """45 deg from glass (n=1.5) into air: above critical angle -> TIR flag."""
    be = backend
    theta_inc = math.radians(45.0)
    _, _, _, tir = _refract(
        be.array([0.0]),
        be.array([math.sin(theta_inc)]),
        be.array([math.cos(theta_inc)]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        1.5,
        1.0,
        be,
    )
    assert be.to_numpy(tir)[0], "TIR expected at 45 deg glass->air"


def test_refract_no_tir_below_critical(backend):
    """30 deg from glass into air: below critical angle -> no TIR."""
    be = backend
    theta_inc = math.radians(30.0)
    _, _, _, tir = _refract(
        be.array([0.0]),
        be.array([math.sin(theta_inc)]),
        be.array([math.cos(theta_inc)]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        1.5,
        1.0,
        be,
    )
    assert not be.to_numpy(tir)[0]


# ---------------------------------------------------------------------------
# §1.4  _reflect
# ---------------------------------------------------------------------------

# copy-math tests (coplanarity, unit direction, canonical normals)


def test_reflect_canonical_z_normal(backend):
    """d=(0,0,1), n=(0,0,1) -> reflected = (0,0,-1).
    optiland tests/test_rays.py::test_reflect
    """
    be = backend
    L, M, N = be.array([0.0]), be.array([0.0]), be.array([1.0])
    nx, ny, nz = be.array([0.0]), be.array([0.0]), be.array([1.0])
    Lout, Mout, Nout = _reflect(L, M, N, nx, ny, nz, be)
    assert_tier_d(Lout, 0.0, be=be)
    assert_tier_d(Mout, 0.0, be=be)
    assert_tier_d(Nout, -1.0, be=be)


def test_reflect_canonical_x_normal(backend):
    """d=(1,0,0), n=(1,0,0) -> reflected = (-1,0,0)."""
    be = backend
    Lout, Mout, Nout = _reflect(
        be.array([1.0]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        be.array([0.0]),
        be.array([0.0]),
        be,
    )
    assert_tier_d(Lout, -1.0, be=be)
    assert_tier_d(Mout, 0.0, be=be)
    assert_tier_d(Nout, 0.0, be=be)


def test_reflect_unit_direction(backend):
    """||d_out|| == 1 — copy-math."""
    be = backend
    theta = math.radians(30.0)
    Lout, Mout, Nout = _reflect(
        be.array([0.0]),
        be.array([math.sin(theta)]),
        be.array([math.cos(theta)]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        be,
    )
    L_np = be.to_numpy(Lout)[0]
    M_np = be.to_numpy(Mout)[0]
    N_np = be.to_numpy(Nout)[0]
    assert abs(L_np**2 + M_np**2 + N_np**2 - 1.0) < 1e-12


def test_reflect_coplanar(backend):
    """Reflected ray lies in the plane of incidence - copy-math.

    (d_in x n_aligned) must be parallel to (d_out x n_aligned).
    """
    be = backend
    theta = math.radians(30.0)
    L_in, M_in, N_in = 0.0, math.sin(theta), math.cos(theta)
    nx, ny, nz = 0.0, 0.0, 1.0

    Lout, Mout, Nout = _reflect(
        be.array([L_in]),
        be.array([M_in]),
        be.array([N_in]),
        be.array([nx]),
        be.array([ny]),
        be.array([nz]),
        be,
    )
    L_out = be.to_numpy(Lout)[0]
    M_out = be.to_numpy(Mout)[0]
    N_out = be.to_numpy(Nout)[0]

    cx_in = M_in * nz - N_in * ny
    cy_in = N_in * nx - L_in * nz
    cz_in = L_in * ny - M_in * nx

    cx_out = M_out * nz - N_out * ny
    cy_out = N_out * nx - L_out * nz
    cz_out = L_out * ny - M_out * nx

    cross_x = cy_in * cz_out - cz_in * cy_out
    cross_y = cz_in * cx_out - cx_in * cz_out
    cross_z = cx_in * cy_out - cy_in * cx_out
    assert abs(cross_x) < 1e-12 and abs(cross_y) < 1e-12 and abs(cross_z) < 1e-12


# ---------------------------------------------------------------------------
# §1.5  _surface_step and _trace_surfaces
# ---------------------------------------------------------------------------


def test_trace_history_shape(backend):
    """History has shape (n_surf+1, n_rays, 3)."""
    be = backend
    structure, params = _singlet_setup(be)
    n_rays = 5
    rays = make_rays_batch(be, ys=np.linspace(-0.5, 0.5, n_rays), z=-200.0)

    _, history = _trace_surfaces(rays, structure, params, be)
    assert len(history) == structure.n_surfaces + 1
    for h in history:
        h_np = be.to_numpy(h)
        assert h_np.shape == (n_rays, 3), f"expected ({n_rays}, 3), got {h_np.shape}"


def test_trace_ray_coords_vs_reference(backend):
    """Plano-convex singlet at z=101.6667: copy-array golden from Optiland.

    Generator: golden/generate_ray_coords.py
    System: R=50 conic (k=0), flat back, n=1.5, d=5mm.
    Pupil heights: [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9] mm (collimated beam).
    """
    be = backend
    structure, params = _singlet_setup(be)
    pupil_ys = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    rays = make_rays_batch(be, ys=pupil_ys, z=-200.0)

    final, _ = _trace_surfaces(rays, structure, params, be)

    z_img = 101.6667
    z_np = be.to_numpy(final.z)
    M_np = be.to_numpy(final.M)
    N_np = be.to_numpy(final.N)
    x_np = be.to_numpy(final.x)
    y_np = be.to_numpy(final.y)
    t_prop = (z_img - z_np) / N_np
    x_img = x_np + t_prop * be.to_numpy(final.L)
    y_img = y_np + t_prop * M_np

    # copy-array golden from golden/generate_ray_coords.py
    y_golden = np.array(
        [
            8.46953573e-05,
            2.52026877e-05,
            3.22508522e-06,
            0.0,
            -3.22508522e-06,
            -2.52026877e-05,
            -8.46953573e-05,
        ]
    )
    x_golden = np.zeros(7)

    np.testing.assert_allclose(x_img, x_golden, atol=1e-7)
    np.testing.assert_allclose(y_img, y_golden, rtol=1e-5, atol=1e-8)


def test_trace_4f_imaging(backend):
    """Chief ray from y=+1 images to y~=-0.9664 in a biconvex 4f system.

    model-pattern: magnification is negative and |mag| ~= 1.
    copy-value: golden -0.9663708617629430 computed from Optiland clone.
    System: biconvex R=100, n=1.5, d=10mm; image plane at z=310.
    """
    be = backend
    structure, params = _4f_setup(be)

    M_init = -1.0 / 100.0
    N_init = math.sqrt(1 - M_init**2)
    rays = make_rays(be, y=1.0, z=-100.0, M=M_init, N=N_init)

    final, _ = _trace_surfaces(rays, structure, params, be)

    z_img = 310.0
    dt = (z_img - be.to_numpy(final.z)[0]) / be.to_numpy(final.N)[0]
    y_img = be.to_numpy(final.y)[0] + dt * be.to_numpy(final.M)[0]

    # golden value from golden/generate_ray_coords.py
    assert_tier_a(y_img, -0.9663708617629430)
    assert y_img < 0.0, "image must be inverted"


def test_trace_opd_flat_plate(backend):
    """OPD through flat glass plate (n=1.5, d=5mm) = n*d = 7.5mm.

    copy-math: ray starts at surface 1, so first t=0; second t~=5.
    """
    be = backend
    structure = _Structure(
        n_surfaces=2,
        is_plane=(True, True),
        reflective=(False, False),
    )
    params = _Params(
        z=be.array([0.0, 5.0]),
        radii=be.array([1.0, 1.0]),
        conics=be.array([0.0, 0.0]),
        n1=be.array([1.0, 1.5]),
        n2=be.array([1.5, 1.0]),
        semi_aperture=be.array([math.inf, math.inf]),
    )
    rays = make_rays(be, x=0.0, y=0.0, z=0.0, L=0.0, M=0.0, N=1.0)

    final, _ = _trace_surfaces(rays, structure, params, be)

    np.testing.assert_allclose(be.to_numpy(final.opd), [7.5], atol=1e-12)


def test_trace_spherical_aberration_scaling(backend):
    """RMS spot radius scales as aperture^3 for a single refracting surface.

    model-pattern: dominant 3rd-order spherical aberration of a plano-convex
    singlet means RMS radius over a marginal fan scales as aperture^3.
    """
    be = backend
    structure, params = _singlet_setup(be)
    z_img = 101.6667

    rms_values = []
    apertures = [0.3, 0.6, 0.9]
    for ap in apertures:
        n_rays = 7
        ys = np.linspace(-ap, ap, n_rays)
        rays = make_rays_batch(be, ys=ys, z=-200.0)
        final, _ = _trace_surfaces(rays, structure, params, be)

        z_np = be.to_numpy(final.z)
        y_np = be.to_numpy(final.y)
        M_np = be.to_numpy(final.M)
        N_np = be.to_numpy(final.N)
        t_prop = (z_img - z_np) / N_np
        y_img = y_np + t_prop * M_np

        centroid = np.mean(y_img)
        rms = math.sqrt(np.mean((y_img - centroid) ** 2))
        rms_values.append(rms)

    log_ap = np.log(apertures)
    log_rms = np.log(rms_values)
    p = np.polyfit(log_ap, log_rms, 1)[0]
    assert 2.5 < p < 3.5, f"Expected exponent ~3 for spherical aberration, got {p:.2f}"


def test_trace_parity(jax_backend):
    """Numpy vs JAX: Tier-B parity for full trace."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    pupil_ys = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])

    struct_np, params_np = _singlet_setup(be_np)
    rays_np = make_rays_batch(be_np, ys=pupil_ys, z=-200.0)
    final_np, _ = _trace_surfaces(rays_np, struct_np, params_np, be_np)

    struct_jax, params_jax = _singlet_setup(be_jax)
    rays_jax = make_rays_batch(be_jax, ys=pupil_ys, z=-200.0)
    final_jax, _ = _trace_surfaces(rays_jax, struct_jax, params_jax, be_jax)

    for field in ("x", "y", "z", "L", "M", "N", "opd"):
        a = getattr(final_np, field)
        b = be_jax.to_numpy(getattr(final_jax, field))
        assert_tier_b(a, b)


# ---------------------------------------------------------------------------
# §11  valid semantics / NaN hygiene
# ---------------------------------------------------------------------------


def test_valid_miss_sets_invalid(backend):
    """Ray outside the aperture -> valid=False."""
    be = backend
    structure = _Structure(
        n_surfaces=1,
        is_plane=(True,),
        reflective=(False,),
    )
    params = _Params(
        z=be.array(
            [
                0.0,
            ]
        ),
        radii=be.array([1.0]),
        conics=be.array([0.0]),
        n1=be.array([1.0]),
        n2=be.array([1.0]),
        semi_aperture=be.array(
            [
                3.0,
            ]
        ),
    )
    rays = make_rays(be, x=0.0, y=5.0, z=-10.0, N=1.0)
    final, _ = _trace_surfaces(rays, structure, params, be)
    assert not be.to_numpy(final.valid)[0], "ray outside aperture should be invalid"


def test_valid_tir_sets_invalid(backend):
    """TIR event -> valid=False for the affected ray only."""
    be = backend
    theta_tir = math.radians(50.0)  # > critical angle for glass->air
    M_in = math.sin(theta_tir)
    N_in = math.cos(theta_tir)

    structure = _Structure(
        n_surfaces=1,
        is_plane=(True,),
        reflective=(False,),
    )
    params = _Params(
        z=be.array([0.0]),
        radii=be.array([1.0]),
        conics=be.array([0.0]),
        n1=be.array([1.5]),  # glass -> air: TIR possible
        n2=be.array([1.0]),
        semi_aperture=be.array([math.inf]),
    )
    rays = make_rays(be, z=-5.0, M=M_in, N=N_in)
    final, _ = _trace_surfaces(rays, structure, params, be)
    assert not be.to_numpy(final.valid)[0], "TIR should mark ray invalid"


def test_valid_clip_sets_invalid(backend):
    """Rays outside semi_aperture are clipped (valid=False); others stay True."""
    be = backend
    ys = np.arange(6, dtype=np.float64)  # [0, 1, 2, 3, 4, 5]
    structure = _Structure(
        n_surfaces=1,
        is_plane=(True,),
        reflective=(False,),
    )
    params = _Params(
        z=be.array([0.0]),
        radii=be.array([1.0]),
        conics=be.array([0.0]),
        n1=be.array([1.0]),
        n2=be.array([1.0]),
        semi_aperture=be.array([3.5]),  # r > 3.5 -> invalid
    )
    rays = make_rays_batch(be, ys=ys, z=-5.0)
    final, _ = _trace_surfaces(rays, structure, params, be)
    valid_np = be.to_numpy(final.valid)
    # y=0..3 -> r <= 3.5 -> True; y=4,5 -> r > 3.5 -> False
    np.testing.assert_array_equal(valid_np, [True, True, True, True, False, False])


def test_valid_sticky(backend):
    """Once False, valid stays False through subsequent surfaces."""
    be = backend
    structure = _Structure(
        n_surfaces=2,
        is_plane=(True, True),
        reflective=(False, False),
    )
    params = _Params(
        z=be.array([0.0, 5.0]),
        radii=be.array([1.0, 1.0]),
        conics=be.array([0.0, 0.0]),
        n1=be.array([1.0, 1.0]),
        n2=be.array([1.0, 1.0]),
        semi_aperture=be.array([3.0, math.inf]),  # only first clips
    )
    rays = make_rays_batch(be, ys=[1.0, 5.0], z=-5.0)
    final, _ = _trace_surfaces(rays, structure, params, be)
    valid_np = be.to_numpy(final.valid)
    assert valid_np[0], "y=1 inside both apertures"
    assert not valid_np[1], "y=5 clipped at surface 1, must stay invalid"


def test_valid_inf_semi_disables_clip(backend):
    """semi_aperture=inf -> no clipping regardless of ray position."""
    be = backend
    structure = _Structure(
        n_surfaces=1,
        is_plane=(True,),
        reflective=(False,),
    )
    params = _Params(
        z=be.array(
            [
                0.0,
            ]
        ),
        radii=be.array([1.0]),
        conics=be.array([0.0]),
        n1=be.array([1.0]),
        n2=be.array([1.0]),
        semi_aperture=be.array(
            [
                math.inf,
            ]
        ),
    )
    rays = make_rays_batch(be, ys=[100.0, 1000.0], z=-5.0)
    final, _ = _trace_surfaces(rays, structure, params, be)
    assert be.to_numpy(final.valid).all(), "inf semi_aperture should never clip"


# ---------------------------------------------------------------------------
# JAX jit / vmap
# ---------------------------------------------------------------------------


def test_jit_runs(jax_backend):
    """_trace_surfaces runs under jax.jit without error."""
    be = jax_backend
    structure, params = _singlet_setup(be)
    rays = make_rays_batch(be, ys=np.array([-0.3, 0.0, 0.3]), z=-200.0)

    def trace_fn(r, p):
        return _trace_surfaces(r, structure, p, be)[0]

    result = be.jit(trace_fn)(rays, params)
    assert be.to_numpy(result.y).shape == (3,)


def test_vmap_over_fields(jax_backend):
    """vmap over field offsets matches loop — Tier C."""
    be = jax_backend
    structure, params = _4f_setup(be)

    def trace_one(y0):
        M_init = -y0 / 100.0
        N_init = be._jnp.sqrt(1.0 - M_init**2)
        r = Rays(
            x=be.array([0.0]),
            y=be.array([y0]),
            z=be.array([-100.0]),
            L=be.array([0.0]),
            M=be.array([M_init]),
            N=be.array([N_init]),
            i=be.array([1.0]),
            w=be.array([0.55]),
            opd=be.array([0.0]),
            valid=be.array([True]),
        )
        final, _ = _trace_surfaces(r, structure, params, be)
        return final.y

    field_vals = np.array([0.5, 1.0, 1.5])
    fields = be.array(field_vals)
    result_vmap = be.vmap(trace_one, in_axes=0, out_axes=0)(fields)
    result_loop = be.stack(
        [trace_one(float(be.to_numpy(fields[i]))) for i in range(3)], axis=0
    )
    np.testing.assert_allclose(
        be.to_numpy(result_vmap),
        be.to_numpy(result_loop),
        rtol=1e-9,
        atol=1e-10,
    )
