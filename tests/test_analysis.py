"""Phase 4 tests — spot, psf, irradiance (IMPLEMENTATION_PLAN §6/§7/§9/§11).

Assertion helpers are defined inline (conftest is fixtures only).
"""

from __future__ import annotations

import numpy as np

from optisketch.analysis import irradiance, psf, spot
from optisketch.backends import NumpyBackend
from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.rays import Rays
from optisketch.sources import collimated_source, point_source
from optisketch.systems import four_f

# ---------------------------------------------------------------------------
# Assertion helpers (inline, per §0.4)
# ---------------------------------------------------------------------------


def _ac(a, b, rtol, atol, be=None):
    a_np = be.to_numpy(a) if be is not None else np.asarray(a)
    b_np = be.to_numpy(b) if be is not None else np.asarray(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def assert_tier_a(a, b, be=None):
    _ac(a, b, 1e-5, 1e-7, be)


def assert_tier_b(a, b, be=None):
    _ac(a, b, 1e-11, 1e-12, be)


def assert_tier_d(a, b, be=None):
    _ac(a, b, 0.0, 1e-12, be)


# ---------------------------------------------------------------------------
# Helpers to build a focused on-axis ray bundle
# ---------------------------------------------------------------------------


def _traced_to_image(system, src):
    """Emit *src*, trace through *system*, propagate to the image plane."""
    be = system.backend
    rays = emit_and_trace(system, src)
    return _propagate_to_plane(rays, float(system.image_z), be)


def emit_and_trace(system, src):
    """Emit *src* and trace it through every surface of *system*."""
    from optisketch.sources import emit

    be = system.backend
    rays = emit(src, system)
    final, _ = _trace_surfaces(rays, system.structure, system.params, be)
    return final


# ---------------------------------------------------------------------------
# §6 spot
# ---------------------------------------------------------------------------


def test_spot_on_axis_centroid(backend):
    """On-axis point source: spot centroid is at the origin (symmetric pupil)."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    # hexapolar sampling is rotationally symmetric → exact zero centroid on-axis
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=6, pupil_pattern="hex")
    rays = _traced_to_image(sys, src)
    s = spot(rays, backend=be)
    np.testing.assert_allclose([s.cx, s.cy], [0.0, 0.0], atol=1e-9)


def test_spot_ideal_rms_zero(backend):
    """A perfectly converging bundle at its focus has near-zero RMS."""
    be = backend
    n = 16
    # Rays all aimed to cross (0,0,zf) exactly: direction = normalize(focus - origin)
    px = np.linspace(-2.0, 2.0, n)
    py = np.zeros(n)
    z0 = 0.0
    zf = 10.0
    dz = zf - z0
    norm = np.sqrt(px**2 + dz**2)
    L = -px / norm
    M = np.zeros(n)
    N = dz / norm
    rays = Rays(
        x=be.asarray(px),
        y=be.asarray(py),
        z=be.asarray(np.full(n, z0)),
        L=be.asarray(L),
        M=be.asarray(M),
        N=be.asarray(N),
        i=be.asarray(np.ones(n)),
        w=be.asarray(np.full(n, 0.55)),
        opd=be.asarray(np.zeros(n)),
        valid=be.asarray(np.ones(n, dtype=bool)),
    )
    at_focus = _propagate_to_plane(rays, zf, be)
    s = spot(at_focus, backend=be)
    assert s.rms < 1e-9


def test_spot_excludes_invalid(backend):
    """Rays flagged invalid do not contribute to centroid or RMS."""
    be = backend
    n = 8
    x = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0])
    valid = np.array([True, True, True, True, False, False, False, False])
    rays = Rays(
        x=be.asarray(x),
        y=be.asarray(np.zeros(n)),
        z=be.asarray(np.zeros(n)),
        L=be.asarray(np.zeros(n)),
        M=be.asarray(np.zeros(n)),
        N=be.asarray(np.ones(n)),
        i=be.asarray(np.ones(n)),
        w=be.asarray(np.full(n, 0.55)),
        opd=be.asarray(np.zeros(n)),
        valid=be.asarray(valid),
    )
    s = spot(rays, backend=be)
    assert s.n_valid == 4
    np.testing.assert_allclose(s.cx, 0.0, atol=1e-12)
    assert s.geo_radius < 1e-9


def test_spot_chief_vs_centroid(backend):
    """Off-axis bundle: chief and centroid reference modes give different RMS."""
    be = backend
    n = 9
    x = np.linspace(0.0, 4.0, n)  # asymmetric about its centroid
    rays = Rays(
        x=be.asarray(x),
        y=be.asarray(np.zeros(n)),
        z=be.asarray(np.zeros(n)),
        L=be.asarray(np.zeros(n)),
        M=be.asarray(np.zeros(n)),
        N=be.asarray(np.ones(n)),
        i=be.asarray(np.ones(n)),
        w=be.asarray(np.full(n, 0.55)),
        opd=be.asarray(np.zeros(n)),
        valid=be.asarray(np.ones(n, dtype=bool)),
    )
    s_cen = spot(rays, reference="centroid", backend=be)
    s_chief = spot(rays, reference="chief", backend=be)
    assert abs(s_cen.rms - s_chief.rms) > 1e-6


def test_spot_parity(jax_backend):
    """spot gives identical statistics on NumPy and JAX — Tier B."""
    be_jax = jax_backend
    be_np = NumpyBackend()
    sys_np = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_jax)
    src = point_source((0.5, 0.0), z_object=-100.0, n_samples=64)
    s_np = spot(_traced_to_image(sys_np, src), backend=be_np)
    s_jax = spot(_traced_to_image(sys_jax, src), backend=be_jax)
    assert_tier_b(s_np.cx, s_jax.cx)
    assert_tier_b(s_np.cy, s_jax.cy)
    assert_tier_b(s_np.rms, s_jax.rms)


# ---------------------------------------------------------------------------
# §7 psf
# ---------------------------------------------------------------------------


def test_psf_sums_to_one(backend):
    """The PSF kernel is normalised to unit sum."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    k = psf(sys, (0.0, 0.0), n_rays=256, grid=(32, 32))
    assert_tier_a(be.to_numpy(be.sum(k)), 1.0)


def test_psf_no_nan_inf(backend):
    """The PSF has the requested shape and contains no NaN/Inf."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    k = be.to_numpy(psf(sys, (0.0, 0.0), n_rays=200, grid=(48, 48)))
    assert k.shape == (48, 48)
    assert np.all(np.isfinite(k))


def test_psf_on_axis_centered(backend):
    """The on-axis PSF is centred: its intensity centroid lands at the grid centre.

    A geometric ray-histogram has a noisy modal peak (a single edge ray can be
    the max bin), so centring is verified via the intensity-weighted centroid,
    which is the quantity that matters for shift-free convolution.
    """
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    grid = (41, 41)
    k = be.to_numpy(psf(sys, (0.0, 0.0), n_rays=600, grid=grid))
    gy, gx = k.shape
    yy, xx = np.mgrid[0:gy, 0:gx].astype(float)
    cy = float(np.sum(k * yy))
    cx = float(np.sum(k * xx))
    assert abs(cy - (gy - 1) / 2.0) < 1.0
    assert abs(cx - (gx - 1) / 2.0) < 1.0


def test_psf_through_focus_symmetry(backend):
    """Defocus broadens the PSF symmetrically about best focus.

    The system's best focus is offset from ``image_z``, so the sweep is
    referenced to the empirically minimum-width focus before comparing the
    widths at symmetric defocus offsets.
    """
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)

    def width(focus):
        # fixed window so the comparison is meaningful across focus
        k = be.to_numpy(
            psf(sys, (0.0, 0.0), focus=focus, n_rays=600, grid=(64, 64), extent=0.5)
        )
        gy, gx = k.shape
        yy, xx = np.mgrid[0:gy, 0:gx].astype(float)
        cy = np.sum(k * yy)
        cx = np.sum(k * xx)
        var = np.sum(k * ((yy - cy) ** 2 + (xx - cx) ** 2))
        return float(np.sqrt(var))

    sweep = np.linspace(-2.0, 14.0, 17)
    widths = [width(f) for f in sweep]
    f0 = float(sweep[int(np.argmin(widths))])
    w_plus = width(f0 + 3.0)
    w_minus = width(f0 - 3.0)
    np.testing.assert_allclose(w_plus, w_minus, rtol=0.25)


def test_psf_parity(jax_backend):
    """psf gives identical kernels on NumPy and JAX — Tier B."""
    be_jax = jax_backend
    be_np = NumpyBackend()
    sys_np = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_jax)
    k_np = psf(sys_np, (0.3, 0.0), n_rays=256, grid=(32, 32), extent=0.4)
    k_jax = psf(sys_jax, (0.3, 0.0), n_rays=256, grid=(32, 32), extent=0.4)
    assert_tier_b(be_np.to_numpy(k_np), be_jax.to_numpy(k_jax))


# ---------------------------------------------------------------------------
# §9 irradiance
# ---------------------------------------------------------------------------


def test_irradiance_uniform_flat(backend):
    """A collimated beam at the pupil plane gives a roughly uniform histogram."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=8.0, backend=be)
    src = collimated_source((0.0, 0.0), pupil_pattern="disk", n_samples=4000)
    # sample at the pupil plane itself (rays are spread across the aperture)
    h = be.to_numpy(irradiance(sys, src, z=sys.pupil_z, grid=(12, 12), extent=8.0))
    occupied = h[h > 0]
    # within the illuminated disk the counts are fairly even
    assert occupied.std() / occupied.mean() < 0.8


def test_irradiance_valid_sum(backend):
    """Total histogram weight equals the summed weight of valid rays."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=200)
    final = emit_and_trace(sys, src)
    final = _propagate_to_plane(final, float(sys.image_z), be)
    wv = be.where(final.valid, final.i, be.zeros_like(final.i))
    expected = float(be.to_numpy(be.sum(wv)))
    h = irradiance(sys, src, z=float(sys.image_z), grid=(32, 32))
    assert_tier_a(be.to_numpy(be.sum(h)), expected)


def test_irradiance_parity(jax_backend):
    """irradiance gives identical histograms on NumPy and JAX — Tier B."""
    be_jax = jax_backend
    be_np = NumpyBackend()
    sys_np = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_jax)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=200)
    zi_np = float(sys_np.image_z)
    zi_jax = float(sys_jax.image_z)
    h_np = irradiance(sys_np, src, z=zi_np, grid=(24, 24), extent=2.0)
    h_jax = irradiance(sys_jax, src, z=zi_jax, grid=(24, 24), extent=2.0)
    assert_tier_b(be_np.to_numpy(h_np), be_jax.to_numpy(h_jax))


# ---------------------------------------------------------------------------
# §11 hygiene
# ---------------------------------------------------------------------------


def test_nan_does_not_poison_reductions(backend):
    """A NaN in an invalid ray must not corrupt centroid/RMS over valid rays."""
    be = backend
    n = 6
    x = np.array([0.0, 1.0, -1.0, np.nan, np.nan, np.nan])
    valid = np.array([True, True, True, False, False, False])
    rays = Rays(
        x=be.asarray(x),
        y=be.asarray(np.array([0.0, 0.0, 0.0, np.nan, np.nan, np.nan])),
        z=be.asarray(np.zeros(n)),
        L=be.asarray(np.zeros(n)),
        M=be.asarray(np.zeros(n)),
        N=be.asarray(np.ones(n)),
        i=be.asarray(np.ones(n)),
        w=be.asarray(np.full(n, 0.55)),
        opd=be.asarray(np.zeros(n)),
        valid=be.asarray(valid),
    )
    s = spot(rays, backend=be)
    assert np.isfinite(s.cx) and np.isfinite(s.rms)
    np.testing.assert_allclose(s.cx, 0.0, atol=1e-12)
    assert s.n_valid == 3
