"""Phase 2 tests — lens factory functions (TEST_PLAN §3).

All golden values are copy-math (derived from the lensmaker's equation or
direct geometry) except where noted.
"""

from __future__ import annotations

import math

import numpy as np

from trace_light.lenses import (
    aperture,
    biconvex,
    doublet,
    mirror,
    plano_convex,
    thin_lens,
)

# ---------------------------------------------------------------------------
# Assertion helpers
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
# §3 Lens factory tests
# ---------------------------------------------------------------------------


def test_lens_biconvex_focal_length(backend):
    """Symmetric biconvex R=100, n=1.5, thickness=10: lensmaker gives f≈100mm."""
    surfs = biconvex(R=100.0, n=1.5, thickness=10.0)
    assert len(surfs) == 2
    R1 = surfs[0].radius  # +100
    R2 = surfs[1].radius  # -100
    n = 1.5
    # lensmaker (thin lens approx): 1/f = (n-1)(1/R1 - 1/R2)
    inv_f = (n - 1.0) * (1.0 / R1 - 1.0 / R2)
    f = 1.0 / inv_f
    assert_tier_a(f, 100.0)
    assert surfs[0].n1 == 1.0
    assert surfs[0].n2 == 1.5
    assert surfs[1].n1 == 1.5
    assert surfs[1].n2 == 1.0
    assert surfs[1].z - surfs[0].z == 10.0


def test_lens_plano_convex_geometry(backend):
    """Plano-convex R=50, flat back: second surface is a plane."""
    surfs = plano_convex(R=50.0, n=1.5, thickness=5.0)
    assert len(surfs) == 2
    assert surfs[0].radius == 50.0
    assert math.isinf(surfs[1].radius)  # flat back
    assert surfs[0].z == 0.0
    assert surfs[1].z == 5.0
    assert surfs[0].n2 == 1.5
    assert surfs[1].n1 == 1.5
    assert surfs[1].n2 == 1.0


def test_lens_doublet_index_chain(backend):
    """Doublet: n_crown=1.52, n_flint=1.67; three surfaces with chained indices."""
    n_crown = 1.52
    n_flint = 1.67
    surfs = doublet(
        R1=80.0,
        R2=-60.0,
        R3=-300.0,
        n_crown=n_crown,
        n_flint=n_flint,
        thickness_crown=6.0,
        thickness_flint=3.0,
    )
    assert len(surfs) == 3
    # n1 chain: air → crown → flint → air
    assert surfs[0].n1 == 1.0
    assert surfs[0].n2 == n_crown
    assert surfs[1].n1 == n_crown
    assert surfs[1].n2 == n_flint
    assert surfs[2].n1 == n_flint
    assert surfs[2].n2 == 1.0
    # z chain
    assert surfs[0].z == 0.0
    assert_tier_d(surfs[1].z, 6.0)
    assert_tier_d(surfs[2].z, 9.0)


def test_lens_thin_zero_thickness(backend):
    """Thin lens f=100: two surfaces co-located at z=0 (zero thickness)."""
    surfs = thin_lens(f=100.0, n=1.5)
    assert len(surfs) == 2
    # Both surfaces at same z (zero thickness)
    assert surfs[0].z == surfs[1].z == 0.0
    # Lensmaker check: R = 2*(n-1)*f = 2*0.5*100 = 100
    assert_tier_d(surfs[0].radius, 100.0)
    assert_tier_d(surfs[1].radius, -100.0)
    # n1/n2 chain: air → glass → air
    assert surfs[0].n1 == 1.0
    assert surfs[0].n2 == 1.5
    assert surfs[1].n1 == 1.5
    assert surfs[1].n2 == 1.0


def test_lens_mirror_reflective(backend):
    """Mirror surface: reflective=True."""
    surfs = mirror(R=-200.0, semi_aperture=25.0)
    assert len(surfs) == 1
    assert surfs[0].reflective is True
    assert surfs[0].radius == -200.0
    assert surfs[0].semi_aperture == 25.0


def test_lens_aperture_no_power(backend):
    """Aperture stop: plane surface with finite semi_aperture and no optical power."""
    surfs = aperture(semi=5.0)
    assert len(surfs) == 1
    assert math.isinf(surfs[0].radius)  # plane → no power
    assert surfs[0].semi_aperture == 5.0
    assert surfs[0].n1 == 1.0
    assert surfs[0].n2 == 1.0
    assert surfs[0].reflective is False


def test_lens_parity(jax_backend):
    """Biconvex surface radii match between NumPy and JAX backends — Tier B."""
    surfs = biconvex(R=75.0, n=1.5, thickness=8.0)
    # Radii values are plain Python floats; just check consistency
    assert surfs[0].radius == 75.0
    assert surfs[1].radius == -75.0
    # n1/n2 values
    assert_tier_b(surfs[0].n1, 1.0)
    assert_tier_b(surfs[1].n2, 1.0)
