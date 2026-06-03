"""Tests for the public ``trace`` wrapper (DESIGN §10) and traced spacings
(``_Params.z`` / ``_Params.semi_aperture``, DESIGN §4.2).

Assertion helpers are defined inline (conftest is fixtures only).
"""

from __future__ import annotations

import numpy as np

from optisketch import trace
from optisketch.backends import NumpyBackend
from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.optimize import minimize
from optisketch.sources import emit, point_source
from optisketch.systems import four_f


def _ac(a, b, rtol, atol, be=None):
    a_np = be.to_numpy(a) if be is not None else np.asarray(a)
    b_np = be.to_numpy(b) if be is not None else np.asarray(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def assert_tier_b(a, b, be=None):
    _ac(a, b, 1e-11, 1e-12, be)


# ---------------------------------------------------------------------------
# Public trace wrapper (#6, DESIGN §10)
# ---------------------------------------------------------------------------


def test_trace_matches_emit_plus_trace_surfaces(backend):
    """rt.trace(sys, src) == emit + _trace_surfaces, and history is the right length."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.3, 0.0), z_object=-100.0, n_samples=21)

    final, history = trace(sys, src)

    rays = emit(src, sys)
    final_ref, _history_ref = _trace_surfaces(rays, sys.structure, sys.params, be)

    assert len(history) == sys.structure.n_surfaces + 1
    assert_tier_b(final.x, final_ref.x, be)
    assert_tier_b(final.y, final_ref.y, be)
    assert_tier_b(final.N, final_ref.N, be)


def test_trace_backend_override():
    """A NumPy-built system can be traced on JAX via the backend override."""
    import pytest

    pytest.importorskip("jax")
    from optisketch.backends import jax as make_jax

    be_np = NumpyBackend()
    be_jax = make_jax()
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_np)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=13)

    final_np, _ = trace(sys, src)
    final_jax, _ = trace(sys, src, backend=be_jax)

    assert_tier_b(be_np.to_numpy(final_np.y), be_jax.to_numpy(final_jax.y))


# ---------------------------------------------------------------------------
# Traced spacings (#7, DESIGN §4.2)
# ---------------------------------------------------------------------------


def test_z_and_semi_aperture_are_traced_params(backend):
    """z and semi_aperture live on the traced _Params, not the static _Structure."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    # Structure holds only the static kind flags.
    assert not hasattr(sys.structure, "z")
    assert not hasattr(sys.structure, "semi_apertures")
    # Params carries z and semi_aperture as arrays of length n_surfaces.
    assert len(be.to_numpy(sys.params.z)) == sys.structure.n_surfaces
    assert len(be.to_numpy(sys.params.semi_aperture)) == sys.structure.n_surfaces


def test_spacing_is_differentiable(jax_backend):
    """The trace is differentiable w.r.t. a per-surface axial position (spacing).

    This is the payoff of moving ``z`` into traced ``_Params``: gradient flows
    through the spacing, which the previous static-``z`` layout could not do.
    """
    be = jax_backend
    import jax

    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=64)
    rays = emit(src, sys)

    def spot_variance_from_last_z(z_last):
        # shift the final lens surface axially and measure on-axis spot variance
        z_arr = sys.params.z.at[-1].set(z_last)
        params = sys.params._replace(z=z_arr)
        traced, _ = _trace_surfaces(rays, sys.structure, params, be)
        at = _propagate_to_plane(traced, float(sys.image_z), be)
        v = at.valid
        zr = be.zeros_like(at.x)
        n = be.sum(be.where(v, be.ones_like(at.x), zr))
        cx = be.sum(be.where(v, at.x, zr)) / n
        cy = be.sum(be.where(v, at.y, zr)) / n
        dx = be.where(v, at.x - cx, zr)
        dy = be.where(v, at.y - cy, zr)
        return be.sum(dx * dx + dy * dy) / n

    z0 = float(be.to_numpy(sys.params.z)[-1])
    g = float(be.to_numpy(jax.grad(spot_variance_from_last_z)(be.asarray(z0))))
    assert np.isfinite(g)
    # central finite difference agrees with the analytic gradient
    h = 1e-3
    fd = (
        float(be.to_numpy(spot_variance_from_last_z(be.asarray(z0 + h))))
        - float(be.to_numpy(spot_variance_from_last_z(be.asarray(z0 - h))))
    ) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-9)


def test_minimize_over_spacing_runs(jax_backend):
    """optimize.minimize can now target the spacing field ``z`` (DESIGN §15)."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=32)
    rays = emit(src, sys)

    def objective(s):
        traced, _ = _trace_surfaces(rays, s.structure, s.params, be)
        at = _propagate_to_plane(traced, float(s.image_z), be)
        v = at.valid
        zr = be.zeros_like(at.x)
        n = be.sum(be.where(v, be.ones_like(at.x), zr))
        cx = be.sum(be.where(v, at.x, zr)) / n
        cy = be.sum(be.where(v, at.y, zr)) / n
        dx = be.where(v, at.x - cx, zr)
        dy = be.where(v, at.y - cy, zr)
        return be.sum(dx * dx + dy * dy) / n

    loss0 = float(be.to_numpy(objective(sys)))
    out = minimize(sys, objective, param="z", lr=1e-1, n_steps=10)
    loss1 = float(be.to_numpy(objective(out)))
    assert np.isfinite(loss1)
    assert loss1 <= loss0
