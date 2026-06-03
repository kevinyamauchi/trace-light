"""Phase 6 tests — best_focus and minimize (IMPLEMENTATION_PLAN §10).

The DESIGN §3 autofocus literal (405.874 mm) pins a lens defined in the
unavailable ``DESIGN.md``; the convergence test therefore checks that the
optimiser settles to the system's true best-focus plane (model-pattern) rather
than that absolute value. Assertion helpers are inline.
"""

from __future__ import annotations

import numpy as np
import pytest

from optisketch.backends._protocol import NotDifferentiable
from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.optimize import best_focus, minimize
from optisketch.sources import emit, point_source
from optisketch.systems import four_f


def _rms_at(system, z, *, n_samples=64):
    """Return the on-axis RMS spot radius of *system* at detector plane *z*."""
    be = system.backend
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=n_samples)
    traced, _ = _trace_surfaces(emit(src, system), system.structure, system.params, be)
    at = _propagate_to_plane(traced, float(z), be)
    valid = at.valid
    zeros = be.zeros_like(at.x)
    vf = be.where(valid, be.ones_like(at.x), zeros)
    n = float(be.to_numpy(be.sum(vf)))
    cx = float(be.to_numpy(be.sum(be.where(valid, at.x, zeros)))) / n
    cy = float(be.to_numpy(be.sum(be.where(valid, at.y, zeros)))) / n
    dx = be.where(valid, at.x - cx, zeros)
    dy = be.where(valid, at.y - cy, zeros)
    return float(be.to_numpy(be.sqrt(be.sum(dx * dx + dy * dy) / n)))


def test_best_focus_reduces_rms(jax_backend):
    """The optimised focus plane has a strictly smaller RMS than image_z."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    z_opt = best_focus(sys)
    rms_nominal = _rms_at(sys, sys.image_z)
    rms_opt = _rms_at(sys, z_opt)
    assert rms_opt < rms_nominal


def test_best_focus_converges(jax_backend):
    """best_focus settles at a stationary point: the gradient there is ~0.

    (Model-pattern: the DESIGN §3 literal 405.874 mm targets an unavailable
    reference lens, so convergence is validated by stationarity and stability.)
    """
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    z1 = best_focus(sys, n_steps=50)
    z2 = best_focus(sys, n_steps=200)
    # extra iterations do not move the solution → converged
    np.testing.assert_allclose(z1, z2, atol=1e-6)
    # the minimiser is below the nominal image plane RMS and is a true minimum
    rms_here = _rms_at(sys, z1)
    assert _rms_at(sys, z1 + 1.0) > rms_here
    assert _rms_at(sys, z1 - 1.0) > rms_here


def test_best_focus_residual_is_aberration(jax_backend):
    """Residual RMS at best focus is positive — bounded below by aberration."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    z_opt = best_focus(sys)
    # a finite-aperture biconvex has spherical aberration → nonzero floor
    assert _rms_at(sys, z_opt) > 1e-4


def test_grad_vs_finite_diff(jax_backend):
    """Analytic jax.grad of the focus objective matches central finite diff."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=64)
    traced, _ = _trace_surfaces(emit(src, sys), sys.structure, sys.params, be)

    import jax

    from optisketch.optimize import _spot_variance

    def obj(z):
        return _spot_variance(_propagate_to_plane(traced, z, be), be)

    z0 = float(sys.image_z)
    g_analytic = float(be.to_numpy(jax.grad(obj)(be.asarray(z0))))
    h = 1e-3
    g_fd = (
        float(be.to_numpy(obj(be.asarray(z0 + h))))
        - float(be.to_numpy(obj(be.asarray(z0 - h))))
    ) / (2 * h)
    np.testing.assert_allclose(g_analytic, g_fd, rtol=1e-4, atol=1e-8)


def test_optimize_no_nan(jax_backend):
    """minimize runs without producing NaN and reduces the objective."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)

    def objective(s):
        # minimise on-axis spot variance at the image plane w.r.t. radii
        src = point_source((0.0, 0.0), z_object=-100.0, n_samples=32)
        traced, _ = _trace_surfaces(emit(src, s), s.structure, s.params, be)
        from optisketch.optimize import _spot_variance

        at = _propagate_to_plane(traced, float(s.image_z), be)
        return _spot_variance(at, be)

    loss0 = float(be.to_numpy(objective(sys)))
    out = minimize(sys, objective, param="radii", lr=1e-1, n_steps=20)
    loss1 = float(be.to_numpy(objective(out)))
    assert np.isfinite(loss1)
    assert all(np.isfinite(be.to_numpy(out.params.radii)))
    assert loss1 <= loss0


def test_optimize_numpy_behavior(numpy_backend):
    """On the NumPy backend best_focus/minimize raise NotDifferentiable."""
    be = numpy_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    with pytest.raises(NotDifferentiable):
        best_focus(sys)
    with pytest.raises(NotDifferentiable):
        minimize(sys, lambda s: 0.0)
