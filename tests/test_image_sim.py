"""Phase 5 tests — image_sim (IMPLEMENTATION_PLAN §8).

Assertion helpers are defined inline (conftest is fixtures only).
"""

from __future__ import annotations

import numpy as np

from optisketch.analysis import image_sim
from optisketch.backends import NumpyBackend
from optisketch.systems import four_f

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _ac(a, b, rtol, atol, be=None):
    a_np = be.to_numpy(a) if be is not None else np.asarray(a)
    b_np = be.to_numpy(b) if be is not None else np.asarray(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def assert_tier_b(a, b, be=None):
    _ac(a, b, 1e-11, 1e-12, be)


def assert_tier_d(a, b, be=None):
    _ac(a, b, 0.0, 1e-12, be)


def _point_object(ny=32, nx=32):
    """Return a (ny, nx) object with a single bright central pixel."""
    obj = np.zeros((ny, nx), dtype=np.float64)
    obj[ny // 2, nx // 2] = 1.0
    return obj


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_image_sim_2d_shape(backend):
    """A 2-D object with scalar focus yields a 2-D (ny, nx) image."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=4.0, backend=be)
    obj = be.asarray(_point_object(24, 24))
    img = image_sim(
        sys, obj, extent=0.5, psf="single", focus=0.0, psf_grid=(15, 15), n_rays=128
    )
    assert tuple(be.to_numpy(img).shape) == (24, 24)


def test_image_sim_3d_shape(backend):
    """A 3-D object + focus array yields a (nf, ny, nx) focal stack."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=4.0, backend=be)
    obj = be.asarray(np.stack([_point_object(20, 20) for _ in range(3)], axis=0))
    focus = np.array([-2.0, 0.0, 2.0])
    img = image_sim(
        sys,
        obj,
        extent=0.5,
        psf="single",
        focus=focus,
        psf_grid=(15, 15),
        n_rays=128,
        depth_extent=1.0,
    )
    assert tuple(be.to_numpy(img).shape) == (3, 20, 20)


def test_image_sim_no_nan(backend):
    """The image has positive maximum and no NaN."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=4.0, backend=be)
    obj = be.asarray(_point_object(24, 24))
    img = be.to_numpy(
        image_sim(
            sys,
            obj,
            extent=0.5,
            psf="varying",
            grid=(2, 2),
            psf_grid=(15, 15),
            n_rays=128,
        )
    )
    assert np.all(np.isfinite(img))
    assert img.max() > 0.0


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_image_sim_single_matches_varying(backend):
    """Near-axis (shift-invariant) system: single ≈ varying image energy/structure."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be)
    obj = be.asarray(_point_object(24, 24))
    # near best focus, small field span → PSF is effectively field-invariant
    common = {"focus": 6.0, "psf_grid": (15, 15), "n_rays": 2000}
    img_single = be.to_numpy(image_sim(sys, obj, extent=0.15, psf="single", **common))
    img_varying = be.to_numpy(
        image_sim(sys, obj, extent=0.15, psf="varying", grid=(2, 2), **common)
    )
    # total energy matches closely (partition of unity, field-invariant PSF)
    np.testing.assert_allclose(img_single.sum(), img_varying.sum(), rtol=1e-6)
    # the two images are essentially identical in structure
    corr = np.corrcoef(img_single.ravel(), img_varying.ravel())[0, 1]
    assert corr > 0.99
    peak = img_single.max()
    np.testing.assert_allclose(img_varying, img_single, atol=0.05 * peak)


def test_image_sim_energy_conserved(backend):
    """Image energy tracks object energy times throughput (~1, unclipped PSF)."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be)
    obj_np = np.zeros((28, 28), dtype=np.float64)
    obj_np[10:18, 10:18] = 1.0  # block of energy away from the borders
    obj = be.asarray(obj_np)
    img = be.to_numpy(
        image_sim(
            sys, obj, extent=0.3, psf="single", focus=6.0, psf_grid=(15, 15), n_rays=400
        )
    )
    # normalised PSF (sum 1) and unclipped rays → energy approximately preserved
    np.testing.assert_allclose(img.sum(), obj_np.sum(), rtol=0.05)


def test_image_sim_parity(jax_backend):
    """image_sim gives the same image on NumPy and JAX — Tier B."""
    be_jax = jax_backend
    be_np = NumpyBackend()
    sys_np = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be_jax)
    obj_np = _point_object(24, 24)
    common = {
        "extent": 0.3,
        "psf": "single",
        "focus": 0.0,
        "psf_grid": (15, 15),
        "n_rays": 256,
    }
    img_np = image_sim(sys_np, be_np.asarray(obj_np), **common)
    img_jax = image_sim(sys_jax, be_jax.asarray(obj_np), **common)
    _ac(be_np.to_numpy(img_np), be_jax.to_numpy(img_jax), 1e-9, 1e-10)
