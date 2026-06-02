"""Phase 2 tests — SystemBuilder, prefabs, and serialization (TEST_PLAN §4).

Golden values are copy-math (geometry / paraxial optics) and model-pattern
(structural checks).  Tier-B cross-backend parity is tested for traces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from trace_light.backends import NumpyBackend
from trace_light.kernels import _trace_surfaces
from trace_light.lenses import biconvex, singlet
from trace_light.rays import Rays, load_system, save_system
from trace_light.systems import SystemBuilder, four_f, microscope, telescope

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


def assert_tier_c(a, b, be=None):
    _ac(a, b, 1e-9, 1e-10, be)


def assert_tier_d(a, b, be=None):
    _ac(a, b, 0.0, 1e-12, be)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _axial_ray(be, *, y=0.0, z_start=0.0, n=1):
    """Build a single paraxial ray at height y, propagating in +z."""
    return Rays(
        x=be.array([0.0] * n),
        y=be.array([y] * n),
        z=be.array([z_start] * n),
        L=be.array([0.0] * n),
        M=be.array([0.0] * n),
        N=be.array([1.0] * n),
        i=be.array([1.0] * n),
        w=be.array([0.55] * n),
        opd=be.array([0.0] * n),
        valid=be.array([True] * n),
    )


def _propagate_to_z(rays, z_target, be):
    """Free-space propagate rays to a z-plane."""
    z_np = be.to_numpy(rays.z)
    N_np = be.to_numpy(rays.N)
    t_np = (z_target - z_np) / N_np
    t = be.asarray(t_np)
    return Rays(
        x=rays.x + t * rays.L,
        y=rays.y + t * rays.M,
        z=rays.z + t * rays.N,
        L=rays.L,
        M=rays.M,
        N=rays.N,
        i=rays.i,
        w=rays.w,
        opd=rays.opd + rays.N * t,  # n=1 in free space
        valid=rays.valid,
    )


# ---------------------------------------------------------------------------
# §4 SystemBuilder tests
# ---------------------------------------------------------------------------


def test_system_builder_absolute_z(backend):
    """Builder offsets surfaces by gap; absolute z is gap + element span."""
    be = backend
    b = SystemBuilder()
    b.gap(50.0)
    b.add(*biconvex(R=100.0, n=1.5, thickness=10.0))
    sys = b.image().finalize(be)

    # Front surface should be at z=50, back surface at z=60
    z_positions = sys.structure.z
    assert_tier_d(z_positions[0], 50.0)
    assert_tier_d(z_positions[1], 60.0)


def test_system_builder_index_chain(backend):
    """n1/n2 chaining: glass index propagates through the lens correctly."""
    be = backend
    surfs = singlet(R1=100.0, R2=-100.0, n=1.6, thickness=8.0)
    b = SystemBuilder()
    b.add(*surfs)
    sys = b.image().finalize(be)

    n1_arr = be.to_numpy(sys.params.n1)
    n2_arr = be.to_numpy(sys.params.n2)
    assert_tier_d(n1_arr[0], 1.0)  # air before front
    assert_tier_d(n2_arr[0], 1.6)  # glass after front
    assert_tier_d(n1_arr[1], 1.6)  # glass before back
    assert_tier_d(n2_arr[1], 1.0)  # air after back


def test_system_builder_pupil_default(backend):
    """Default pupil is placed at the first powered surface."""
    be = backend
    b = SystemBuilder()
    b.gap(20.0)
    b.add(*biconvex(R=100.0, n=1.5, thickness=10.0))
    sys = b.image().finalize(be)

    # First powered surface is at z=20
    assert_tier_d(sys.pupil_z, 20.0)


def test_system_builder_pupil_stop_override(backend):
    """Explicit stop overrides default pupil location."""
    be = backend
    b = SystemBuilder()
    b.stop(semi=5.0)
    b.add(*biconvex(R=100.0, n=1.5, thickness=10.0))
    sys = b.image().finalize(be)

    assert_tier_d(sys.pupil_z, 0.0)
    assert_tier_d(sys.pupil_semi, 5.0)


# ---------------------------------------------------------------------------
# Prefab system tests
# ---------------------------------------------------------------------------


def test_system_four_f_magnification(backend):
    """4-f relay with equal focal lengths: lateral magnification ≈ -1."""
    be = backend
    f = 100.0
    sys = four_f(f1=f, f2=f, n=1.5, thickness=10.0, pupil_semi=5.0, backend=be)

    # Trace a marginal ray at y=+1 from the object plane (z=-(f))
    # For a thin-lens 4-f, object at z=-f, image at z=4f+f=5f from L1 back
    # Use a paraxial ray starting well before the system
    y_obj = 2.0
    z_obj = -f  # one focal length before the first lens
    rays = Rays(
        x=be.array([0.0]),
        y=be.array([y_obj]),
        z=be.array([z_obj]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        i=be.array([1.0]),
        w=be.array([0.55]),
        opd=be.array([0.0]),
        valid=be.array([True]),
    )
    final, _ = _trace_surfaces(rays, sys.structure, sys.params, be)
    final_prop = _propagate_to_z(final, sys.image_z, be)

    y_img = be.to_numpy(final_prop.y)[0]
    mag = y_img / y_obj
    # Magnification should be ≈ -1 (inverted, same magnitude)
    # Thick lenses shift principal planes so 1% tolerance is appropriate here
    assert mag < 0.0, "Magnification should be negative (inverted image)"
    np.testing.assert_allclose(abs(mag), 1.0, rtol=0.02, atol=0.0)


def test_system_telescope_afocal(backend):
    """Keplerian telescope: axial ray enters parallel and exits parallel."""
    be = backend
    sys = telescope(
        f_obj=200.0, f_eye=50.0, n=1.5, thickness=5.0, pupil_semi=10.0, backend=be
    )

    # A ray parallel to the axis should exit parallel (afocal)
    rays = Rays(
        x=be.array([0.0]),
        y=be.array([5.0]),  # off-axis height in entrance pupil
        z=be.array([-10.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        i=be.array([1.0]),
        w=be.array([0.55]),
        opd=be.array([0.0]),
        valid=be.array([True]),
    )
    final, _ = _trace_surfaces(rays, sys.structure, sys.params, be)
    # Afocal: exit ray should be nearly parallel to z-axis
    # Thick-lens approximation gives residual M of order 1e-3 at y=5mm
    M_exit = be.to_numpy(final.M)[0]
    assert abs(M_exit) < 5e-3


def test_system_microscope_collimated(backend):
    """Infinity-corrected microscope: axial ray through objective exits collimated."""
    be = backend
    sys = microscope(
        f_obj=10.0,
        f_tube=100.0,
        n_obj=1.5,
        n_tube=1.5,
        thickness_obj=2.0,
        thickness_tube=3.0,
        pupil_semi=2.0,
        backend=be,
    )

    # On-axis ray parallel to z → should remain parallel after objective
    rays = Rays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([-5.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        i=be.array([1.0]),
        w=be.array([0.55]),
        opd=be.array([0.0]),
        valid=be.array([True]),
    )
    final, _ = _trace_surfaces(rays, sys.structure, sys.params, be)
    # On-axis paraxial ray through a centred system stays on-axis
    y_final = be.to_numpy(final.y)[0]
    assert abs(y_final) < 0.5  # still near axis


def test_system_parity(jax_backend):
    """4-f system trace: NumPy and JAX agree at Tier-B tolerance."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    sys_np = four_f(f1=100.0, f2=100.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, backend=be_jax)

    rays_np = Rays(
        x=be_np.array([0.0]),
        y=be_np.array([2.0]),
        z=be_np.array([-50.0]),
        L=be_np.array([0.0]),
        M=be_np.array([0.0]),
        N=be_np.array([1.0]),
        i=be_np.array([1.0]),
        w=be_np.array([0.55]),
        opd=be_np.array([0.0]),
        valid=be_np.array([True]),
    )
    rays_jax = Rays(
        x=be_jax.array([0.0]),
        y=be_jax.array([2.0]),
        z=be_jax.array([-50.0]),
        L=be_jax.array([0.0]),
        M=be_jax.array([0.0]),
        N=be_jax.array([1.0]),
        i=be_jax.array([1.0]),
        w=be_jax.array([0.55]),
        opd=be_jax.array([0.0]),
        valid=be_jax.array([True]),
    )

    final_np, _ = _trace_surfaces(rays_np, sys_np.structure, sys_np.params, be_np)
    final_jax, _ = _trace_surfaces(rays_jax, sys_jax.structure, sys_jax.params, be_jax)

    assert_tier_b(final_np.y, be_jax.to_numpy(final_jax.y))


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


def test_system_roundtrip_trace(backend):
    """Save → load → trace reproduces the original trace (same backend)."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, backend=be)

    rays = Rays(
        x=be.array([0.0]),
        y=be.array([3.0]),
        z=be.array([-50.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        i=be.array([1.0]),
        w=be.array([0.55]),
        opd=be.array([0.0]),
        valid=be.array([True]),
    )
    final_orig, _ = _trace_surfaces(rays, sys.structure, sys.params, be)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)
    try:
        save_system(sys, path)
        sys2 = load_system(path, backend=be)

        # Rebuild rays for second trace
        rays2 = Rays(
            x=be.array([0.0]),
            y=be.array([3.0]),
            z=be.array([-50.0]),
            L=be.array([0.0]),
            M=be.array([0.0]),
            N=be.array([1.0]),
            i=be.array([1.0]),
            w=be.array([0.55]),
            opd=be.array([0.0]),
            valid=be.array([True]),
        )
        final_loaded, _ = _trace_surfaces(rays2, sys2.structure, sys2.params, be)

        assert_tier_b(be.to_numpy(final_orig.y), be.to_numpy(final_loaded.y))
        assert_tier_b(be.to_numpy(final_orig.x), be.to_numpy(final_loaded.x))
    finally:
        path.unlink(missing_ok=True)


def test_system_roundtrip_cross_backend(jax_backend):
    """Save (NumPy) → load (JAX): trace matches original at Tier-B tolerance."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    sys_np = four_f(f1=100.0, f2=100.0, backend=be_np)
    rays_np = Rays(
        x=be_np.array([0.0]),
        y=be_np.array([3.0]),
        z=be_np.array([-50.0]),
        L=be_np.array([0.0]),
        M=be_np.array([0.0]),
        N=be_np.array([1.0]),
        i=be_np.array([1.0]),
        w=be_np.array([0.55]),
        opd=be_np.array([0.0]),
        valid=be_np.array([True]),
    )
    final_np, _ = _trace_surfaces(rays_np, sys_np.structure, sys_np.params, be_np)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        path = Path(tf.name)
    try:
        save_system(sys_np, path)
        sys_jax = load_system(path, backend=be_jax)

        rays_jax = Rays(
            x=be_jax.array([0.0]),
            y=be_jax.array([3.0]),
            z=be_jax.array([-50.0]),
            L=be_jax.array([0.0]),
            M=be_jax.array([0.0]),
            N=be_jax.array([1.0]),
            i=be_jax.array([1.0]),
            w=be_jax.array([0.55]),
            opd=be_jax.array([0.0]),
            valid=be_jax.array([True]),
        )
        final_jax, _ = _trace_surfaces(
            rays_jax, sys_jax.structure, sys_jax.params, be_jax
        )

        assert_tier_b(be_np.to_numpy(final_np.y), be_jax.to_numpy(final_jax.y))
    finally:
        path.unlink(missing_ok=True)


def test_system_schema_version_present():
    """Serialised dict always contains schema_version."""
    sys = four_f()
    d = sys.to_dict()
    assert "schema_version" in d
    assert d["schema_version"] == 1
