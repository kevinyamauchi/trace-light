"""Phase 3 tests — Sources, pupil sampling, and emit (TEST_PLAN §Phase 3).

All golden values are copy-math (physics invariants) or model-pattern
(structural checks).
"""

from __future__ import annotations

import math

import numpy as np

from optisketch.backends import NumpyBackend
from optisketch.kernels import _trace_surfaces
from optisketch.sources import (
    Source,
    _pupil_disk,
    _pupil_hex,
    _pupil_ring,
    collimated_source,
    emit,
    point_source,
)
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


def assert_tier_c(a, b, be=None):
    _ac(a, b, 1e-9, 1e-10, be)


def assert_tier_d(a, b, be=None):
    _ac(a, b, 0.0, 1e-12, be)


# ---------------------------------------------------------------------------
# §3 Source factory tests
# ---------------------------------------------------------------------------


def test_source_point_n_rays_valid(backend):
    """point_source with n_samples=13: emit produces 13 valid rays."""
    be = backend
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=13)
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    rays = emit(src, sys)
    assert be.to_numpy(rays.valid).all()
    assert len(be.to_numpy(rays.x)) == 13


def test_source_point_unit_directions(backend):
    """Point source: emitted ray directions are unit vectors."""
    be = backend
    src = point_source((1.0, 0.5), z_object=-200.0, n_samples=7)
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    rays = emit(src, sys)
    L = be.to_numpy(rays.L)
    M = be.to_numpy(rays.M)
    N = be.to_numpy(rays.N)
    mag = np.sqrt(L**2 + M**2 + N**2)
    np.testing.assert_allclose(mag, 1.0, rtol=1e-12, atol=1e-12)


def test_source_point_chief_ray(backend):
    """Chief ray (pupil centre, px=py=0) aims from source toward pupil centre."""
    be = backend
    fx, fy, fz = 1.0, 0.0, -150.0
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((fx, fy), z_object=fz, pupil_pattern="disk", n_samples=1)
    rays = emit(src, sys)
    # Ray starts at the source position
    x_np = be.to_numpy(rays.x)
    y_np = be.to_numpy(rays.y)
    z_np = be.to_numpy(rays.z)
    np.testing.assert_allclose(x_np, fx, atol=1e-12)
    np.testing.assert_allclose(y_np, fy, atol=1e-12)
    np.testing.assert_allclose(z_np, fz, atol=1e-12)
    # Direction points toward the pupil (N > 0)
    N_np = be.to_numpy(rays.N)
    assert N_np[0] > 0.0


def test_source_collimated_direction(backend):
    """Collimated source angle=(0, theta): M=sin(theta), N=cos(theta)."""
    be = backend
    theta = math.radians(5.0)
    src = collimated_source((0.0, theta), n_samples=5)
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    rays = emit(src, sys)
    M_np = be.to_numpy(rays.M)
    N_np = be.to_numpy(rays.N)
    np.testing.assert_allclose(M_np, math.sin(theta), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(N_np, math.cos(theta), rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Pupil pattern tests
# ---------------------------------------------------------------------------


def test_pupil_disk_centroid(backend):
    """Disk centroid is approximately (0, 0) for large n."""
    px, py = _pupil_disk(500)
    np.testing.assert_allclose(np.mean(px), 0.0, atol=0.05)
    np.testing.assert_allclose(np.mean(py), 0.0, atol=0.05)


def test_pupil_disk_radius(backend):
    """All disk samples have radius ≤ 1."""
    px, py = _pupil_disk(100)
    r = np.sqrt(px**2 + py**2)
    assert np.all(r <= 1.0 + 1e-12)


def test_pupil_hex_count(backend):
    """Hexapolar count equals 1 + 3*rings*(rings+1) — from Optiland convention."""
    for rings in [1, 2, 3, 5]:
        px, _py = _pupil_hex(rings)
        expected = 1 + 3 * rings * (rings + 1)
        assert len(px) == expected, f"rings={rings}: got {len(px)}, expected {expected}"


def test_pupil_ring_symmetry(backend):
    """Ring pattern: all sample radii == 1."""
    px, py = _pupil_ring(12)
    r = np.sqrt(px**2 + py**2)
    np.testing.assert_allclose(r, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# emit vmap/parity tests (JAX only)
# ---------------------------------------------------------------------------


def test_emit_vmap_fields(jax_backend):
    """vmap(emit) over field matches looped emit — Tier C."""
    be = jax_backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)

    fields = np.array(
        [
            [0.0, 0.0, -150.0],
            [1.0, 0.0, -150.0],
            [-1.0, 0.5, -150.0],
        ]
    )

    # Loop reference
    loop_rays = []
    for f in fields:
        src = Source(
            kind="point", field=f, wavelength=0.55, pupil_pattern="fan", n_samples=5
        )
        r = emit(src, sys)
        loop_rays.append(be.to_numpy(r.y))
    loop_y = np.stack(loop_rays, axis=0)

    # vmap
    import jax
    import jax.numpy as jnp

    fields_jax = jnp.array(fields)

    def emit_one(field):
        src = Source(
            kind="point", field=field, wavelength=0.55, pupil_pattern="fan", n_samples=5
        )
        return emit(src, sys).y

    vmap_y = be.to_numpy(jax.vmap(emit_one)(fields_jax))
    np.testing.assert_allclose(vmap_y, loop_y, rtol=1e-9, atol=1e-10)


def test_emit_parity(jax_backend):
    """emit gives same ray bundle on NumPy and JAX backends — Tier B."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    sys_np = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_np)
    sys_jax = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be_jax)

    src = point_source((0.5, 0.5), z_object=-150.0, n_samples=7)

    rays_np = emit(src, sys_np)
    rays_jax = emit(src, sys_jax)

    assert_tier_b(be_np.to_numpy(rays_np.x), be_jax.to_numpy(rays_jax.x))
    assert_tier_b(be_np.to_numpy(rays_np.y), be_jax.to_numpy(rays_jax.y))
    assert_tier_b(be_np.to_numpy(rays_np.L), be_jax.to_numpy(rays_jax.L))
    assert_tier_b(be_np.to_numpy(rays_np.M), be_jax.to_numpy(rays_jax.M))
    assert_tier_b(be_np.to_numpy(rays_np.N), be_jax.to_numpy(rays_jax.N))


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


def test_emit_trace_end_to_end(backend):
    """Full pipeline: point_source → emit → _trace_surfaces → valid rays."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=7)
    rays = emit(src, sys)
    final, history = _trace_surfaces(rays, sys.structure, sys.params, be)
    assert len(history) == sys.structure.n_surfaces + 1
    # On-axis rays should all be valid through a well-designed system
    valid = be.to_numpy(final.valid)
    assert valid.all()
