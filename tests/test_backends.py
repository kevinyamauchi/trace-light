"""Phase 0 tests — Backend abstraction (TEST_PLAN §2).

§2.1  Op parity & correctness
§2.2  Capability flags & graceful degradation
§2.3  Transform parity (JAX only)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trace_light.backends import NotDifferentiable, NumpyBackend

# ---------------------------------------------------------------------------
# §2.1  Op parity & correctness
# ---------------------------------------------------------------------------

# copy-value checks against analytic literals (TEST_PLAN §2.1)


def test_ops_sin_values(numpy_backend):
    # optiland tests/test_backend.py::test_sin (copy-value)
    x = numpy_backend.array([0.0, math.pi / 2, math.pi])
    np.testing.assert_allclose(numpy_backend.sin(x), [0.0, 1.0, 0.0], atol=1e-12)


def test_ops_cos_values(numpy_backend):
    x = numpy_backend.array([0.0, math.pi / 2, math.pi])
    np.testing.assert_allclose(numpy_backend.cos(x), [1.0, 0.0, -1.0], atol=1e-12)


def test_ops_sqrt_values(numpy_backend):
    x = numpy_backend.array([0.0, 1.0, 4.0, 9.0])
    np.testing.assert_allclose(numpy_backend.sqrt(x), [0.0, 1.0, 2.0, 3.0], atol=1e-14)


def test_ops_abs_values(numpy_backend):
    x = numpy_backend.array([-3.0, 0.0, 2.5])
    np.testing.assert_allclose(numpy_backend.abs(x), [3.0, 0.0, 2.5], atol=1e-14)


def test_ops_sign_values(numpy_backend):
    x = numpy_backend.array([-1.5, 0.0, 3.0])
    np.testing.assert_allclose(numpy_backend.sign(x), [-1.0, 0.0, 1.0], atol=1e-14)


def test_ops_where_values(numpy_backend):
    be = numpy_backend
    c = be.array([True, False, True])
    a = be.array([1.0, 2.0, 3.0])
    b = be.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(be.where(c, a, b), [1.0, 5.0, 3.0])


def test_ops_isfinite(numpy_backend):
    x = numpy_backend.array([1.0, float("inf"), float("nan"), -float("inf")])
    np.testing.assert_array_equal(
        numpy_backend.isfinite(x), [True, False, False, False]
    )


def test_ops_isnan(numpy_backend):
    x = numpy_backend.array([1.0, float("nan"), 0.0])
    np.testing.assert_array_equal(numpy_backend.isnan(x), [False, True, False])


def test_ops_stack(numpy_backend):
    be = numpy_backend
    a = be.array([1.0, 2.0])
    b = be.array([3.0, 4.0])
    s = be.stack([a, b], axis=0)
    assert s.shape == (2, 2)
    np.testing.assert_allclose(s[0], [1.0, 2.0])
    np.testing.assert_allclose(s[1], [3.0, 4.0])


def test_ops_sum_mean(numpy_backend):
    x = numpy_backend.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(numpy_backend.sum(x), 10.0)
    np.testing.assert_allclose(numpy_backend.mean(x), 2.5)


def test_ops_minimum_maximum(numpy_backend):
    be = numpy_backend
    a = be.array([1.0, 5.0, 3.0])
    b = be.array([4.0, 2.0, 3.0])
    np.testing.assert_allclose(be.minimum(a, b), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(be.maximum(a, b), [4.0, 5.0, 3.0])


def test_ops_zeros_full(numpy_backend):
    be = numpy_backend
    z = be.zeros(3)
    assert z.dtype == np.float64
    np.testing.assert_array_equal(z, [0.0, 0.0, 0.0])

    f = be.full(3, 7.0)
    assert f.dtype == np.float64
    np.testing.assert_array_equal(f, [7.0, 7.0, 7.0])


def test_ops_linspace(numpy_backend):
    ls = numpy_backend.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(ls, [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-14)


def test_all_ops_parity(jax_backend):
    """Tier-B cross-backend parity for all ops (authored fresh, TEST_PLAN §2.1)."""
    be_np = NumpyBackend()
    be_jax = jax_backend

    rng = np.random.default_rng(42)
    x_np = rng.uniform(0.1, 2.0, size=10).astype(np.float64)
    x_jax = be_jax.array(x_np)

    rtol, atol = 1e-11, 1e-12
    for name, fn_np, fn_jax in [
        ("sqrt", be_np.sqrt, be_jax.sqrt),
        ("sin", be_np.sin, be_jax.sin),
        ("cos", be_np.cos, be_jax.cos),
        ("abs", be_np.abs, be_jax.abs),
        ("sign", be_np.sign, be_jax.sign),
    ]:
        np.testing.assert_allclose(
            fn_np(x_np),
            be_jax.to_numpy(fn_jax(x_jax)),
            rtol=rtol,
            atol=atol,
            err_msg=f"parity failed for {name}",
        )


# ---------------------------------------------------------------------------
# §2.2  Capability flags & graceful degradation
# ---------------------------------------------------------------------------


def test_numpy_name(numpy_backend):
    assert numpy_backend.name == "numpy"


def test_numpy_not_differentiable(numpy_backend):
    assert numpy_backend.is_differentiable is False


def test_numpy_no_jit(numpy_backend):
    assert numpy_backend.supports_jit is False


def test_numpy_grad_raises(numpy_backend):
    with pytest.raises(NotDifferentiable):
        numpy_backend.grad(lambda x: x)


def test_numpy_jit_is_identity(numpy_backend):
    def f(x):
        return x * 2.0

    assert numpy_backend.jit(f) is f


def test_numpy_vmap_equals_loop(numpy_backend):
    """vmap(f)(batch) == stack([f(x) for x in batch]) — Tier C."""
    be = numpy_backend

    def f(x):
        return be.array([be.sum(x), be.mean(x)])

    batch = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    result = be.vmap(f, in_axes=0, out_axes=0)(batch)
    expected = np.stack([f(batch[i]) for i in range(4)], axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-10)


def test_numpy_float64_output(numpy_backend):
    be = numpy_backend
    x = be.array([1.0, 2.0])
    assert x.dtype == np.float64
    assert be.sqrt(x).dtype == np.float64
    assert be.zeros(3).dtype == np.float64


def test_jax_name(jax_backend):
    assert jax_backend.name == "jax"


def test_jax_differentiable(jax_backend):
    assert jax_backend.is_differentiable is True


def test_jax_supports_jit(jax_backend):
    assert jax_backend.supports_jit is True


def test_jax_float64_output(jax_backend):
    x = jax_backend.array([1.0, 2.0])
    assert x.dtype == np.float64


# ---------------------------------------------------------------------------
# §2.3  Transform parity (JAX only)
# ---------------------------------------------------------------------------


def test_jax_jit_matches_eager(jax_backend):
    """jit(f)(x) == f(x) — Tier C."""
    be = jax_backend

    def f(x):
        return be.sin(x) * be.cos(x) + be.sqrt(be.abs(x))

    x = be.array([0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        be.to_numpy(be.jit(f)(x)), be.to_numpy(f(x)), rtol=1e-9, atol=1e-10
    )


def test_jax_vmap_matches_loop(jax_backend):
    """vmap(f)(batch) == stack([f(row) for row]) — Tier C."""
    be = jax_backend

    def f(x):
        return be.sin(x) + be.cos(x)

    data = np.arange(12.0).reshape(4, 3)
    batch = be.array(data)
    result = be.vmap(f, in_axes=0, out_axes=0)(batch)
    expected = np.stack([be.to_numpy(f(be.array(data[i]))) for i in range(4)], axis=0)
    np.testing.assert_allclose(be.to_numpy(result), expected, rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Pyodide / no-JAX import isolation
# ---------------------------------------------------------------------------


def test_import_without_jax(monkeypatch):
    """``import trace_light`` must not import jax at module level."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.setitem(sys.modules, "jax.numpy", None)

    import trace_light

    importlib.reload(trace_light)
