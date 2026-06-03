"""Pytest fixtures for the optisketch test suite.

Backend fixtures
----------------
``numpy_backend``   always available; returns a NumpyBackend.
``jax_backend``     skips the test when jax is not installed; returns a JaxBackend.
``backend``         parametrized over both; skips the jax variant when jax is absent.
"""

from __future__ import annotations

import pytest

from optisketch.backends import NumpyBackend


@pytest.fixture
def numpy_backend():
    """Return a NumpyBackend instance."""
    return NumpyBackend()


@pytest.fixture
def jax_backend():
    """Return a JaxBackend instance; skip if jax is not installed."""
    pytest.importorskip("jax", reason="jax not installed")
    from optisketch.backends import jax as make_jax

    return make_jax()


@pytest.fixture(params=["numpy", "jax"], ids=lambda b: f"backend={b}")
def backend(request):
    """Parametrize a test over numpy and jax backends."""
    if request.param == "numpy":
        return NumpyBackend()
    pytest.importorskip("jax", reason="jax not installed")
    from optisketch.backends import jax as make_jax

    return make_jax()
