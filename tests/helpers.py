"""Assertion helpers shared across the test suite.

Tolerance tiers (TEST_PLAN §0.2)
---------------------------------
* **A** reference/golden:      rtol=1e-5,  atol=1e-7
* **B** cross-backend parity:  rtol=1e-11, atol=1e-12
* **C** transform parity:      rtol=1e-9,  atol=1e-10
* **D** exact invariant:       atol=1e-12
"""

from __future__ import annotations

import numpy as np


def assert_allclose(a, b, rtol=1e-5, atol=1e-7, be=None):
    """Assert element-wise closeness after converting to NumPy.

    If ``be`` is provided it is used to call ``to_numpy``; otherwise
    ``np.asarray`` is used directly (works for NumPy arrays and plain Python
    scalars alike).
    """
    a_np = be.to_numpy(a) if be is not None else np.asarray(a)
    b_np = be.to_numpy(b) if be is not None else np.asarray(b)
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)


def assert_tier_a(a, b, be=None):
    """Tier A: reference/golden (rtol=1e-5, atol=1e-7)."""
    assert_allclose(a, b, rtol=1e-5, atol=1e-7, be=be)


def assert_tier_b(a, b, be=None):
    """Tier B: cross-backend parity (rtol=1e-11, atol=1e-12)."""
    assert_allclose(a, b, rtol=1e-11, atol=1e-12, be=be)


def assert_tier_c(a, b, be=None):
    """Tier C: transform parity (rtol=1e-9, atol=1e-10)."""
    assert_allclose(a, b, rtol=1e-9, atol=1e-10, be=be)


def assert_tier_d(a, b, be=None):
    """Tier D: exact invariant (atol=1e-12)."""
    assert_allclose(a, b, rtol=0.0, atol=1e-12, be=be)
