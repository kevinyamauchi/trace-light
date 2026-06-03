"""Backend constructors.

Usage::

    from trace_light.backends import numpy, jax

    be = numpy()
    be_jax = jax()  # raises ImportError if jax is not installed
"""

from __future__ import annotations

from trace_light.backends._numpy import NumpyBackend
from trace_light.backends._protocol import Backend, NotDifferentiable


def numpy() -> NumpyBackend:
    """Return a :class:`NumpyBackend` instance."""
    return NumpyBackend()


def jax():
    """Return a :class:`JaxBackend` instance.

    Raises :exc:`ImportError` if jax is not installed.
    """
    from trace_light.backends._jax import JaxBackend

    return JaxBackend()


__all__ = [
    "Backend",
    "NotDifferentiable",
    "NumpyBackend",
    "jax",
    "numpy",
]
