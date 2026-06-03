"""JAX backend — constructed only when jax is importable.

``import jax`` is intentionally kept inside ``JaxBackend.__init__`` so that
importing this module does not require JAX to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trace_light.backends._protocol import Backend

if TYPE_CHECKING:
    from collections.abc import Callable


class JaxBackend(Backend):
    """Backend that wraps JAX.

    * ``jit`` compiles with ``jax.jit``.
    * ``vmap`` uses ``jax.vmap``.
    * ``grad`` uses ``jax.grad``.
    * 64-bit floats are enabled via ``jax_enable_x64`` in ``__init__``.
    """

    name: str = "jax"
    is_differentiable: bool = True
    supports_jit: bool = True

    def __init__(self) -> None:
        """Initialise the JAX backend.

        Imports JAX, enables 64-bit float support, and caches the ``jax``
        and ``jax.numpy`` module references for use in all methods.

        Raises
        ------
        ImportError
            If JAX is not installed in the current environment.
        """
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        self._jax = jax
        self._jnp = jnp

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: Any, dtype: Any = None) -> Any:
        """Convert *x* to a JAX array.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype.

        Returns
        -------
        jax.Array
            JAX array containing the data from *x*.
        """
        return self._jnp.asarray(x, dtype=dtype)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert *x* to a JAX array, avoiding a copy when possible.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype.

        Returns
        -------
        jax.Array
            JAX array containing the data from *x*.
        """
        return self._jnp.asarray(x, dtype=dtype)

    def zeros(self, shape: int | tuple[int, ...], dtype: Any = None) -> Any:
        """Return a zero-filled JAX array of the given shape.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the output array.
        dtype : dtype, optional
            Desired dtype. Defaults to ``float64``.

        Returns
        -------
        jax.Array
            Zero-filled array of the requested shape.
        """
        dt = dtype if dtype is not None else self._jnp.float64
        return self._jnp.zeros(shape, dtype=dt)

    def zeros_like(self, x: Any) -> Any:
        """Return a zero array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : jax.Array
            Reference array.

        Returns
        -------
        jax.Array
            Zero-filled array matching the shape and dtype of *x*.
        """
        return self._jnp.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        """Return a ones array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : jax.Array
            Reference array.

        Returns
        -------
        jax.Array
            One-filled array matching the shape and dtype of *x*.
        """
        return self._jnp.ones_like(x)

    def full_like(self, x: Any, fill_value: Any) -> Any:
        """Return an array filled with *fill_value*, shaped like *x*.

        Parameters
        ----------
        x : jax.Array
            Reference array.
        fill_value : scalar
            Value to fill the output array with.

        Returns
        -------
        jax.Array
            Array matching the shape and dtype of *x*, filled with
            *fill_value*.
        """
        return self._jnp.full_like(x, fill_value)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
    ) -> Any:
        """Return a JAX array of the given shape filled with *fill_value*.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the output array.
        fill_value : scalar
            Value to fill the output array with.
        dtype : dtype, optional
            Desired dtype. Defaults to ``float64``.

        Returns
        -------
        jax.Array
            Array of the requested shape filled with *fill_value*.
        """
        return self._jnp.full(
            shape, fill_value, dtype=dtype if dtype is not None else self._jnp.float64
        )

    def linspace(self, start: float, stop: float, num: int) -> Any:
        """Return *num* evenly spaced values over ``[start, stop]``.

        Parameters
        ----------
        start : float
            First value of the sequence.
        stop : float
            Last value of the sequence (inclusive).
        num : int
            Number of values to generate.

        Returns
        -------
        jax.Array
            1-D array of *num* evenly spaced floats.
        """
        return self._jnp.linspace(start, stop, num)

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------

    def sqrt(self, x: Any) -> Any:
        """Elementwise square root.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Elementwise square root of *x*.
        """
        return self._jnp.sqrt(x)

    def sin(self, x: Any) -> Any:
        """Elementwise sine (input in radians).

        Parameters
        ----------
        x : jax.Array
            Input array (radians).

        Returns
        -------
        jax.Array
            Elementwise sine of *x*.
        """
        return self._jnp.sin(x)

    def cos(self, x: Any) -> Any:
        """Elementwise cosine (input in radians).

        Parameters
        ----------
        x : jax.Array
            Input array (radians).

        Returns
        -------
        jax.Array
            Elementwise cosine of *x*.
        """
        return self._jnp.cos(x)

    def abs(self, x: Any) -> Any:
        """Elementwise absolute value.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Elementwise absolute value of *x*.
        """
        return self._jnp.abs(x)

    def sign(self, x: Any) -> Any:
        """Elementwise sign: -1, 0, or +1.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array
            Array with values in ``{-1, 0, +1}``.
        """
        return self._jnp.sign(x)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        """Elementwise conditional selection.

        Parameters
        ----------
        cond : bool array
            Condition mask.
        x : jax.Array
            Values selected where *cond* is True.
        y : jax.Array
            Values selected where *cond* is False.

        Returns
        -------
        jax.Array
            Array with values from *x* where *cond* is True, else from *y*.
        """
        return self._jnp.where(cond, x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        """Elementwise minimum of *x* and *y*.

        Parameters
        ----------
        x : jax.Array
            First input array.
        y : jax.Array
            Second input array.

        Returns
        -------
        jax.Array
            Elementwise minimum.
        """
        return self._jnp.minimum(x, y)

    def maximum(self, x: Any, y: Any) -> Any:
        """Elementwise maximum of *x* and *y*.

        Parameters
        ----------
        x : jax.Array
            First input array.
        y : jax.Array
            Second input array.

        Returns
        -------
        jax.Array
            Elementwise maximum.
        """
        return self._jnp.maximum(x, y)

    def isfinite(self, x: Any) -> Any:
        """Elementwise test for finite values.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array of bool
            True where *x* is finite (not NaN and not Inf).
        """
        return self._jnp.isfinite(x)

    def isnan(self, x: Any) -> Any:
        """Elementwise test for NaN.

        Parameters
        ----------
        x : jax.Array
            Input array.

        Returns
        -------
        jax.Array of bool
            True where *x* is NaN.
        """
        return self._jnp.isnan(x)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum of array elements over the given axis.

        Parameters
        ----------
        x : jax.Array
            Input array.
        axis : int, optional
            Axis along which to sum. When None, sums all elements.

        Returns
        -------
        jax.Array or scalar
            Sum of *x*.
        """
        return self._jnp.sum(x, axis=axis)

    def mean(self, x: Any, axis: int | None = None) -> Any:
        """Mean of array elements over the given axis.

        Parameters
        ----------
        x : jax.Array
            Input array.
        axis : int, optional
            Axis along which to average. When None, averages all elements.

        Returns
        -------
        jax.Array or scalar
            Mean of *x*.
        """
        return self._jnp.mean(x, axis=axis)

    def max(self, x: Any, axis: int | None = None) -> Any:
        """Maximum of array elements over the given axis.

        Parameters
        ----------
        x : jax.Array
            Input array.
        axis : int, optional
            Axis along which to find the maximum. When None, returns the
            global maximum.

        Returns
        -------
        jax.Array or scalar
            Maximum value(s) of *x*.
        """
        return self._jnp.max(x, axis=axis)

    # ------------------------------------------------------------------
    # Array manipulation
    # ------------------------------------------------------------------

    def stack(self, arrays: list[Any], axis: int = 0) -> Any:
        """Stack a sequence of arrays along a new axis.

        Parameters
        ----------
        arrays : list of jax.Array
            Arrays to stack. All must have the same shape.
        axis : int, optional
            Position in the result array where the new axis is inserted.
            Default is 0.

        Returns
        -------
        jax.Array
            Stacked array with one more dimension than the inputs.
        """
        return self._jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Any], axis: int = 0) -> Any:
        """Concatenate arrays along an existing axis.

        Parameters
        ----------
        arrays : list of jax.Array
            Arrays to concatenate. All must have the same shape except
            along *axis*.
        axis : int, optional
            Axis along which to concatenate. Default is 0.

        Returns
        -------
        jax.Array
            Concatenated array.
        """
        return self._jnp.concatenate(arrays, axis=axis)

    # ------------------------------------------------------------------
    # Function transforms
    # ------------------------------------------------------------------

    def jit(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """JIT-compile *f* with ``jax.jit``.

        Parameters
        ----------
        f : callable
            Function to compile.

        Returns
        -------
        callable
            JAX-compiled version of *f*.
        """
        return self._jax.jit(f)

    def vmap(
        self,
        f: Callable[..., Any],
        in_axes: int | tuple[int | None, ...] = 0,
        out_axes: int = 0,
    ) -> Callable[..., Any]:
        """Vectorise *f* over a batch axis using ``jax.vmap``.

        Parameters
        ----------
        f : callable
            Function to vectorise.
        in_axes : int or tuple of int or None, optional
            Which axis of each argument to batch over. None means the
            argument is not batched. Default is 0.
        out_axes : int, optional
            Axis in the output where the batch dimension is placed.
            Default is 0.

        Returns
        -------
        callable
            JAX-vectorised version of *f*.
        """
        return self._jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

    def grad(
        self,
        f: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable[..., Any]:
        """Return the gradient function of *f* using ``jax.grad``.

        Parameters
        ----------
        f : callable
            Scalar-valued function to differentiate.
        argnums : int or tuple of int, optional
            Which positional argument(s) to differentiate with respect to.
            Default is 0 (first argument).

        Returns
        -------
        callable
            Function that computes the gradient of *f* with respect to
            the argument(s) specified by *argnums*.
        """
        return self._jax.grad(f, argnums=argnums)

    # ------------------------------------------------------------------
    # Histogramming and convolution
    # ------------------------------------------------------------------

    def histogram2d(
        self,
        x: Any,
        y: Any,
        bins: tuple[int, int],
        range: tuple[tuple[float, float], tuple[float, float]],
        weights: Any = None,
    ) -> Any:
        """Return the 2-D weighted histogram counts via ``jax.numpy.histogram2d``.

        Parameters
        ----------
        x : jax.Array
            First-coordinate sample values (binned along axis 0).
        y : jax.Array
            Second-coordinate sample values (binned along axis 1).
        bins : tuple of int
            ``(nx, ny)`` bin counts.
        range : tuple of tuple of float
            ``((xmin, xmax), (ymin, ymax))`` histogram extent.
        weights : jax.Array, optional
            Per-sample weights. None → unit weights.

        Returns
        -------
        jax.Array
            2-D histogram counts of shape *bins*.
        """
        h, _, _ = self._jnp.histogram2d(
            x,
            y,
            bins=list(bins),
            range=range,
            weights=weights,
        )
        return h

    def fftconvolve(self, a: Any, b: Any, mode: str = "same") -> Any:
        """FFT-based convolution via ``jax.scipy.signal.fftconvolve``.

        Parameters
        ----------
        a : jax.Array
            First input array.
        b : jax.Array
            Convolution kernel.
        mode : str, optional
            ``"full"``, ``"same"``, or ``"valid"``. Default ``"same"``.

        Returns
        -------
        jax.Array
            Convolution of *a* and *b*.
        """
        from jax.scipy.signal import fftconvolve as _fftconvolve

        return _fftconvolve(a, b, mode=mode)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert a JAX array to a NumPy ndarray.

        Parameters
        ----------
        x : jax.Array
            JAX array to convert.

        Returns
        -------
        numpy.ndarray
            NumPy copy of *x*.
        """
        return np.asarray(x)
