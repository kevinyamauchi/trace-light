"""NumPy backend — always available, not differentiable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trace_light.backends._protocol import Backend, NotDifferentiable

if TYPE_CHECKING:
    from collections.abc import Callable


class NumpyBackend(Backend):
    """Backend that wraps NumPy.

    * ``jit`` is the identity (returns ``f`` unchanged).
    * ``vmap`` is a Python loop + ``np.stack``.
    * ``grad`` raises :exc:`NotDifferentiable`.
    * All float arrays default to ``float64``.
    """

    name: str = "numpy"
    is_differentiable: bool = False
    supports_jit: bool = False

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: Any, dtype: Any = None) -> np.ndarray:
        """Convert *x* to a NumPy array.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype.

        Returns
        -------
        numpy.ndarray
            Array containing the data from *x*.
        """
        return np.asarray(x, dtype=dtype)

    def asarray(self, x: Any, dtype: Any = None) -> np.ndarray:
        """Convert *x* to a NumPy array, avoiding a copy when possible.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype.

        Returns
        -------
        numpy.ndarray
            Array containing the data from *x*.
        """
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape: int | tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Return a zero-filled NumPy array of the given shape.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the output array.
        dtype : dtype, optional
            Desired dtype. Defaults to ``float64``.

        Returns
        -------
        numpy.ndarray
            Zero-filled array of the requested shape.
        """
        return np.zeros(shape, dtype=dtype if dtype is not None else np.float64)

    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        """Return a zero array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : numpy.ndarray
            Reference array.

        Returns
        -------
        numpy.ndarray
            Zero-filled array matching the shape and dtype of *x*.
        """
        return np.zeros_like(x)

    def ones_like(self, x: np.ndarray) -> np.ndarray:
        """Return a ones array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : numpy.ndarray
            Reference array.

        Returns
        -------
        numpy.ndarray
            One-filled array matching the shape and dtype of *x*.
        """
        return np.ones_like(x)

    def full_like(self, x: np.ndarray, fill_value: Any) -> np.ndarray:
        """Return an array filled with *fill_value*, shaped like *x*.

        Parameters
        ----------
        x : numpy.ndarray
            Reference array.
        fill_value : scalar
            Value to fill the output array with.

        Returns
        -------
        numpy.ndarray
            Array matching the shape and dtype of *x*, filled with
            *fill_value*.
        """
        return np.full_like(x, fill_value)

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
    ) -> np.ndarray:
        """Return a NumPy array of the given shape filled with *fill_value*.

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
        numpy.ndarray
            Array of the requested shape filled with *fill_value*.
        """
        dt = dtype if dtype is not None else np.float64
        return np.full(shape, fill_value, dtype=dt)

    def linspace(self, start: float, stop: float, num: int) -> np.ndarray:
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
        numpy.ndarray
            1-D array of *num* evenly spaced floats.
        """
        return np.linspace(start, stop, num)

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Elementwise square root.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Elementwise square root of *x*.
        """
        return np.sqrt(x)

    def sin(self, x: np.ndarray) -> np.ndarray:
        """Elementwise sine (input in radians).

        Parameters
        ----------
        x : numpy.ndarray
            Input array (radians).

        Returns
        -------
        numpy.ndarray
            Elementwise sine of *x*.
        """
        return np.sin(x)

    def cos(self, x: np.ndarray) -> np.ndarray:
        """Elementwise cosine (input in radians).

        Parameters
        ----------
        x : numpy.ndarray
            Input array (radians).

        Returns
        -------
        numpy.ndarray
            Elementwise cosine of *x*.
        """
        return np.cos(x)

    def abs(self, x: np.ndarray) -> np.ndarray:
        """Elementwise absolute value.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Elementwise absolute value of *x*.
        """
        return np.abs(x)

    def sign(self, x: np.ndarray) -> np.ndarray:
        """Elementwise sign: -1, 0, or +1.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Array with values in ``{-1, 0, +1}``.
        """
        return np.sign(x)

    def where(self, cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Elementwise conditional selection.

        Parameters
        ----------
        cond : bool array
            Condition mask.
        x : numpy.ndarray
            Values selected where *cond* is True.
        y : numpy.ndarray
            Values selected where *cond* is False.

        Returns
        -------
        numpy.ndarray
            Array with values from *x* where *cond* is True, else from *y*.
        """
        return np.where(cond, x, y)

    def minimum(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Elementwise minimum of *x* and *y*.

        Parameters
        ----------
        x : numpy.ndarray
            First input array.
        y : numpy.ndarray
            Second input array.

        Returns
        -------
        numpy.ndarray
            Elementwise minimum.
        """
        return np.minimum(x, y)

    def maximum(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Elementwise maximum of *x* and *y*.

        Parameters
        ----------
        x : numpy.ndarray
            First input array.
        y : numpy.ndarray
            Second input array.

        Returns
        -------
        numpy.ndarray
            Elementwise maximum.
        """
        return np.maximum(x, y)

    def isfinite(self, x: np.ndarray) -> np.ndarray:
        """Elementwise test for finite values.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray of bool
            True where *x* is finite (not NaN and not Inf).
        """
        return np.isfinite(x)

    def isnan(self, x: np.ndarray) -> np.ndarray:
        """Elementwise test for NaN.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray of bool
            True where *x* is NaN.
        """
        return np.isnan(x)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Sum of array elements over the given axis.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.
        axis : int, optional
            Axis along which to sum. When None, sums all elements.

        Returns
        -------
        numpy.ndarray or scalar
            Sum of *x*.
        """
        return np.sum(x, axis=axis)

    def mean(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Mean of array elements over the given axis.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.
        axis : int, optional
            Axis along which to average. When None, averages all elements.

        Returns
        -------
        numpy.ndarray or scalar
            Mean of *x*.
        """
        return np.mean(x, axis=axis)

    def max(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Maximum of array elements over the given axis.

        Parameters
        ----------
        x : numpy.ndarray
            Input array.
        axis : int, optional
            Axis along which to find the maximum. When None, returns the
            global maximum.

        Returns
        -------
        numpy.ndarray or scalar
            Maximum value(s) of *x*.
        """
        return np.max(x, axis=axis)

    # ------------------------------------------------------------------
    # Array manipulation
    # ------------------------------------------------------------------

    def stack(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Stack a sequence of arrays along a new axis.

        Parameters
        ----------
        arrays : list of numpy.ndarray
            Arrays to stack. All must have the same shape.
        axis : int, optional
            Position in the result array where the new axis is inserted.
            Default is 0.

        Returns
        -------
        numpy.ndarray
            Stacked array with one more dimension than the inputs.
        """
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenate arrays along an existing axis.

        Parameters
        ----------
        arrays : list of numpy.ndarray
            Arrays to concatenate. All must have the same shape except
            along *axis*.
        axis : int, optional
            Axis along which to concatenate. Default is 0.

        Returns
        -------
        numpy.ndarray
            Concatenated array.
        """
        return np.concatenate(arrays, axis=axis)

    # ------------------------------------------------------------------
    # Function transforms
    # ------------------------------------------------------------------

    def jit(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """Return *f* unchanged (NumPy does not support JIT compilation).

        Parameters
        ----------
        f : callable
            Function to (notionally) compile.

        Returns
        -------
        callable
            *f* unchanged.
        """
        return f

    def vmap(
        self,
        f: Callable[..., Any],
        in_axes: int | tuple[int | None, ...] = 0,
        out_axes: int = 0,
    ) -> Callable[..., Any]:
        """Return a vectorised version of *f* using a Python loop and ``np.stack``.

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
            Vectorised function that applies *f* element-by-element along
            the batch axis and stacks the results.
        """

        def vmapped(*args: Any) -> Any:
            """Apply *f* over a batch axis using a Python loop.

            Parameters
            ----------
            *args : Any
                Batched and un-batched arguments forwarded to *f*.

            Returns
            -------
            Any
                Stacked outputs from *f* along *out_axes*.
            """
            n_args = len(args)
            if isinstance(in_axes, (list, tuple)):
                axes = list(in_axes)
                # pad to length of args if shorter
                axes += [None] * (n_args - len(axes))
            else:
                axes = [in_axes] * n_args

            # determine batch size from first batched arg
            batch_size = None
            for a, ax in zip(args, axes, strict=False):
                if ax is not None:
                    batch_size = np.asarray(a).shape[ax]
                    break
            if batch_size is None:
                return f(*args)

            results = []
            for i in range(batch_size):
                sliced = tuple(
                    np.take(a, i, axis=ax) if ax is not None else a
                    for a, ax in zip(args, axes, strict=False)
                )
                results.append(f(*sliced))

            if isinstance(results[0], tuple):
                return tuple(
                    np.stack([r[j] for r in results], axis=out_axes)
                    for j in range(len(results[0]))
                )
            return np.stack(results, axis=out_axes)

        return vmapped

    def grad(
        self,
        f: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable[..., Any]:
        """Raise :exc:`NotDifferentiable` — NumPy does not support autodiff.

        Parameters
        ----------
        f : callable
            Function to differentiate (unused).
        argnums : int or tuple of int, optional
            Argument indices to differentiate with respect to (unused).

        Returns
        -------
        callable
            Never returns; always raises.

        Raises
        ------
        NotDifferentiable
            Always, because NumpyBackend does not support automatic
            differentiation. Use :class:`JaxBackend` instead.
        """
        raise NotDifferentiable(
            "NumpyBackend does not support automatic differentiation. "
            "Use JaxBackend for gradient computation."
        )

    # ------------------------------------------------------------------
    # Histogramming and convolution
    # ------------------------------------------------------------------

    def histogram2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: tuple[int, int],
        range: tuple[tuple[float, float], tuple[float, float]],
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the 2-D weighted histogram counts via ``numpy.histogram2d``.

        Parameters
        ----------
        x : numpy.ndarray
            First-coordinate sample values (binned along axis 0).
        y : numpy.ndarray
            Second-coordinate sample values (binned along axis 1).
        bins : tuple of int
            ``(nx, ny)`` bin counts.
        range : tuple of tuple of float
            ``((xmin, xmax), (ymin, ymax))`` histogram extent.
        weights : numpy.ndarray, optional
            Per-sample weights. None → unit weights.

        Returns
        -------
        numpy.ndarray
            2-D histogram counts of shape *bins*.
        """
        w = None if weights is None else np.asarray(weights)
        h, _, _ = np.histogram2d(
            np.asarray(x),
            np.asarray(y),
            bins=bins,
            range=range,
            weights=w,
        )
        return h

    def fftconvolve(
        self, a: np.ndarray, b: np.ndarray, mode: str = "same"
    ) -> np.ndarray:
        """FFT-based convolution via ``scipy.signal.fftconvolve``.

        Parameters
        ----------
        a : numpy.ndarray
            First input array.
        b : numpy.ndarray
            Convolution kernel.
        mode : str, optional
            ``"full"``, ``"same"``, or ``"valid"``. Default ``"same"``.

        Returns
        -------
        numpy.ndarray
            Convolution of *a* and *b*.
        """
        from scipy.signal import fftconvolve as _fftconvolve

        return _fftconvolve(np.asarray(a), np.asarray(b), mode=mode)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert a backend array to a NumPy ndarray.

        Parameters
        ----------
        x : array_like
            Array to convert.

        Returns
        -------
        numpy.ndarray
            NumPy view or copy of *x*.
        """
        return np.asarray(x)
