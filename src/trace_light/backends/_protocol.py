"""Backend protocol and shared exception."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class NotDifferentiable(Exception):
    """Raised when gradient computation is requested on a non-differentiable backend."""


class Backend(ABC):
    """Abstract base class for array-computation backends.

    All kernels receive an explicit backend instance; there is no global state.

    Attributes
    ----------
    name : str
        Human-readable backend identifier (e.g. ``"numpy"``, ``"jax"``).
    is_differentiable : bool
        True if :meth:`grad` is supported.
    supports_jit : bool
        True if :meth:`jit` produces a compiled function.
    """

    # Subclasses set these as class attributes.
    name: str
    is_differentiable: bool
    supports_jit: bool

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    @abstractmethod
    def array(self, x: Any, dtype: Any = None) -> Any:
        """Convert *x* to a backend array.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype. When None the backend chooses a default
            (typically float64).

        Returns
        -------
        array
            Backend array containing the data from *x*.
        """

    @abstractmethod
    def asarray(self, x: Any, dtype: Any = None) -> Any:
        """Convert *x* to a backend array, avoiding a copy when possible.

        Parameters
        ----------
        x : array_like
            Input data.
        dtype : dtype, optional
            Desired output dtype.

        Returns
        -------
        array
            Backend array. May share memory with *x* if *x* is already
            the correct type and dtype.
        """

    @abstractmethod
    def zeros(self, shape: int | tuple[int, ...], dtype: Any = None) -> Any:
        """Return an array of zeros with the given shape.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the output array.
        dtype : dtype, optional
            Desired dtype. Defaults to float64.

        Returns
        -------
        array
            Zero-filled array of the requested shape and dtype.
        """

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        """Return a zero array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : array
            Reference array.

        Returns
        -------
        array
            Zero-filled array matching the shape and dtype of *x*.
        """

    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        """Return a ones array with the same shape and dtype as *x*.

        Parameters
        ----------
        x : array
            Reference array.

        Returns
        -------
        array
            One-filled array matching the shape and dtype of *x*.
        """

    @abstractmethod
    def full_like(self, x: Any, fill_value: Any) -> Any:
        """Return an array filled with *fill_value*, shaped like *x*.

        Parameters
        ----------
        x : array
            Reference array.
        fill_value : scalar
            Value to fill the output array with.

        Returns
        -------
        array
            Array matching the shape and dtype of *x*, filled with
            *fill_value*.
        """

    @abstractmethod
    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
    ) -> Any:
        """Return an array of the given shape filled with *fill_value*.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the output array.
        fill_value : scalar
            Value to fill the output array with.
        dtype : dtype, optional
            Desired dtype. Defaults to float64.

        Returns
        -------
        array
            Array of the requested shape filled with *fill_value*.
        """

    @abstractmethod
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
        array
            1-D array of *num* evenly spaced floats.
        """

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """Elementwise square root.

        Parameters
        ----------
        x : array
            Input array.

        Returns
        -------
        array
            Elementwise square root of *x*.
        """

    @abstractmethod
    def sin(self, x: Any) -> Any:
        """Elementwise sine (input in radians).

        Parameters
        ----------
        x : array
            Input array (radians).

        Returns
        -------
        array
            Elementwise sine of *x*.
        """

    @abstractmethod
    def cos(self, x: Any) -> Any:
        """Elementwise cosine (input in radians).

        Parameters
        ----------
        x : array
            Input array (radians).

        Returns
        -------
        array
            Elementwise cosine of *x*.
        """

    @abstractmethod
    def abs(self, x: Any) -> Any:
        """Elementwise absolute value.

        Parameters
        ----------
        x : array
            Input array.

        Returns
        -------
        array
            Elementwise absolute value of *x*.
        """

    @abstractmethod
    def sign(self, x: Any) -> Any:
        """Elementwise sign: -1, 0, or +1.

        Parameters
        ----------
        x : array
            Input array.

        Returns
        -------
        array
            Array with values in ``{-1, 0, +1}``.
        """

    @abstractmethod
    def where(self, cond: Any, x: Any, y: Any) -> Any:
        """Elementwise conditional selection.

        Parameters
        ----------
        cond : bool array
            Condition mask.
        x : array
            Values selected where *cond* is True.
        y : array
            Values selected where *cond* is False.

        Returns
        -------
        array
            Array with values from *x* where *cond* is True, else from *y*.
        """

    @abstractmethod
    def minimum(self, x: Any, y: Any) -> Any:
        """Elementwise minimum of *x* and *y*.

        Parameters
        ----------
        x : array
            First input array.
        y : array
            Second input array.

        Returns
        -------
        array
            Elementwise minimum.
        """

    @abstractmethod
    def maximum(self, x: Any, y: Any) -> Any:
        """Elementwise maximum of *x* and *y*.

        Parameters
        ----------
        x : array
            First input array.
        y : array
            Second input array.

        Returns
        -------
        array
            Elementwise maximum.
        """

    @abstractmethod
    def isfinite(self, x: Any) -> Any:
        """Elementwise test for finite values.

        Parameters
        ----------
        x : array
            Input array.

        Returns
        -------
        bool array
            True where *x* is finite (not NaN and not Inf).
        """

    @abstractmethod
    def isnan(self, x: Any) -> Any:
        """Elementwise test for NaN.

        Parameters
        ----------
        x : array
            Input array.

        Returns
        -------
        bool array
            True where *x* is NaN.
        """

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    @abstractmethod
    def sum(self, x: Any, axis: int | None = None) -> Any:
        """Sum of array elements over the given axis.

        Parameters
        ----------
        x : array
            Input array.
        axis : int, optional
            Axis along which to sum. When None, sums all elements.

        Returns
        -------
        array or scalar
            Sum of *x*.
        """

    @abstractmethod
    def mean(self, x: Any, axis: int | None = None) -> Any:
        """Mean of array elements over the given axis.

        Parameters
        ----------
        x : array
            Input array.
        axis : int, optional
            Axis along which to average. When None, averages all elements.

        Returns
        -------
        array or scalar
            Mean of *x*.
        """

    @abstractmethod
    def max(self, x: Any, axis: int | None = None) -> Any:
        """Maximum of array elements over the given axis.

        Parameters
        ----------
        x : array
            Input array.
        axis : int, optional
            Axis along which to find the maximum. When None, returns the
            global maximum.

        Returns
        -------
        array or scalar
            Maximum value(s) of *x*.
        """

    # ------------------------------------------------------------------
    # Array manipulation
    # ------------------------------------------------------------------

    @abstractmethod
    def stack(self, arrays: list[Any], axis: int = 0) -> Any:
        """Stack a sequence of arrays along a new axis.

        Parameters
        ----------
        arrays : list of array
            Arrays to stack. All must have the same shape.
        axis : int, optional
            Position in the result array where the new axis is inserted.
            Default is 0.

        Returns
        -------
        array
            Stacked array with one more dimension than the inputs.
        """

    @abstractmethod
    def concatenate(self, arrays: list[Any], axis: int = 0) -> Any:
        """Concatenate arrays along an existing axis.

        Parameters
        ----------
        arrays : list of array
            Arrays to concatenate. All must have the same shape except
            along *axis*.
        axis : int, optional
            Axis along which to concatenate. Default is 0.

        Returns
        -------
        array
            Concatenated array.
        """

    # ------------------------------------------------------------------
    # Function transforms
    # ------------------------------------------------------------------

    @abstractmethod
    def jit(self, f: Callable[..., Any]) -> Callable[..., Any]:
        """JIT-compile a function (identity on non-JIT backends).

        Parameters
        ----------
        f : callable
            Function to compile.

        Returns
        -------
        callable
            Compiled function, or *f* unchanged on backends that do not
            support JIT.
        """

    @abstractmethod
    def vmap(
        self,
        f: Callable[..., Any],
        in_axes: int | tuple[int | None, ...] = 0,
        out_axes: int = 0,
    ) -> Callable[..., Any]:
        """Vectorise *f* over a batch axis.

        Parameters
        ----------
        f : callable
            Function to vectorise. Must accept and return arrays.
        in_axes : int or tuple of int or None, optional
            Specifies which axis of each argument to batch over. An
            element of None means that argument is not batched. Default
            is 0 (batch first axis of every argument).
        out_axes : int, optional
            Axis position in the output where the batch dimension is
            placed. Default is 0.

        Returns
        -------
        callable
            Vectorised function.
        """

    @abstractmethod
    def grad(
        self,
        f: Callable[..., Any],
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable[..., Any]:
        """Return the gradient function of *f*.

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
            Function that computes the gradient of *f*.

        Raises
        ------
        NotDifferentiable
            If the backend does not support automatic differentiation.
        """

    # ------------------------------------------------------------------
    # Histogramming and convolution (analysis primitives)
    # ------------------------------------------------------------------

    @abstractmethod
    def histogram2d(
        self,
        x: Any,
        y: Any,
        bins: tuple[int, int],
        range: tuple[tuple[float, float], tuple[float, float]],
        weights: Any = None,
    ) -> Any:
        """Return the 2-D weighted histogram of paired samples.

        Semantics match :func:`numpy.histogram2d` (only the counts array is
        returned, not the bin edges). The output is indexed ``H[i, j]`` where
        *i* runs over the *x* binning and *j* over the *y* binning.

        Parameters
        ----------
        x : array
            First-coordinate sample values (binned along axis 0).
        y : array
            Second-coordinate sample values (binned along axis 1).
        bins : tuple of int
            ``(nx, ny)`` bin counts for each axis.
        range : tuple of tuple of float
            ``((xmin, xmax), (ymin, ymax))`` histogram extent.
        weights : array, optional
            Per-sample weights. When None each sample contributes 1.

        Returns
        -------
        array
            2-D histogram counts of shape ``bins``.
        """

    @abstractmethod
    def fftconvolve(self, a: Any, b: Any, mode: str = "same") -> Any:
        """FFT-based convolution of two N-D arrays.

        Semantics match :func:`scipy.signal.fftconvolve`.

        Parameters
        ----------
        a : array
            First input array.
        b : array
            Second input array (the convolution kernel).
        mode : str, optional
            Output size mode: ``"full"``, ``"same"``, or ``"valid"``.
            Default ``"same"``.

        Returns
        -------
        array
            Convolution of *a* and *b*.
        """

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        """Convert a backend array to a NumPy ndarray.

        Parameters
        ----------
        x : array
            Backend array to convert.

        Returns
        -------
        numpy.ndarray
            NumPy view or copy of *x*.
        """
