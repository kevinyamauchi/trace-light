"""Differentiable parameter optimization (Phase 6).

Two routines are provided:

* :func:`best_focus` — find the detector plane that minimises the on-axis spot
  variance by gradient descent over the image-plane position.
* :func:`minimize` — general gradient-descent optimisation of a chosen traced
  parameter field (radii, conics, indices) against a user objective.

Both require a differentiable backend (:class:`~optisketch.backends.JaxBackend`).
On the NumPy backend they raise :exc:`~optisketch.backends.NotDifferentiable`,
consistent with ``backend.is_differentiable == False``.

.. note::

   The DESIGN §3 autofocus reference values (408.000 mm / 90.93 µm →
   405.874 mm / 11.81 µm) pin a specific lens defined in ``DESIGN.md``, which is
   not vendored in this repository. The implementation is therefore validated by
   the convergence and gradient-consistency properties of the optimiser rather
   than against that absolute literal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optisketch.backends._protocol import NotDifferentiable
from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.sources import emit, point_source

if TYPE_CHECKING:
    from collections.abc import Callable

    from optisketch.rays import System
    from optisketch.sources import Source


def _spot_variance(rays: Any, be: Any) -> Any:
    """Return the valid-ray spot variance about the centroid (scalar).

    Parameters
    ----------
    rays : Rays
        Ray bundle to evaluate.
    be : Backend
        Array-computation backend.

    Returns
    -------
    array
        Scalar spot variance ``mean(|r - r_centroid|²)`` over valid rays.
    """
    valid = rays.valid
    zeros = be.zeros_like(rays.x)
    vf = be.where(valid, be.ones_like(rays.x), zeros)
    total = be.sum(vf)
    n = be.maximum(total, be.ones_like(total))
    xv = be.where(valid, rays.x, zeros)
    yv = be.where(valid, rays.y, zeros)
    cx = be.sum(xv) / n
    cy = be.sum(yv) / n
    dx = be.where(valid, rays.x - cx, zeros)
    dy = be.where(valid, rays.y - cy, zeros)
    return be.sum(dx * dx + dy * dy) / n


def best_focus(
    system: System,
    source: Source | None = None,
    *,
    lr: float | None = None,
    n_steps: int = 50,
) -> float:
    """Find the detector z-position that minimises on-axis spot variance.

    The rays are traced once through the system; the only free variable is the
    detector plane *z*, to which the exit rays are propagated. Spot variance is
    a smooth (quadratic, in the paraxial limit) function of *z*, so plain
    gradient descent converges to the best-focus plane.

    Parameters
    ----------
    system : System
        Optical system to focus. Must be bound to a differentiable backend.
    source : Source, optional
        Source to trace. Defaults to an on-axis point source one nominal object
        distance in front of the pupil.
    lr : float, optional
        Gradient-descent step scaling. When None (default) the step is scaled
        by the inverse curvature of the (quadratic) variance-vs-z objective,
        giving Newton-like convergence in a handful of steps.
    n_steps : int, optional
        Number of descent steps. Default 50.

    Returns
    -------
    float
        The optimised detector z-position (mm).

    Raises
    ------
    NotDifferentiable
        If ``system.backend`` does not support automatic differentiation.
    """
    be = system.backend
    if not be.is_differentiable:
        raise NotDifferentiable(
            "best_focus requires a differentiable backend (JaxBackend). "
            "The NumpyBackend cannot compute gradients."
        )

    if source is None:
        source = point_source((0.0, 0.0), z_object=-100.0, n_samples=64)

    rays = emit(source, system)
    traced, _ = _trace_surfaces(rays, system.structure, system.params, be)

    def objective(z: Any) -> Any:
        """Spot variance at detector plane *z*.

        Parameters
        ----------
        z : array
            Detector plane z-position (mm).

        Returns
        -------
        array
            Scalar spot variance.
        """
        at_plane = _propagate_to_plane(traced, z, be)
        return _spot_variance(at_plane, be)

    grad = be.grad(objective)
    z0 = be.asarray(float(system.image_z))

    if lr is None:
        # variance is quadratic in z → constant curvature; step = grad / hess
        hess = be.grad(grad)(z0)
        hess_np = float(be.to_numpy(hess))
        step = 1.0 / hess_np if hess_np > 0.0 else 1.0
    else:
        step = float(lr)

    z = z0
    for _ in range(n_steps):
        z = z - step * grad(z)
    return float(be.to_numpy(z))


def minimize(
    system: System,
    objective: Callable[[System], Any],
    param: str = "radii",
    *,
    lr: float = 1e-3,
    n_steps: int = 100,
) -> System:
    """Gradient-descent optimisation of one traced parameter field.

    The named field of ``system.params`` (one of ``"radii"``, ``"conics"``,
    ``"n1"``, ``"n2"``) is treated as the free variable. At each step the
    objective is evaluated on a system rebuilt with the current parameter
    values, and the field is updated along the negative gradient.

    Parameters
    ----------
    system : System
        Starting system. Must be bound to a differentiable backend.
    objective : callable
        Function mapping a :class:`System` to a scalar loss to minimise.
    param : str, optional
        Name of the ``_Params`` field to optimise: one of ``"z"`` (spacings),
        ``"radii"``, ``"conics"``, ``"n1"``, ``"n2"``, or ``"semi_aperture"``.
        Default ``"radii"``. Because ``z`` and ``semi_aperture`` are traced
        (DESIGN §4.2), spacings and apertures are optimisable here too.
    lr : float, optional
        Learning rate. Default ``1e-3``.
    n_steps : int, optional
        Number of descent steps. Default 100.

    Returns
    -------
    System
        A new system with the optimised parameter field.

    Raises
    ------
    NotDifferentiable
        If ``system.backend`` does not support automatic differentiation.
    """
    be = system.backend
    if not be.is_differentiable:
        raise NotDifferentiable(
            "minimize requires a differentiable backend (JaxBackend). "
            "The NumpyBackend cannot compute gradients."
        )

    x = getattr(system.params, param)

    def loss(values: Any) -> Any:
        """Objective evaluated on a system with *values* in the chosen field.

        Parameters
        ----------
        values : array
            Candidate values for the optimised parameter field.

        Returns
        -------
        array
            Scalar objective value.
        """
        new_params = system.params._replace(**{param: values})
        candidate = system._replace(params=new_params)
        return objective(candidate)

    grad = be.grad(loss)
    for _ in range(n_steps):
        x = x - lr * grad(x)

    final_params = system.params._replace(**{param: x})
    return system._replace(params=final_params)
