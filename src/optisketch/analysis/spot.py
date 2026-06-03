"""Spot-diagram statistics for a traced ray bundle (Phase 4).

:func:`spot` reduces a :class:`~optisketch.rays.Rays` bundle to centroid,
RMS, and geometric-radius statistics over the valid rays. All reductions are
NaN-safe: invalid rays are masked out via ``backend.where`` before any sum or
maximum, so a NaN in a missed/TIR ray never corrupts the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from optisketch.backends._protocol import Backend
    from optisketch.rays import Rays


class SpotStats(NamedTuple):
    """Summary statistics of a spot diagram.

    Attributes
    ----------
    cx : float
        x-coordinate of the centroid of valid ray hits (mm).
    cy : float
        y-coordinate of the centroid of valid ray hits (mm).
    rms : float
        Root-mean-square spot radius about the reference point (mm).
    geo_radius : float
        Maximum distance of any valid ray from the reference point (mm).
    n_valid : int
        Number of valid rays contributing to the statistics.
    """

    cx: float
    cy: float
    rms: float
    geo_radius: float
    n_valid: int


def _default_backend() -> Backend:
    """Return a NumpyBackend instance without importing it at module load.

    Returns
    -------
    Backend
        A fresh :class:`~optisketch.backends.NumpyBackend`.
    """
    from optisketch.backends._numpy import NumpyBackend

    return NumpyBackend()


def spot(
    rays: Rays,
    *,
    reference: str = "centroid",
    backend: Backend | None = None,
) -> SpotStats:
    """Compute spot-diagram statistics for a ray bundle.

    Statistics are evaluated on the final ``(x, y)`` positions of the rays,
    masked on ``rays.valid``. Two reference modes are supported: ``"centroid"``
    measures RMS/geometric radius about the centroid of the valid hits, while
    ``"chief"`` measures them about the chief ray (ray index 0).

    Parameters
    ----------
    rays : Rays
        Ray bundle, typically the output of a trace at the image plane.
    reference : str, optional
        ``"centroid"`` (default) or ``"chief"``. Selects the reference point
        for the RMS and geometric-radius statistics.
    backend : Backend, optional
        Array-computation backend. Defaults to
        :class:`~optisketch.backends.NumpyBackend`.

    Returns
    -------
    SpotStats
        Centroid, RMS radius, geometric radius, and valid-ray count.

    Raises
    ------
    ValueError
        If *reference* is not ``"centroid"`` or ``"chief"``.
    """
    if reference not in ("centroid", "chief"):
        raise ValueError(
            f"Unknown reference {reference!r}. Choose 'centroid' or 'chief'."
        )

    be = backend if backend is not None else _default_backend()

    valid = rays.valid
    zeros = be.zeros_like(rays.x)
    # mask positions: invalid rays contribute 0 (never NaN) to the sums
    xv = be.where(valid, rays.x, zeros)
    yv = be.where(valid, rays.y, zeros)
    vf = be.where(valid, be.ones_like(rays.x), zeros)

    n = be.sum(vf)
    n_safe = be.maximum(n, be.ones_like(n))
    cx = be.sum(xv) / n_safe
    cy = be.sum(yv) / n_safe

    if reference == "chief":
        rx = rays.x[0]
        ry = rays.y[0]
    else:
        rx = cx
        ry = cy

    dx = be.where(valid, rays.x - rx, zeros)
    dy = be.where(valid, rays.y - ry, zeros)
    r2 = dx * dx + dy * dy

    rms = be.sqrt(be.sum(r2) / n_safe)
    geo = be.sqrt(be.max(be.where(valid, r2, zeros)))

    to_np = be.to_numpy
    return SpotStats(
        cx=float(to_np(cx)),
        cy=float(to_np(cy)),
        rms=float(to_np(rms)),
        geo_radius=float(to_np(geo)),
        n_valid=int(to_np(n)),
    )
