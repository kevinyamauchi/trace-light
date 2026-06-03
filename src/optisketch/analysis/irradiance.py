"""Irradiance (flux density) estimation at a detector plane (Phase 4).

:func:`irradiance` traces a :class:`~optisketch.sources.Source` through a
system, propagates the exiting rays to a target plane, and accumulates a
weighted 2-D histogram of the valid hits. It is a thin wrapper over
:func:`~optisketch.kernels._trace_surfaces` and the backend ``histogram2d``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.sources import emit

if TYPE_CHECKING:
    from optisketch.rays import System
    from optisketch.sources import Source


def irradiance(
    system: System,
    source: Source,
    z: float,
    grid: tuple[int, int] = (64, 64),
    *,
    extent: float | tuple[float, float] | None = None,
) -> Any:
    """Compute the irradiance distribution of *source* at plane ``z``.

    The source is emitted into the system, traced through every surface, and
    propagated in free space to ``z``. Valid hits are binned into a weighted
    2-D histogram (weights are the per-ray intensities). When *extent* is None
    the window is sized to enclose all valid hits, so the histogram sum equals
    the total weight of the valid rays.

    Parameters
    ----------
    system : System
        Optical system to trace through.
    source : Source
        Ray source to emit.
    z : float
        Detector-plane z-position (mm).
    grid : tuple of int, optional
        ``(ny, nx)`` histogram grid shape. Default ``(64, 64)``.
    extent : float or tuple of float, optional
        Half-width of the window about the origin (mm). Scalar applies to both
        axes; ``(ey, ex)`` sets them separately. When None it is derived from
        the data.

    Returns
    -------
    array
        2-D irradiance histogram of shape *grid*, indexed ``[iy, ix]``.
    """
    be = system.backend
    rays = emit(source, system)
    final, _ = _trace_surfaces(rays, system.structure, system.params, be)
    final = _propagate_to_plane(final, float(z), be)

    xs, ys, valid = final.x, final.y, final.valid
    zeros = be.zeros_like(xs)
    wv = be.where(valid, final.i, be.zeros_like(final.i))

    if extent is None:
        ax = float(be.to_numpy(be.max(be.where(valid, be.abs(xs), zeros))))
        ay = float(be.to_numpy(be.max(be.where(valid, be.abs(ys), zeros))))
        ex = ax * 1.0 + 1e-6
        ey = ay * 1.0 + 1e-6
    elif isinstance(extent, (tuple, list)):
        ey, ex = float(extent[0]), float(extent[1])
    else:
        ex = ey = float(extent)

    xs_safe = be.where(valid, xs, zeros)
    ys_safe = be.where(valid, ys, zeros)
    rng = ((-ey, ey), (-ex, ex))
    return be.histogram2d(ys_safe, xs_safe, bins=grid, range=rng, weights=wv)
