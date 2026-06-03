"""Point-spread-function estimation by geometric ray histogramming (Phase 4).

:func:`psf` traces a point emitter through a :class:`~trace_light.rays.System`,
propagates the exiting rays to a (possibly defocused) detector plane, and
histograms the valid image-plane hits into a normalised 2-D kernel centred on
the spot centroid. Because the kernel is centred on the centroid, the result is
a shift-invariant impulse response suitable for convolution in
:func:`~trace_light.analysis.image_sim.image_sim`.

Sweeping ``depth`` (object-side axial position) or ``focus`` (detector-side
axial position) produces a through-focus stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trace_light.kernels import _propagate_to_plane, _trace_surfaces
from trace_light.sources import emit, point_source

if TYPE_CHECKING:
    from trace_light.rays import System


def _masked_centroid(xs: Any, ys: Any, valid: Any, be: Any) -> tuple[float, float, int]:
    """Return the NaN-safe centroid of valid samples as Python floats.

    Parameters
    ----------
    xs : array
        x-coordinates of the samples.
    ys : array
        y-coordinates of the samples.
    valid : bool array
        Mask selecting the samples that contribute.
    be : Backend
        Array-computation backend.

    Returns
    -------
    cx : float
        Centroid x-coordinate (0.0 if no valid samples).
    cy : float
        Centroid y-coordinate.
    n : int
        Number of valid samples.
    """
    zeros = be.zeros_like(xs)
    vf = be.where(valid, be.ones_like(xs), zeros)
    n = float(be.to_numpy(be.sum(vf)))
    if n < 1.0:
        return 0.0, 0.0, 0
    cx = float(be.to_numpy(be.sum(be.where(valid, xs, zeros)))) / n
    cy = float(be.to_numpy(be.sum(be.where(valid, ys, zeros)))) / n
    return cx, cy, int(n)


def psf(
    system: System,
    field: tuple[float, float] = (0.0, 0.0),
    *,
    depth: float = 0.0,
    focus: float = 0.0,
    wavelength: float | None = None,
    n_rays: int = 256,
    grid: tuple[int, int] = (64, 64),
    extent: float | None = None,
    z_object: float = -100.0,
) -> Any:
    """Estimate the geometric PSF kernel of *system* for a point emitter.

    A point source at lateral position *field* and axial position
    ``z_object + depth`` is traced through the system; the exiting rays are
    propagated to the detector plane ``image_z + focus`` and histogrammed into
    a grid centred on the spot centroid. The kernel is normalised to unit sum.

    Parameters
    ----------
    system : System
        Optical system to evaluate.
    field : tuple of float, optional
        ``(x, y)`` lateral position of the point emitter (mm). Default on-axis.
    depth : float, optional
        Axial offset of the emitter from the nominal object plane (mm).
        Default 0.
    focus : float, optional
        Axial offset of the detector plane from ``system.image_z`` (mm).
        Default 0.
    wavelength : float, optional
        Emission wavelength (µm). Defaults to the system's first wavelength.
    n_rays : int, optional
        Number of pupil samples (rays). Default 256.
    grid : tuple of int, optional
        ``(ny, nx)`` kernel grid shape. Default ``(64, 64)``.
    extent : float, optional
        Half-width of the kernel window about the centroid (mm). When None it
        is derived from the geometric spread of the valid rays.
    z_object : float, optional
        Nominal object-plane z-position (mm). Default ``-100.0``.

    Returns
    -------
    array
        Normalised 2-D PSF kernel of shape *grid*, indexed ``[iy, ix]``.
    """
    be = system.backend
    wl = float(system.wavelengths[0]) if wavelength is None else float(wavelength)

    src = point_source(
        (float(field[0]), float(field[1])),
        z_object=float(z_object) + float(depth),
        wavelength=wl,
        pupil_pattern="disk",
        n_samples=n_rays,
    )
    rays = emit(src, system)
    final, _ = _trace_surfaces(rays, system.structure, system.params, be)

    z_det = float(system.image_z) + float(focus)
    final = _propagate_to_plane(final, z_det, be)

    xs, ys, valid = final.x, final.y, final.valid
    cx, cy, _n = _masked_centroid(xs, ys, valid, be)

    if extent is None:
        zeros = be.zeros_like(xs)
        dx = be.where(valid, xs - cx, zeros)
        dy = be.where(valid, ys - cy, zeros)
        r2 = dx * dx + dy * dy
        spread = float(be.to_numpy(be.sqrt(be.max(be.where(valid, r2, zeros)))))
        extent = spread * 1.1 + 1e-6
    e = float(extent)

    # weights: valid-ray intensity, zero elsewhere; keep invalid coords in-range
    wv = be.where(valid, final.i, be.zeros_like(final.i))
    xs_safe = be.where(valid, xs, be.full_like(xs, cx))
    ys_safe = be.where(valid, ys, be.full_like(ys, cy))

    rng = ((cy - e, cy + e), (cx - e, cx + e))
    h = be.histogram2d(ys_safe, xs_safe, bins=grid, range=rng, weights=wv)

    total = be.sum(h)
    total_safe = be.where(total > 0.0, total, be.ones_like(total))
    return h / total_safe
