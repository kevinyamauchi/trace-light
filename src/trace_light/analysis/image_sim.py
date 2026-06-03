"""Incoherent image simulation by PSF convolution (Phase 5).

:func:`image_sim` forms an image as the incoherent sum, over object depth
slices, of each slice convolved with the system PSF appropriate to its
``(field, depth, focus)``. Two modes are provided:

* ``psf="single"`` — one lateral PSF per depth (shift-invariant); each slice is
  FFT-convolved with that single kernel.
* ``psf="varying"`` — PSFs sampled on a coarse ``grid`` of field points per
  depth. Each field region is selected by a separable partition-of-unity weight
  map; the object slice is multiplied by the weight, convolved with the local
  PSF, and the contributions are summed. Because the weights sum to one
  everywhere, a system whose PSF is field-independent reproduces the
  ``psf="single"`` result exactly.

A scalar ``focus`` yields a 2-D image; an array ``focus`` yields a 3-D focal
stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trace_light.analysis.psf import psf as _psf

if TYPE_CHECKING:
    from trace_light.rays import System


def _axis_hats(n: int, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Return separable partition-of-unity hat weights along one axis.

    For *g* nodes spread evenly over ``[0, n-1]``, each pixel is assigned
    linear-interpolation weights to the nodes such that the weights sum to one
    at every pixel (a partition of unity).

    Parameters
    ----------
    n : int
        Number of pixels along the axis.
    g : int
        Number of field-sample nodes along the axis.

    Returns
    -------
    weights : numpy.ndarray
        Array of shape ``(n, g)``; column *k* is the weight of node *k*.
    nodes : numpy.ndarray
        Node pixel positions, shape ``(g,)``.
    """
    coords = np.arange(n, dtype=np.float64)
    nodes = np.linspace(0.0, n - 1, g)
    weights = np.zeros((n, g), dtype=np.float64)
    for k in range(g):
        e = np.zeros(g, dtype=np.float64)
        e[k] = 1.0
        weights[:, k] = np.interp(coords, nodes, e)
    return weights, nodes


def _pixel_to_field(node_px: float, n: int, half_extent: float) -> float:
    """Map a node pixel position to a lateral field coordinate (mm).

    Parameters
    ----------
    node_px : float
        Node position in pixel units (0 .. n-1).
    n : int
        Number of pixels along the axis.
    half_extent : float
        Half-width of the field of view along this axis (mm).

    Returns
    -------
    float
        Field coordinate in millimetres, centred on the array.
    """
    if n <= 1:
        return 0.0
    frac = (node_px - (n - 1) / 2.0) / ((n - 1) / 2.0)
    return frac * half_extent


def image_sim(
    system: System,
    obj: Any,
    extent: float | tuple[float, float],
    *,
    psf: str = "varying",
    field: tuple[float, float] = (0.0, 0.0),
    grid: tuple[int, int] = (3, 3),
    focus: float | Any = 0.0,
    wavelength: float | None = None,
    psf_grid: tuple[int, int] = (31, 31),
    n_rays: int = 256,
    z_object: float = -100.0,
    depth_extent: float = 0.0,
) -> Any:
    """Simulate the incoherent image of *obj* formed by *system*.

    Parameters
    ----------
    system : System
        Imaging system.
    obj : array
        Object intensity. Shape ``(ny, nx)`` for a planar object or
        ``(nz, ny, nx)`` for a volume (incoherently summed over depth).
    extent : float or tuple of float
        Lateral half-width of the field of view (mm). Scalar applies to both
        axes; ``(ey, ex)`` sets them separately.
    psf : str, optional
        ``"varying"`` (field-dependent, default) or ``"single"``
        (shift-invariant).
    field : tuple of float, optional
        Base lateral field offset (mm) added to every PSF evaluation.
    grid : tuple of int, optional
        ``(gy, gx)`` coarse field-sampling grid for ``psf="varying"``.
        Ignored for ``psf="single"``. Default ``(3, 3)``.
    focus : float or array, optional
        Detector focus offset(s) from ``system.image_z`` (mm). A scalar yields
        a 2-D image; a 1-D array yields a 3-D ``(nf, ny, nx)`` stack.
    wavelength : float, optional
        Imaging wavelength (µm). Defaults to the system's first wavelength.
    psf_grid : tuple of int, optional
        ``(ny, nx)`` grid of each evaluated PSF kernel. Default ``(31, 31)``.
    n_rays : int, optional
        Pupil samples per PSF evaluation. Default 256.
    z_object : float, optional
        Nominal object-plane z-position (mm). Default ``-100.0``.
    depth_extent : float, optional
        Half-range of object axial depth mapped across the ``nz`` slices (mm).
        Default 0 (all slices at the nominal plane).

    Returns
    -------
    array
        Simulated image of shape ``(ny, nx)`` (scalar *focus*) or
        ``(nf, ny, nx)`` (array *focus*).

    Raises
    ------
    ValueError
        If *psf* is not ``"single"`` or ``"varying"`` or *obj* is not 2-D/3-D.
    """
    if psf not in ("single", "varying"):
        raise ValueError(f"Unknown psf mode {psf!r}. Choose 'single' or 'varying'.")

    be = system.backend

    shape = tuple(obj.shape)
    if len(shape) == 2:
        ny, nx = shape
        slices = [obj]
        nz = 1
    elif len(shape) == 3:
        nz, ny, nx = shape
        slices = [obj[zi] for zi in range(nz)]
    else:
        raise ValueError("obj must be 2-D (ny, nx) or 3-D (nz, ny, nx).")

    if isinstance(extent, (tuple, list)):
        ext_y, ext_x = float(extent[0]), float(extent[1])
    else:
        ext_y = ext_x = float(extent)

    # The PSF kernel must share the object's pixel pitch so the discrete
    # convolution is physically meaningful. Pixel pitch = full width / pixels.
    pitch = (2.0 * ext_x) / nx
    psf_extent = pitch * psf_grid[1] / 2.0

    # focus may be scalar or a 1-D array of focal planes
    focus_arr = np.atleast_1d(np.asarray(be.to_numpy(focus), dtype=np.float64))
    scalar_focus = np.ndim(be.to_numpy(focus)) == 0

    # depth per slice
    if nz == 1:
        depths = [0.0]
    else:
        depths = list(np.linspace(-depth_extent, depth_extent, nz))

    # field sampling weight maps (varying mode only)
    if psf == "varying":
        gy, gx = grid
        wy, nodes_y = _axis_hats(ny, gy)
        wx, nodes_x = _axis_hats(nx, gx)

    images = []
    for f in focus_arr:
        acc = be.zeros((ny, nx))
        for zi in range(nz):
            depth = depths[zi]
            slc = slices[zi]
            if psf == "single":
                k = _psf(
                    system,
                    field=field,
                    depth=depth,
                    focus=float(f),
                    wavelength=wavelength,
                    n_rays=n_rays,
                    grid=psf_grid,
                    extent=psf_extent,
                    z_object=z_object,
                )
                acc = acc + be.fftconvolve(slc, k, mode="same")
            else:
                for gi in range(gy):
                    fy = field[1] + _pixel_to_field(nodes_y[gi], ny, ext_y)
                    for gj in range(gx):
                        fx = field[0] + _pixel_to_field(nodes_x[gj], nx, ext_x)
                        wmap = be.asarray(np.outer(wy[:, gi], wx[:, gj]))
                        k = _psf(
                            system,
                            field=(fx, fy),
                            depth=depth,
                            focus=float(f),
                            wavelength=wavelength,
                            n_rays=n_rays,
                            grid=psf_grid,
                            extent=psf_extent,
                            z_object=z_object,
                        )
                        acc = acc + be.fftconvolve(slc * wmap, k, mode="same")
        images.append(acc)

    if scalar_focus:
        return images[0]
    return be.stack(images, axis=0)
