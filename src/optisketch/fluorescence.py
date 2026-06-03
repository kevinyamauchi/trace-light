"""Fluorescence imaging as a composition of existing analysis pieces (Phase 7).

This module adds **no new core ray-tracing code**. It documents and wraps the
two-pass fluorescence pipeline of DESIGN §9 as a thin composition:

1. *Excitation* (optional): an excitation irradiance ``I_exc`` over the sample
   volume, from :func:`~optisketch.analysis.irradiance` or a uniform constant
   for widefield/Köhler illumination.
2. *Emission volume*: the incoherent source ``emission = I_exc * fluorophore``
   (one-photon) or ``I_exc**2 * fluorophore`` (two-photon).
3. *Collection*: imaging of ``emission`` at the **emission** wavelength via
   :func:`~optisketch.analysis.image_sim`.

The §9.4 modality table (widefield / confocal / two-photon / light-sheet / 3-D)
is realised as excitation-side variations layered on this same pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optisketch.analysis.image_sim import image_sim

if TYPE_CHECKING:
    from optisketch.rays import System


def emission_volume(
    fluorophore: Any,
    excitation: Any = 1.0,
    *,
    two_photon: bool = False,
) -> Any:
    """Build the incoherent emission volume from fluorophore and excitation.

    Parameters
    ----------
    fluorophore : array
        Fluorophore concentration map, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
    excitation : array or float, optional
        Excitation irradiance ``I_exc`` broadcast against *fluorophore*. Default
        ``1.0`` (uniform widefield illumination).
    two_photon : bool, optional
        When True the emission scales as ``I_exc**2`` (two-photon absorption);
        otherwise linearly with ``I_exc``. Default False.

    Returns
    -------
    array
        Emission intensity volume, same shape as *fluorophore*.
    """
    exc = excitation**2 if two_photon else excitation
    return fluorophore * exc


def widefield(
    collection_system: System,
    fluorophore: Any,
    extent: float | tuple[float, float],
    *,
    excitation: Any = 1.0,
    focus: float | Any = 0.0,
    wavelength_em: float | None = None,
    psf: str = "varying",
    grid: tuple[int, int] = (3, 3),
    psf_grid: tuple[int, int] = (31, 31),
    n_rays: int = 256,
    z_object: float = -100.0,
    depth_extent: float = 0.0,
) -> Any:
    """Simulate a widefield fluorescence image (uniform excitation).

    This is the canonical composition: build ``emission = excitation *
    fluorophore`` then image it at the emission wavelength.

    Parameters
    ----------
    collection_system : System
        Imaging (collection) optical system.
    fluorophore : array
        Fluorophore map, shape ``(ny, nx)`` or ``(nz, ny, nx)``.
    extent : float or tuple of float
        Lateral half-width of the field of view (mm).
    excitation : array or float, optional
        Excitation irradiance. Default ``1.0`` (uniform widefield).
    focus : float or array, optional
        Detector focus offset(s) from ``collection_system.image_z`` (mm).
    wavelength_em : float, optional
        Emission wavelength (µm). Defaults to the system's first wavelength.
    psf : str, optional
        PSF mode forwarded to :func:`~optisketch.analysis.image_sim`.
    grid : tuple of int, optional
        Coarse field grid for the varying PSF.
    psf_grid : tuple of int, optional
        Per-PSF kernel grid.
    n_rays : int, optional
        Pupil samples per PSF evaluation.
    z_object : float, optional
        Nominal object-plane z-position (mm).
    depth_extent : float, optional
        Half-range of object axial depth mapped across the slices (mm).

    Returns
    -------
    array
        Simulated fluorescence image (2-D) or focal stack (3-D).
    """
    emission = emission_volume(fluorophore, excitation, two_photon=False)
    return image_sim(
        collection_system,
        emission,
        extent,
        psf=psf,
        grid=grid,
        focus=focus,
        wavelength=wavelength_em,
        psf_grid=psf_grid,
        n_rays=n_rays,
        z_object=z_object,
        depth_extent=depth_extent,
    )
