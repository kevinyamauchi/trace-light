"""Read-only analysis functions (Phases 4-5).

* :func:`spot` — spot-diagram statistics for a ray bundle.
* :func:`psf` — geometric point-spread-function kernel.
* :func:`irradiance` — flux-density histogram at a plane.
* :func:`image_sim` — incoherent image formation by PSF convolution.
"""

from __future__ import annotations

from optisketch.analysis.image_sim import image_sim
from optisketch.analysis.irradiance import irradiance
from optisketch.analysis.psf import psf
from optisketch.analysis.spot import SpotStats, spot

__all__ = [
    "SpotStats",
    "image_sim",
    "irradiance",
    "psf",
    "spot",
]
