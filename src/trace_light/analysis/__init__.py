"""Read-only analysis functions (Phases 4-5).

* :func:`spot` — spot-diagram statistics for a ray bundle.
* :func:`psf` — geometric point-spread-function kernel.
* :func:`irradiance` — flux-density histogram at a plane.
* :func:`image_sim` — incoherent image formation by PSF convolution.
"""

from __future__ import annotations

from trace_light.analysis.image_sim import image_sim
from trace_light.analysis.irradiance import irradiance
from trace_light.analysis.psf import psf
from trace_light.analysis.spot import SpotStats, spot

__all__ = [
    "SpotStats",
    "image_sim",
    "irradiance",
    "psf",
    "spot",
]
