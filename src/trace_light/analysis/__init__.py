"""Analysis functions (spot, psf, irradiance, image_sim) - Phases 4-5."""

from __future__ import annotations

from typing import Any


def spot(*args: Any, **kwargs: Any) -> None:
    """Compute a spot diagram for a ray bundle at an image plane.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 4.
    """
    raise NotImplementedError("analysis.spot is Phase 4")


def psf(*args: Any, **kwargs: Any) -> None:
    """Compute the point spread function for an optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 4.
    """
    raise NotImplementedError("analysis.psf is Phase 4")


def irradiance(*args: Any, **kwargs: Any) -> None:
    """Compute the irradiance distribution at a detector plane.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 4.
    """
    raise NotImplementedError("analysis.irradiance is Phase 4")


def image_sim(*args: Any, **kwargs: Any) -> None:
    """Simulate an image formed by an optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 5.
    """
    raise NotImplementedError("analysis.image_sim is Phase 5")
