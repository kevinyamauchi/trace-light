"""Ray source factories and emit — Phase 3."""

from __future__ import annotations

from typing import Any


def point_source(*args: Any, **kwargs: Any) -> None:
    """Create a point light source.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 3.
    """
    raise NotImplementedError("sources module is Phase 3")


def collimated_source(*args: Any, **kwargs: Any) -> None:
    """Create a collimated (plane-wave) ray source.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 3.
    """
    raise NotImplementedError("sources module is Phase 3")


def extended_source(*args: Any, **kwargs: Any) -> None:
    """Create an extended (spatially distributed) ray source.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 3.
    """
    raise NotImplementedError("sources module is Phase 3")


def emit(*args: Any, **kwargs: Any) -> None:
    """Emit rays from a source into a ray bundle.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 3.
    """
    raise NotImplementedError("emit is Phase 3")
