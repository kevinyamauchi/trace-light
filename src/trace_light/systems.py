"""Prefab optical systems and SystemBuilder — Phase 2."""

from __future__ import annotations

from typing import Any


class SystemBuilder:
    """Builder for assembling multi-surface optical systems — Phase 2.

    Raises
    ------
    NotImplementedError
        Always on construction. This class will be implemented in Phase 2.
    """

    def __init__(self) -> None:
        """Initialise the SystemBuilder.

        Raises
        ------
        NotImplementedError
            Always. This class will be implemented in Phase 2.
        """
        raise NotImplementedError("SystemBuilder is Phase 2")


def four_f(*args: Any, **kwargs: Any) -> None:
    """Create a 4-f relay optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 2.
    """
    raise NotImplementedError("systems module is Phase 2")


def microscope(*args: Any, **kwargs: Any) -> None:
    """Create a simple microscope optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 2.
    """
    raise NotImplementedError("systems module is Phase 2")


def relay(*args: Any, **kwargs: Any) -> None:
    """Create a relay lens optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 2.
    """
    raise NotImplementedError("systems module is Phase 2")


def telescope(*args: Any, **kwargs: Any) -> None:
    """Create a telescope optical system.

    Parameters
    ----------
    *args : Any
        Positional arguments (API not yet defined).
    **kwargs : Any
        Keyword arguments (API not yet defined).

    Raises
    ------
    NotImplementedError
        Always. This function will be implemented in Phase 2.
    """
    raise NotImplementedError("systems module is Phase 2")
