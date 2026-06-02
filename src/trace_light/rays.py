"""Core data structures for rays and surface parameters.

``Rays`` is the only user-facing type in this module.  ``_Structure`` and
``_Params`` are implementation-internal and never exposed at the package
boundary.
"""

from __future__ import annotations

from typing import NamedTuple


class Rays(NamedTuple):
    """Immutable bundle of ray state arrays.

    All fields are 1-D arrays of the same length (one element per ray).
    Every kernel returns a **new** :class:`Rays` rather than mutating.

    Attributes
    ----------
    x, y, z : array
        Ray positions in the global coordinate frame (mm).
    L, M, N : array
        Direction cosines (unit vector: L²+M²+N²=1).
    i : array
        Ray weights / intensities.
    w : array
        Wavelengths (µm).
    opd : array
        Accumulated optical path (mm).
    valid : bool array
        Mask — False once a ray misses, undergoes TIR, or exits the aperture.
        Sticky: once False it stays False.
    """

    x: object
    y: object
    z: object
    L: object
    M: object
    N: object
    i: object
    w: object
    opd: object
    valid: object


class _Structure(NamedTuple):
    """Static (hashable) surface parameters — used to key JAX recompilation.

    All values are plain Python scalars / tuples, never traced arrays.
    """

    n_surfaces: int
    is_plane: tuple  # bool per surface
    reflective: tuple  # bool per surface
    semi_apertures: tuple  # float per surface; ``math.inf`` disables clipping
    z: tuple  # absolute z position (mm) per surface


class _Params(NamedTuple):
    """Traced numerical parameters — JAX differentiates through these.

    All fields are 1-D arrays of length ``n_surfaces``.
    """

    radii: object  # R (ignored when is_plane=True)
    conics: object  # k
    n1: object  # refractive index before each surface
    n2: object  # refractive index after each surface
