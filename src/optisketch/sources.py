# The hexapolar pupil sampling pattern in this module is derived from Optiland
# (https://github.com/optiland/optiland), copyright (c) 2024 Kramer Harrison,
# licensed under the MIT License. See NOTICE and licenses/optiland_license.txt.

"""Ray sources, pupil sampling patterns, and the ``emit`` function.

Phase 3: :class:`Source` NamedTuple, factory functions
(:func:`point_source`, :func:`collimated_source`, :func:`extended_source`),
pupil sampling patterns, and :func:`emit`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from optisketch.backends._protocol import Backend
    from optisketch.rays import Rays, System


# ---------------------------------------------------------------------------
# Source data structure
# ---------------------------------------------------------------------------


class Source(NamedTuple):
    """Description of a ray source.

    Leaf fields (*field*, *wavelength*) are intended to be traced JAX arrays
    so that ``vmap(emit)`` can batch over them.  The structural fields
    (*kind*, *pupil_pattern*, *n_samples*) are static Python values.

    Attributes
    ----------
    kind : str
        ``"point"`` for a finite-conjugate point source, ``"collimated"`` for
        an infinite-conjugate (plane-wave) source.
    field : array-like, shape (3,) or (2,)
        For ``"point"``: ``[x, y, z]`` position of the source (mm).
        For ``"collimated"``: ``[theta_x, theta_y]`` field angles (rad).
    wavelength : float
        Design wavelength (µm).
    pupil_pattern : str
        One of ``"disk"``, ``"hex"``, ``"ring"``, ``"random"``, ``"fan"``.
    n_samples : int
        Number of rays to emit (for ``"hex"`` this is the number of rings;
        total ray count is ``1 + 3*n*(n+1)``).
    weights : array-like or None
        Per-ray intensity weights.  ``None`` → uniform unit weights.
    """

    kind: str
    field: Any
    wavelength: float
    pupil_pattern: str
    n_samples: int
    weights: Any = None


# ---------------------------------------------------------------------------
# Source factory functions
# ---------------------------------------------------------------------------


def point_source(
    field_xy: tuple[float, float] = (0.0, 0.0),
    z_object: float = -100.0,
    *,
    wavelength: float = 0.55,
    pupil_pattern: str = "disk",
    n_samples: int = 7,
    weights: Any = None,
) -> Source:
    """Create a finite-conjugate point source.

    Parameters
    ----------
    field_xy : tuple of float, optional
        ``(x, y)`` position of the source in the object plane (mm).
        Default ``(0.0, 0.0)`` (on-axis).
    z_object : float, optional
        z-coordinate of the object plane (mm).  Default ``-100.0``.
    wavelength : float, optional
        Design wavelength (µm).  Default ``0.55``.
    pupil_pattern : str, optional
        Pupil sampling pattern.  Default ``"disk"``.
    n_samples : int, optional
        Number of pupil samples (rays).  Default 7.
    weights : array-like or None, optional
        Per-ray intensity weights.  None → uniform.

    Returns
    -------
    Source
        A point source whose ``field`` encodes ``[x, y, z_object]``.
    """
    field = np.array(
        [float(field_xy[0]), float(field_xy[1]), float(z_object)], dtype=np.float64
    )
    return Source(
        kind="point",
        field=field,
        wavelength=float(wavelength),
        pupil_pattern=pupil_pattern,
        n_samples=n_samples,
        weights=weights,
    )


def collimated_source(
    field_angle: tuple[float, float] | float = (0.0, 0.0),
    *,
    wavelength: float = 0.55,
    pupil_pattern: str = "disk",
    n_samples: int = 7,
    weights: Any = None,
) -> Source:
    """Create an infinite-conjugate (collimated) ray source.

    Parameters
    ----------
    field_angle : tuple of float or float, optional
        ``(theta_x, theta_y)`` field angles in radians, or a scalar
        ``theta_y`` with ``theta_x=0``.  Default ``(0.0, 0.0)``.
    wavelength : float, optional
        Design wavelength (µm).  Default ``0.55``.
    pupil_pattern : str, optional
        Pupil sampling pattern.  Default ``"disk"``.
    n_samples : int, optional
        Number of pupil samples (rays).  Default 7.
    weights : array-like or None, optional
        Per-ray intensity weights.  None → uniform.

    Returns
    -------
    Source
        A collimated source whose ``field`` encodes ``[theta_x, theta_y]``.
    """
    if np.isscalar(field_angle):
        field = np.array([0.0, float(field_angle)], dtype=np.float64)
    else:
        field = np.array(
            [float(field_angle[0]), float(field_angle[1])], dtype=np.float64
        )
    return Source(
        kind="collimated",
        field=field,
        wavelength=float(wavelength),
        pupil_pattern=pupil_pattern,
        n_samples=n_samples,
        weights=weights,
    )


def extended_source(
    field_points: Any,
    z_object: float = -100.0,
    *,
    wavelength: float = 0.55,
    pupil_pattern: str = "disk",
    n_samples: int = 7,
    weights: Any = None,
) -> list[Source]:
    """Create a list of point sources sampling an extended object.

    This is conceptually ``[point_source(p, z_object, ...) for p in field_points]``;
    use ``vmap(emit)`` or a loop to trace all sources.

    Parameters
    ----------
    field_points : array-like, shape (N, 2)
        Array of ``(x, y)`` object positions (mm).
    z_object : float, optional
        z-coordinate of the object plane (mm).  Default ``-100.0``.
    wavelength : float, optional
        Design wavelength (µm).  Default ``0.55``.
    pupil_pattern : str, optional
        Pupil sampling pattern.  Default ``"disk"``.
    n_samples : int, optional
        Number of pupil samples per source.  Default 7.
    weights : array-like or None, optional
        Per-ray intensity weights.  None → uniform.

    Returns
    -------
    list of Source
        One :class:`Source` per field point.
    """
    pts = np.asarray(field_points, dtype=np.float64)
    return [
        point_source(
            (pts[i, 0], pts[i, 1]),
            z_object,
            wavelength=wavelength,
            pupil_pattern=pupil_pattern,
            n_samples=n_samples,
            weights=weights,
        )
        for i in range(len(pts))
    ]


# ---------------------------------------------------------------------------
# Pupil sampling patterns
# ---------------------------------------------------------------------------


def _pupil_disk(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform disk sampling via Fibonacci/sunflower spiral.

    Parameters
    ----------
    n_samples : int
        Number of sample points.

    Returns
    -------
    px, py : np.ndarray
        Normalised sample coordinates within the unit disk.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    idx = np.arange(n_samples, dtype=np.float64)
    r = np.sqrt((idx + 0.5) / n_samples)
    theta = 2.0 * math.pi * idx / (golden**2)
    return r * np.cos(theta), r * np.sin(theta)


def _pupil_hex(rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Hexapolar sampling with ``1 + 3*rings*(rings+1)`` points.

    Parameters
    ----------
    rings : int
        Number of concentric rings (0 → centre point only).

    Returns
    -------
    px, py : np.ndarray
        Normalised sample coordinates.
    """
    px_list: list[float] = [0.0]
    py_list: list[float] = [0.0]
    for ring in range(1, rings + 1):
        r = ring / rings
        n_on_ring = 6 * ring
        thetas = np.linspace(0.0, 2.0 * math.pi, n_on_ring, endpoint=False)
        px_list.extend((r * np.cos(thetas)).tolist())
        py_list.extend((r * np.sin(thetas)).tolist())
    return np.array(px_list), np.array(py_list)


def _pupil_ring(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly spaced points on the unit circle.

    Parameters
    ----------
    n_samples : int
        Number of sample points.

    Returns
    -------
    px, py : np.ndarray
        Normalised sample coordinates on the unit circle.
    """
    thetas = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    return np.cos(thetas), np.sin(thetas)


def _pupil_random(n_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random points within the unit disk.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    seed : int, optional
        Random seed for reproducibility.  Default 0.

    Returns
    -------
    px, py : np.ndarray
        Normalised sample coordinates within the unit disk.
    """
    rng = np.random.default_rng(seed)
    pts: list[list[float]] = []
    batch = 2 * n_samples
    while len(pts) < n_samples:
        candidates = rng.uniform(-1.0, 1.0, (batch, 2))
        r2 = candidates[:, 0] ** 2 + candidates[:, 1] ** 2
        inside = candidates[r2 <= 1.0]
        pts.extend(inside.tolist())
    arr = np.array(pts[:n_samples])
    return arr[:, 0], arr[:, 1]


def _pupil_fan(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Fan of points along the y-axis from -1 to +1.

    Parameters
    ----------
    n_samples : int
        Number of sample points.

    Returns
    -------
    px, py : np.ndarray
        Normalised sample coordinates along the y-axis.
    """
    py = np.linspace(-1.0, 1.0, n_samples)
    px = np.zeros(n_samples)
    return px, py


_PUPIL_SAMPLERS = {
    "disk": _pupil_disk,
    "ring": _pupil_ring,
    "fan": _pupil_fan,
}


def _sample_pupil(
    pattern: str,
    n_samples: int,
    pupil_semi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate pupil sampler and scale by *pupil_semi*.

    Parameters
    ----------
    pattern : str
        One of ``"disk"``, ``"hex"``, ``"ring"``, ``"random"``, ``"fan"``.
    n_samples : int
        Number of samples (or ring count for ``"hex"``).
    pupil_semi : float
        Semi-aperture radius to scale normalised coordinates by (mm).

    Returns
    -------
    px, py : np.ndarray
        Sample coordinates in the pupil plane (mm).
    """
    if pattern == "hex":
        pxn, pyn = _pupil_hex(n_samples)
    elif pattern == "random":
        pxn, pyn = _pupil_random(n_samples)
    elif pattern in _PUPIL_SAMPLERS:
        pxn, pyn = _PUPIL_SAMPLERS[pattern](n_samples)
    else:
        raise ValueError(
            f"Unknown pupil pattern {pattern!r}. "
            f"Choose from: disk, hex, ring, random, fan."
        )
    return pxn * pupil_semi, pyn * pupil_semi


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------


def emit(source: Source, system: System) -> Rays:
    """Convert *source* into a :class:`~optisketch.rays.Rays` bundle.

    Pupil sampling is computed once in NumPy (static); direction arithmetic
    uses the system backend so that ``vmap(emit)`` over the ``field`` leaf is
    JAX-traceable.

    Parameters
    ----------
    source : Source
        Ray source description.
    system : System
        Compiled optical system providing ``pupil_z``, ``pupil_semi``, and
        the backend.

    Returns
    -------
    Rays
        Initial ray bundle placed at the entrance pupil, ready to pass to
        :func:`~optisketch.kernels._trace_surfaces`.

    Raises
    ------
    ValueError
        If ``source.kind`` is not ``"point"`` or ``"collimated"``.
    """
    from optisketch.rays import Rays

    be: Backend = system.backend
    pupil_z = float(system.pupil_z)
    pupil_semi = float(system.pupil_semi)

    # --- pupil samples (NumPy, static / not traced) -----------------------
    px_np, py_np = _sample_pupil(source.pupil_pattern, source.n_samples, pupil_semi)
    n = len(px_np)

    # Convert to backend arrays (treated as constants under JAX vmap)
    px = be.asarray(px_np)
    py = be.asarray(py_np)
    pz = be.full(n, pupil_z)

    # --- weights -----------------------------------------------------------
    if source.weights is None:
        w_arr = be.asarray(np.ones(n))
    else:
        w_arr = be.asarray(np.asarray(source.weights, dtype=np.float64))

    wl = be.full(n, source.wavelength)
    opd = be.zeros(n)
    valid = be.asarray(np.ones(n, dtype=bool))

    field = source.field

    if source.kind == "point":
        # field = [x, y, z] of the source point
        fx = field[0]
        fy = field[1]
        fz = field[2]
        dx = px - fx
        dy = py - fy
        dz = pz - fz
        mag = be.sqrt(dx * dx + dy * dy + dz * dz)
        L = dx / mag
        M = dy / mag
        N = dz / mag
        x = be.full(n, fx)
        y = be.full(n, fy)
        z = be.full(n, fz)

    elif source.kind == "collimated":
        # field = [theta_x, theta_y] in radians
        theta_x = field[0]
        theta_y = field[1]
        L_val = be.sin(be.asarray(theta_x))
        M_val = be.sin(be.asarray(theta_y))
        N_val = be.sqrt(be.asarray(1.0) - L_val * L_val - M_val * M_val)
        # Broadcast scalars to per-ray arrays
        L = be.full(n, L_val)
        M = be.full(n, M_val)
        N = be.full(n, N_val)
        x = px
        y = py
        z = pz

    else:
        raise ValueError(
            f"Unknown source kind {source.kind!r}. Choose 'point' or 'collimated'."
        )

    return Rays(
        x=x,
        y=y,
        z=z,
        L=L,
        M=M,
        N=N,
        i=w_arr,
        w=wl,
        opd=opd,
        valid=valid,
    )
