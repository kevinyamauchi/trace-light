"""Lens factory functions.

Each factory returns a ``tuple[Surface, ...]`` with surfaces placed relative
to ``z=0`` (first surface at ``z=0``).  Pass the tuple to
:meth:`SystemBuilder.add` to place the element at the builder's current
z-cursor.

All refractive index arguments default to ``n_ambient=1.0`` (air).
"""

from __future__ import annotations

import math

from optisketch.rays import Surface


def singlet(
    *,
    R1: float,
    R2: float,
    n: float,
    thickness: float,
    n_ambient: float = 1.0,
    semi_aperture: float = math.inf,
) -> tuple[Surface, ...]:
    """Create a general singlet (two-surface) lens.

    Parameters
    ----------
    R1 : float
        Radius of curvature of the front surface (mm).  ``math.inf`` → plane.
    R2 : float
        Radius of curvature of the back surface (mm).  ``math.inf`` → plane.
    n : float
        Refractive index of the glass.
    thickness : float
        Centre thickness of the element (mm).
    n_ambient : float, optional
        Refractive index of the surrounding medium.  Default ``1.0`` (air).
    semi_aperture : float, optional
        Semi-aperture radius (mm) applied to both surfaces.

    Returns
    -------
    tuple of Surface
        Two-element tuple ``(front, back)`` with z-positions ``0`` and
        ``thickness``.
    """
    return (
        Surface(
            z=0.0, radius=R1, conic=0.0, n1=n_ambient, n2=n, semi_aperture=semi_aperture
        ),
        Surface(
            z=thickness,
            radius=R2,
            conic=0.0,
            n1=n,
            n2=n_ambient,
            semi_aperture=semi_aperture,
        ),
    )


def biconvex(
    *,
    R: float,
    n: float,
    thickness: float,
    n_ambient: float = 1.0,
    semi_aperture: float = math.inf,
) -> tuple[Surface, ...]:
    """Create a symmetric biconvex lens (R1 = +R, R2 = -R).

    Parameters
    ----------
    R : float
        Magnitude of the radius of curvature (mm).
    n : float
        Refractive index of the glass.
    thickness : float
        Centre thickness (mm).
    n_ambient : float, optional
        Refractive index of the surrounding medium.
    semi_aperture : float, optional
        Semi-aperture radius (mm).

    Returns
    -------
    tuple of Surface
        Two-element tuple ``(front, back)``.
    """
    return singlet(
        R1=R,
        R2=-R,
        n=n,
        thickness=thickness,
        n_ambient=n_ambient,
        semi_aperture=semi_aperture,
    )


def plano_convex(
    *,
    R: float,
    n: float,
    thickness: float,
    n_ambient: float = 1.0,
    semi_aperture: float = math.inf,
) -> tuple[Surface, ...]:
    """Create a plano-convex lens (curved front, flat back).

    Parameters
    ----------
    R : float
        Radius of curvature of the front surface (mm).
    n : float
        Refractive index of the glass.
    thickness : float
        Centre thickness (mm).
    n_ambient : float, optional
        Refractive index of the surrounding medium.
    semi_aperture : float, optional
        Semi-aperture radius (mm).

    Returns
    -------
    tuple of Surface
        Two-element tuple ``(front, back)`` where the back surface is a plane.
    """
    return singlet(
        R1=R,
        R2=math.inf,
        n=n,
        thickness=thickness,
        n_ambient=n_ambient,
        semi_aperture=semi_aperture,
    )


def doublet(
    *,
    R1: float,
    R2: float,
    R3: float,
    n_crown: float,
    n_flint: float,
    thickness_crown: float,
    thickness_flint: float,
    n_ambient: float = 1.0,
    semi_aperture: float = math.inf,
) -> tuple[Surface, ...]:
    """Create an achromatic doublet (crown + flint, cemented).

    Parameters
    ----------
    R1 : float
        Front radius of the crown element (mm).
    R2 : float
        Cemented interface radius (mm).
    R3 : float
        Back radius of the flint element (mm).
    n_crown : float
        Refractive index of the crown glass.
    n_flint : float
        Refractive index of the flint glass.
    thickness_crown : float
        Axial thickness of the crown element (mm).
    thickness_flint : float
        Axial thickness of the flint element (mm).
    n_ambient : float, optional
        Refractive index of the surrounding medium.
    semi_aperture : float, optional
        Semi-aperture radius (mm) applied to all three surfaces.

    Returns
    -------
    tuple of Surface
        Three-element tuple ``(front, cement, back)``.
    """
    z_cement = thickness_crown
    z_back = thickness_crown + thickness_flint
    return (
        Surface(
            z=0.0,
            radius=R1,
            conic=0.0,
            n1=n_ambient,
            n2=n_crown,
            semi_aperture=semi_aperture,
        ),
        Surface(
            z=z_cement,
            radius=R2,
            conic=0.0,
            n1=n_crown,
            n2=n_flint,
            semi_aperture=semi_aperture,
        ),
        Surface(
            z=z_back,
            radius=R3,
            conic=0.0,
            n1=n_flint,
            n2=n_ambient,
            semi_aperture=semi_aperture,
        ),
    )


def thin_lens(
    *,
    f: float,
    n: float = 1.5,
    n_ambient: float = 1.0,
    semi_aperture: float = math.inf,
) -> tuple[Surface, ...]:
    """Create a thin-lens approximation element (zero thickness).

    The element is modelled as two surfaces co-located at ``z=0`` using the
    symmetric biconvex form:
    ``1/f = (n - n_ambient) * 2 / R`` → ``R = 2*(n - n_ambient)*f``.

    Parameters
    ----------
    f : float
        Paraxial focal length (mm).
    n : float, optional
        Refractive index of the glass.  Default 1.5.
    n_ambient : float, optional
        Refractive index of the surrounding medium.  Default 1.0.
    semi_aperture : float, optional
        Semi-aperture radius (mm).

    Returns
    -------
    tuple of Surface
        Two surfaces both at ``z=0`` (zero thickness).
    """
    R = 2.0 * (n - n_ambient) * f
    return (
        Surface(
            z=0.0, radius=R, conic=0.0, n1=n_ambient, n2=n, semi_aperture=semi_aperture
        ),
        Surface(
            z=0.0, radius=-R, conic=0.0, n1=n, n2=n_ambient, semi_aperture=semi_aperture
        ),
    )


def mirror(
    *,
    R: float,
    semi_aperture: float = math.inf,
    n_ambient: float = 1.0,
) -> tuple[Surface, ...]:
    """Create a concave/convex reflective mirror surface.

    Parameters
    ----------
    R : float
        Radius of curvature (mm).  Negative → concave (converging for axial
        rays propagating in +z).
    semi_aperture : float, optional
        Semi-aperture radius (mm).
    n_ambient : float, optional
        Refractive index of the surrounding medium.

    Returns
    -------
    tuple of Surface
        Single-element tuple containing the mirror surface.
    """
    return (
        Surface(
            z=0.0,
            radius=R,
            conic=0.0,
            n1=n_ambient,
            n2=n_ambient,
            semi_aperture=semi_aperture,
            reflective=True,
        ),
    )


def aperture(*, z: float = 0.0, semi: float) -> tuple[Surface, ...]:
    """Create a pure aperture stop (no optical power).

    Parameters
    ----------
    z : float, optional
        z-position of the aperture (mm).  Default 0.0.
    semi : float
        Semi-aperture radius (mm).

    Returns
    -------
    tuple of Surface
        Single-element tuple containing the aperture surface.
    """
    return (
        Surface(
            z=z,
            radius=math.inf,
            conic=0.0,
            n1=1.0,
            n2=1.0,
            semi_aperture=semi,
            reflective=False,
        ),
    )
