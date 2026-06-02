"""Pure functional ray-tracing kernels.

Every function is stateless and backend-agnostic: the caller passes an
explicit ``Backend`` instance as the last argument.  No global state, no
mutation.  Ported from Optiland's ``StandardGeometry`` and ``RealRays``.

References
----------
* ``_intersect`` ← ``optiland/geometries/standard.py::StandardGeometry.distance``
* ``_normal``    ← ``optiland/geometries/standard.py::StandardGeometry.surface_normal``
* ``_refract``   ← ``optiland/rays/real_rays.py::RealRays.refract``
* ``_reflect``   ← ``optiland/rays/real_rays.py::RealRays.reflect``
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from trace_light.rays import Rays, _Params, _Structure

if TYPE_CHECKING:
    from trace_light.backends._protocol import Backend


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------


def _intersect(
    x: Any,
    y: Any,
    z: Any,
    L: Any,
    M: Any,
    N: Any,
    R: Any,
    k: Any,
    is_plane: bool,
    be: Backend,
) -> Any:
    """Return propagation distance *t* to the next surface in the local frame.

    Surface vertex is at the local origin. For planar surfaces, solves
    ``z + t*N = 0``. For conic/spherical surfaces, solves the quadratic
    intersection equation following Optiland's ``StandardGeometry.distance``.

    Parameters
    ----------
    x : array
        Ray x-positions in the local surface frame (mm).
    y : array
        Ray y-positions in the local surface frame (mm).
    z : array
        Ray z-positions in the local surface frame (mm).
    L : array
        Ray direction cosine along x.
    M : array
        Ray direction cosine along y.
    N : array
        Ray direction cosine along z.
    R : array or float
        Surface radius of curvature (mm). Ignored when *is_plane* is True.
    k : array or float
        Conic constant. Zero gives a sphere; ignored when *is_plane* is True.
    is_plane : bool
        When True, treat the surface as a flat plane (z=0 in local frame).
    be : Backend
        Array-computation backend to use for all operations.

    Returns
    -------
    t : array
        Propagation distance to the surface intersection. Non-finite (NaN)
        for rays that miss the surface (negative discriminant).
    """
    if is_plane:
        # intersection with the plane z=0:  z + t*N = 0  →  t = -z/N
        N_safe = be.where(be.abs(N) > 1e-14, N, be.full_like(N, 1e-14))
        return -z / N_safe

    # --- conic / sphere (Optiland StandardGeometry.distance) ---
    # a = (1 + k)*N² + L² + M²  (= 1 + k*N² for unit direction)
    a = k * N * N + L * L + M * M + N * N
    b = 2.0 * k * N * z + 2.0 * L * x + 2.0 * M * y - 2.0 * N * R + 2.0 * N * z
    c = k * z * z - 2.0 * R * z + x * x + y * y + z * z

    disc = b * b - 4.0 * a * c
    # guard sqrt of negative discriminant (miss → t will be non-finite, caught by valid)
    safe_disc = be.where(disc >= 0.0, disc, be.zeros_like(disc))
    sq = be.sqrt(safe_disc)

    t1 = (-b + sq) / (2.0 * a)
    t2 = (-b - sq) / (2.0 * a)

    # pick the intersection closest to the vertex (z=0)
    z1 = z + t1 * N
    z2 = z + t2 * N
    t_conic = be.where(be.abs(z1) <= be.abs(z2), t1, t2)

    # degenerate case a=0 (ray parallel to axis of a sphere with k=-1, etc.)
    b_safe = be.where(be.abs(b) > 1e-14, b, be.full_like(b, 1e-14))
    t_degenerate = -c / b_safe
    t_conic = be.where(a == 0.0, t_degenerate, t_conic)

    # mark misses (negative discriminant) as non-finite so valid catches them
    t_conic = be.where(disc < 0.0, be.full_like(t_conic, float("nan")), t_conic)
    return t_conic


# ---------------------------------------------------------------------------
# Surface normal
# ---------------------------------------------------------------------------


def _normal(
    x: Any,
    y: Any,
    R: Any,
    k: Any,
    is_plane: bool,
    be: Backend,
) -> tuple[Any, Any, Any]:
    """Return outward surface normal (unit vector) at the intersection point.

    For planar surfaces returns ``(0, 0, 1)``. For conic surfaces the gradient
    of the conic equation gives the un-normalised normal; the z-component is
    negative following the Optiland sign convention. :func:`_align_normal`
    corrects the orientation per-ray.

    Parameters
    ----------
    x : array
        Ray x-positions at the surface intersection point (mm).
    y : array
        Ray y-positions at the surface intersection point (mm).
    R : array or float
        Surface radius of curvature (mm).
    k : array or float
        Conic constant.
    is_plane : bool
        When True, return the flat-surface normal ``(0, 0, 1)``.
    be : Backend
        Array-computation backend.

    Returns
    -------
    nx : array
        x-component of the outward surface normal.
    ny : array
        y-component of the outward surface normal.
    nz : array
        z-component of the outward surface normal.
    """
    if is_plane:
        zero = be.zeros_like(x)
        one = be.ones_like(x)
        return zero, zero, one

    r2 = x * x + y * y
    # denom = R * sqrt(1 - (1+k)*r²/R²)
    denom = R * be.sqrt(1.0 - (1.0 + k) * r2 / (R * R))
    dfdx = x / denom
    dfdy = y / denom
    # unnormalized: (dfdx, dfdy, -1)
    mag = be.sqrt(dfdx * dfdx + dfdy * dfdy + 1.0)
    nx = dfdx / mag
    ny = dfdy / mag
    nz = -be.ones_like(x) / mag
    return nx, ny, nz


# ---------------------------------------------------------------------------
# Normal alignment (shared by refract and reflect)
# ---------------------------------------------------------------------------


def _align_normal(
    L: Any,
    M: Any,
    N: Any,
    nx: Any,
    ny: Any,
    nz: Any,
    be: Backend,
) -> tuple[Any, Any, Any, Any]:
    """Flip the surface normal so it opposes the incident ray direction.

    The sign of the normal is corrected so that ``d · n_aligned < 0``
    (the normal points toward the ray source). Mirrors Optiland's
    ``_align_surface_normal``.

    Parameters
    ----------
    L : array
        Incident ray direction cosine along x.
    M : array
        Incident ray direction cosine along y.
    N : array
        Incident ray direction cosine along z.
    nx : array
        x-component of the surface normal (may be misoriented).
    ny : array
        y-component of the surface normal (may be misoriented).
    nz : array
        z-component of the surface normal (may be misoriented).
    be : Backend
        Array-computation backend.

    Returns
    -------
    nx : array
        x-component of the aligned surface normal.
    ny : array
        y-component of the aligned surface normal.
    nz : array
        z-component of the aligned surface normal.
    dot : array
        Absolute value of the dot product ``|d · n_aligned|`` (positive).
    """
    dot = L * nx + M * ny + N * nz
    sgn = be.sign(dot)
    nx = nx * sgn
    ny = ny * sgn
    nz = nz * sgn
    dot = be.abs(dot)
    return nx, ny, nz, dot


# ---------------------------------------------------------------------------
# Refraction (vector Snell's law)
# ---------------------------------------------------------------------------


def _refract(
    L: Any,
    M: Any,
    N: Any,
    nx: Any,
    ny: Any,
    nz: Any,
    n1: Any,
    n2: Any,
    be: Backend,
) -> tuple[Any, Any, Any, Any]:
    """Refract a ray bundle using the vector form of Snell's law.

    Port of Optiland ``RealRays.refract``. TIR rays have their ``valid``
    flag set to False downstream; their direction values are set to the
    ``sqrt(0)`` fallback to avoid propagating NaN.

    Parameters
    ----------
    L : array
        Incident ray direction cosine along x.
    M : array
        Incident ray direction cosine along y.
    N : array
        Incident ray direction cosine along z.
    nx : array
        x-component of the surface normal (pre-alignment).
    ny : array
        y-component of the surface normal (pre-alignment).
    nz : array
        z-component of the surface normal (pre-alignment).
    n1 : array or float
        Refractive index of the incident medium.
    n2 : array or float
        Refractive index of the transmitting medium.
    be : Backend
        Array-computation backend.

    Returns
    -------
    Lout : array
        Refracted ray direction cosine along x.
    Mout : array
        Refracted ray direction cosine along y.
    Nout : array
        Refracted ray direction cosine along z.
    tir : bool array
        True for each ray that undergoes total internal reflection.
    """
    nx, ny, nz, dot = _align_normal(L, M, N, nx, ny, nz, be)

    u = n1 / n2
    discriminant = 1.0 - u * u * (1.0 - dot * dot)
    tir = discriminant < 0.0
    safe_disc = be.where(discriminant >= 0.0, discriminant, be.zeros_like(discriminant))
    root = be.sqrt(safe_disc)

    Lout = u * L + nx * root - u * nx * dot
    Mout = u * M + ny * root - u * ny * dot
    Nout = u * N + nz * root - u * nz * dot
    return Lout, Mout, Nout, tir


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


def _reflect(
    L: Any,
    M: Any,
    N: Any,
    nx: Any,
    ny: Any,
    nz: Any,
    be: Backend,
) -> tuple[Any, Any, Any]:
    """Reflect a ray bundle off a surface: ``d_out = d - 2(d·n)n``.

    Port of Optiland ``RealRays.reflect``.

    Parameters
    ----------
    L : array
        Incident ray direction cosine along x.
    M : array
        Incident ray direction cosine along y.
    N : array
        Incident ray direction cosine along z.
    nx : array
        x-component of the surface normal (pre-alignment).
    ny : array
        y-component of the surface normal (pre-alignment).
    nz : array
        z-component of the surface normal (pre-alignment).
    be : Backend
        Array-computation backend.

    Returns
    -------
    Lout : array
        Reflected ray direction cosine along x.
    Mout : array
        Reflected ray direction cosine along y.
    Nout : array
        Reflected ray direction cosine along z.
    """
    nx, ny, nz, dot = _align_normal(L, M, N, nx, ny, nz, be)
    Lout = L - 2.0 * dot * nx
    Mout = M - 2.0 * dot * ny
    Nout = N - 2.0 * dot * nz
    return Lout, Mout, Nout


# ---------------------------------------------------------------------------
# Single surface step
# ---------------------------------------------------------------------------


def _surface_step(
    rays: Rays,
    z_surf: float,
    is_plane: bool,
    R: Any,
    k: Any,
    n1: Any,
    n2: Any,
    semi: float,
    reflective: bool,
    be: Backend,
) -> Rays:
    """Propagate *rays* through one optical surface and return updated rays.

    Executes the full surface-interaction pipeline in order:

    1. **Localise** — translate ray z-positions to the surface vertex frame.
    2. **Intersect** — compute propagation distance *t* to the surface.
    3. **Propagate** — advance ray positions by *t*.
    4. **Normal** — compute the outward surface normal at the hit point.
    5. **Interact** — apply refraction or reflection.
    6. **Aperture clip** — mark rays outside the semi-aperture as invalid.
    7. **OPD update** — accumulate optical path length.
    8. **Globalise** — translate z-positions back to the global frame.

    Parameters
    ----------
    rays : Rays
        Input ray bundle in the global coordinate frame.
    z_surf : float
        Absolute z-position of the surface vertex in the global frame (mm).
    is_plane : bool
        When True, treat the surface as a flat plane.
    R : array or float
        Surface radius of curvature (mm).
    k : array or float
        Conic constant.
    n1 : array or float
        Refractive index of the medium before the surface.
    n2 : array or float
        Refractive index of the medium after the surface.
    semi : float
        Semi-aperture radius (mm). Pass ``math.inf`` to disable clipping.
    reflective : bool
        When True, apply reflection instead of refraction.
    be : Backend
        Array-computation backend.

    Returns
    -------
    Rays
        Updated ray bundle after the surface interaction (global frame).
        ``valid`` is False for any ray that missed, underwent TIR, or
        fell outside the aperture.
    """
    # 1. Localize
    z_loc = rays.z - z_surf

    # 2. Intersect
    t = _intersect(rays.x, rays.y, z_loc, rays.L, rays.M, rays.N, R, k, is_plane, be)

    # 3. Propagate (still in local z)
    x = rays.x + t * rays.L
    y = rays.y + t * rays.M
    z_hit = z_loc + t * rays.N

    # 4. Normal
    nx, ny, nz = _normal(x, y, R, k, is_plane, be)

    # 5. Interact
    if reflective:
        L, M, N = _reflect(rays.L, rays.M, rays.N, nx, ny, nz, be)
        tir = be.zeros_like(rays.valid)  # reflection cannot TIR
    else:
        L, M, N, tir = _refract(rays.L, rays.M, rays.N, nx, ny, nz, n1, n2, be)

    # 6. Aperture check
    r2 = x * x + y * y
    if math.isinf(semi):
        outside = be.zeros_like(rays.valid)  # no clipping
    else:
        outside = r2 > semi * semi

    # 7. OPD
    opd = rays.opd + n1 * t

    # valid: propagate existing validity AND all new conditions
    valid = rays.valid & be.isfinite(t) & ~outside & ~tir

    # 8. Globalize
    z_global = z_hit + z_surf

    return Rays(
        x=x,
        y=y,
        z=z_global,
        L=L,
        M=M,
        N=N,
        i=rays.i,
        w=rays.w,
        opd=opd,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Full multi-surface trace
# ---------------------------------------------------------------------------


def _trace_surfaces(
    rays_init: Rays,
    structure: _Structure,
    params: _Params,
    be: Backend,
) -> tuple[Rays, list[Any]]:
    """Trace *rays_init* through every surface defined in *structure*.

    The Python ``for`` loop over ``structure.n_surfaces`` unrolls
    statically under ``jax.jit`` because ``_Structure`` contains only
    plain Python scalars.

    Parameters
    ----------
    rays_init : Rays
        Initial ray bundle before any surface interaction.
    structure : _Structure
        Static (hashable) surface geometry parameters: vertex positions,
        semi-apertures, and per-surface flags (plane/conic, reflective).
    params : _Params
        Traced numerical parameters (radii, conics, refractive indices).
        JAX differentiates through these arrays.
    be : Backend
        Array-computation backend.

    Returns
    -------
    final_rays : Rays
        Ray bundle after the last surface.
    history : list of array
        List of ``(n_rays, 3)`` position arrays in ``[x, y, z]`` order
        along the last axis. Length is ``structure.n_surfaces + 1``
        (initial position plus one entry per surface).
    """
    rays = rays_init
    history = [be.stack([rays.x, rays.y, rays.z], axis=-1)]

    for i in range(structure.n_surfaces):
        rays = _surface_step(
            rays,
            z_surf=structure.z[i],
            is_plane=structure.is_plane[i],
            R=params.radii[i],
            k=params.conics[i],
            n1=params.n1[i],
            n2=params.n2[i],
            semi=structure.semi_apertures[i],
            reflective=structure.reflective[i],
            be=be,
        )
        history.append(be.stack([rays.x, rays.y, rays.z], axis=-1))

    return rays, history
