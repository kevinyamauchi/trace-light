"""SystemBuilder and prefab optical systems.

:class:`SystemBuilder` assembles :class:`~trace_light.rays.Surface` objects
into a :class:`~trace_light.rays.System`.  The ``systems.*`` prefabs wrap the
builder to create common configurations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from trace_light.rays import Surface, System, _Params, _Structure

if TYPE_CHECKING:
    from trace_light.backends._protocol import Backend


def _default_backend() -> Backend:
    """Return a NumpyBackend instance (avoids import at module level)."""
    from trace_light.backends._numpy import NumpyBackend

    return NumpyBackend()


class SystemBuilder:
    """Incrementally assemble an optical system from surfaces and gaps.

    Usage::

        b = SystemBuilder(wavelengths=(0.55,))
        b.add(*biconvex(R=100, n=1.5, thickness=10))
        b.gap(190)
        b.add(*biconvex(R=100, n=1.5, thickness=10))
        b.gap(190)
        b.image()
        system = b.finalize()

    The builder tracks a z-cursor.  :meth:`add` offsets supplied surfaces by
    the cursor and advances the cursor to the back of the element.
    :meth:`gap` advances the cursor by a fixed thickness.
    """

    def __init__(self, *, wavelengths: tuple = (0.55,)) -> None:
        """Initialise an empty builder.

        Parameters
        ----------
        wavelengths : tuple of float, optional
            Design wavelengths (µm).  Default ``(0.55,)``.

        Returns
        -------
        None
        """
        self._z: float = 0.0
        self._surfaces: list[Surface] = []
        self._stop_z: float | None = None
        self._stop_semi: float | None = None
        self._image_z: float | None = None
        self._wavelengths: tuple = tuple(wavelengths)

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def add(self, *surfaces: Surface) -> SystemBuilder:
        """Add one or more surfaces to the system.

        Surface z-coordinates are treated as relative to the builder's current
        z-cursor; the builder offsets them and advances the cursor to the back
        of the element.

        Parameters
        ----------
        *surfaces : Surface
            One or more surfaces with z-coordinates relative to 0 (as returned
            by the ``lenses.*`` factories).

        Returns
        -------
        SystemBuilder
            This builder (for chaining).
        """
        if not surfaces:
            return self
        z_off = self._z
        for s in surfaces:
            self._surfaces.append(s._replace(z=s.z + z_off))
        span = max(s.z for s in surfaces)
        self._z += span
        return self

    def gap(self, thickness: float, n: float = 1.0) -> SystemBuilder:
        """Advance the z-cursor by *thickness*.

        The *n* parameter is accepted for API symmetry with builders that track
        medium indices, but is not used internally (indices come from the
        Surface objects supplied to :meth:`add`).

        Parameters
        ----------
        thickness : float
            Axial distance to advance (mm).
        n : float, optional
            Refractive index of the gap medium (informational only).

        Returns
        -------
        SystemBuilder
            This builder (for chaining).
        """
        self._z += thickness
        return self

    def stop(self, semi: float) -> SystemBuilder:
        """Record an aperture stop at the current z-cursor position.

        Overrides the default entrance-pupil location (first powered surface).

        Parameters
        ----------
        semi : float
            Semi-aperture of the stop (mm).

        Returns
        -------
        SystemBuilder
            This builder (for chaining).
        """
        self._stop_z = self._z
        self._stop_semi = semi
        return self

    def image(self, z: float | None = None) -> SystemBuilder:
        """Set the image-plane z-position.

        Parameters
        ----------
        z : float, optional
            Image plane z (mm).  When None, uses the current z-cursor.

        Returns
        -------
        SystemBuilder
            This builder (for chaining).
        """
        self._image_z = self._z if z is None else z
        return self

    def finalize(self, backend: Backend | None = None) -> System:
        """Compile the assembled surfaces into a :class:`System`.

        Parameters
        ----------
        backend : Backend, optional
            Backend to bind.  Defaults to :class:`~trace_light.backends.NumpyBackend`.

        Returns
        -------
        System
            Compiled system ready for ray tracing.

        Raises
        ------
        ValueError
            If no surfaces have been added or no image plane was set.
        """
        if not self._surfaces:
            raise ValueError("No surfaces added to SystemBuilder.")
        if self._image_z is None:
            raise ValueError("Image plane not set; call .image() before .finalize().")

        if backend is None:
            backend = _default_backend()

        # Sort surfaces by z (should already be in order)
        surfs = sorted(self._surfaces, key=lambda s: s.z)

        is_plane = tuple(math.isinf(s.radius) for s in surfs)
        reflective = tuple(s.reflective for s in surfs)

        structure = _Structure(
            n_surfaces=len(surfs),
            is_plane=is_plane,
            reflective=reflective,
        )

        # Traced params: z (spacing) and semi_aperture are traced so they are
        # differentiable / sweepable without recompilation (DESIGN §4.2). Plane
        # radii get a finite placeholder (unused when is_plane=True).
        radii_np = np.array([1.0 if math.isinf(s.radius) else s.radius for s in surfs])
        params = _Params(
            z=backend.asarray(np.array([s.z for s in surfs])),
            radii=backend.asarray(radii_np),
            conics=backend.asarray(np.array([s.conic for s in surfs])),
            n1=backend.asarray(np.array([s.n1 for s in surfs])),
            n2=backend.asarray(np.array([s.n2 for s in surfs])),
            semi_aperture=backend.asarray(np.array([s.semi_aperture for s in surfs])),
        )

        # Determine entrance pupil
        if self._stop_z is not None and self._stop_semi is not None:
            pupil_z = self._stop_z
            pupil_semi = self._stop_semi
        else:
            # Default: first powered (non-plane) surface
            first_powered = next(
                (s for s in surfs if not math.isinf(s.radius)),
                surfs[0],
            )
            pupil_z = first_powered.z
            pupil_semi = first_powered.semi_aperture

        return System(
            structure=structure,
            params=params,
            pupil_z=pupil_z,
            pupil_semi=pupil_semi,
            image_z=self._image_z,
            wavelengths=self._wavelengths,
            backend=backend,
        )


# ---------------------------------------------------------------------------
# Prefab systems
# ---------------------------------------------------------------------------


def four_f(
    *,
    f1: float = 100.0,
    f2: float = 100.0,
    n: float = 1.5,
    thickness: float = 10.0,
    pupil_semi: float = 10.0,
    wavelengths: tuple = (0.55,),
    backend: Backend | None = None,
) -> System:
    """Create a 4-f relay optical system.

    Two biconvex lenses separated by the sum of their focal lengths with the
    image plane at 4x the focal length from the first lens.

    Parameters
    ----------
    f1 : float, optional
        Focal length of the first lens (mm).  Default 100.
    f2 : float, optional
        Focal length of the second lens (mm).  Default 100.
    n : float, optional
        Glass refractive index for both lenses.  Default 1.5.
    thickness : float, optional
        Centre thickness of each lens element (mm).  Default 10.
    pupil_semi : float, optional
        Entrance-pupil semi-aperture (mm).  Default 10.
    wavelengths : tuple, optional
        Design wavelengths (µm).  Default ``(0.55,)``.
    backend : Backend, optional
        Backend to bind.  Defaults to NumpyBackend.

    Returns
    -------
    System
        4-f relay system.
    """
    from trace_light.lenses import biconvex

    b = SystemBuilder(wavelengths=wavelengths)
    b.add(*biconvex(R=f1, n=n, thickness=thickness))
    b.gap(f1 + f2 - thickness)
    b.add(*biconvex(R=f2, n=n, thickness=thickness))
    b.gap(f2 - thickness)
    b.image()
    # Place entrance pupil at front of first lens with user-specified semi
    b._stop_z = 0.0
    b._stop_semi = pupil_semi
    return b.finalize(backend=backend)


def microscope(
    *,
    f_obj: float = 4.0,
    f_tube: float = 200.0,
    n_obj: float = 1.5,
    n_tube: float = 1.5,
    thickness_obj: float = 2.0,
    thickness_tube: float = 5.0,
    pupil_semi: float = 3.0,
    wavelengths: tuple = (0.55,),
    backend: Backend | None = None,
) -> System:
    """Create an infinity-corrected microscope (objective + tube lens).

    The object is placed at the front focal plane of the objective so that
    rays from a point source exit the objective as a collimated bundle, which
    the tube lens focuses to the image plane.

    Parameters
    ----------
    f_obj : float, optional
        Objective focal length (mm).  Default 4.
    f_tube : float, optional
        Tube lens focal length (mm).  Default 200.
    n_obj : float, optional
        Objective glass index.  Default 1.5.
    n_tube : float, optional
        Tube lens glass index.  Default 1.5.
    thickness_obj : float, optional
        Objective centre thickness (mm).  Default 2.
    thickness_tube : float, optional
        Tube lens centre thickness (mm).  Default 5.
    pupil_semi : float, optional
        Entrance-pupil semi-aperture (mm).  Default 3.
    wavelengths : tuple, optional
        Design wavelengths (µm).  Default ``(0.55,)``.
    backend : Backend, optional
        Backend to bind.  Defaults to NumpyBackend.

    Returns
    -------
    System
        Infinity-corrected microscope system.
    """
    from trace_light.lenses import biconvex

    b = SystemBuilder(wavelengths=wavelengths)
    b.add(*biconvex(R=f_obj, n=n_obj, thickness=thickness_obj))
    b.gap(f_obj + f_tube - thickness_obj)
    b.add(*biconvex(R=f_tube, n=n_tube, thickness=thickness_tube))
    b.gap(f_tube - thickness_tube)
    b.image()
    b._stop_z = 0.0
    b._stop_semi = pupil_semi
    return b.finalize(backend=backend)


def relay(
    *,
    f: float = 50.0,
    n: float = 1.5,
    thickness: float = 5.0,
    n_lenses: int = 2,
    pupil_semi: float = 5.0,
    wavelengths: tuple = (0.55,),
    backend: Backend | None = None,
) -> System:
    """Create a relay lens system of *n_lenses* identical biconvex elements.

    Parameters
    ----------
    f : float, optional
        Focal length of each element (mm).  Default 50.
    n : float, optional
        Glass refractive index.  Default 1.5.
    thickness : float, optional
        Centre thickness of each element (mm).  Default 5.
    n_lenses : int, optional
        Number of lens elements.  Default 2.
    pupil_semi : float, optional
        Entrance-pupil semi-aperture (mm).  Default 5.
    wavelengths : tuple, optional
        Design wavelengths (µm).  Default ``(0.55,)``.
    backend : Backend, optional
        Backend to bind.  Defaults to NumpyBackend.

    Returns
    -------
    System
        Relay lens system.
    """
    from trace_light.lenses import biconvex

    b = SystemBuilder(wavelengths=wavelengths)
    for i in range(n_lenses):
        b.add(*biconvex(R=f, n=n, thickness=thickness))
        if i < n_lenses - 1:
            b.gap(2 * f - thickness)
    b.gap(f - thickness)
    b.image()
    b._stop_z = 0.0
    b._stop_semi = pupil_semi
    return b.finalize(backend=backend)


def telescope(
    *,
    f_obj: float = 500.0,
    f_eye: float = 50.0,
    n: float = 1.5,
    thickness: float = 10.0,
    pupil_semi: float = 25.0,
    wavelengths: tuple = (0.55,),
    backend: Backend | None = None,
) -> System:
    """Create a simple Keplerian telescope (objective + eyepiece).

    The back focal plane of the objective coincides with the front focal plane
    of the eyepiece, making the system afocal.

    Parameters
    ----------
    f_obj : float, optional
        Objective focal length (mm).  Default 500.
    f_eye : float, optional
        Eyepiece focal length (mm).  Default 50.
    n : float, optional
        Glass refractive index for both elements.  Default 1.5.
    thickness : float, optional
        Centre thickness of each element (mm).  Default 10.
    pupil_semi : float, optional
        Entrance-pupil semi-aperture (mm).  Default 25.
    wavelengths : tuple, optional
        Design wavelengths (µm).  Default ``(0.55,)``.
    backend : Backend, optional
        Backend to bind.  Defaults to NumpyBackend.

    Returns
    -------
    System
        Afocal Keplerian telescope.
    """
    from trace_light.lenses import biconvex

    b = SystemBuilder(wavelengths=wavelengths)
    b.add(*biconvex(R=f_obj, n=n, thickness=thickness))
    # Shared focal plane between objective back-focus and eyepiece front-focus
    b.gap(f_obj + f_eye - thickness)
    b.add(*biconvex(R=f_eye, n=n, thickness=thickness))
    # No image plane for afocal; place image at exit pupil distance
    b.gap(f_eye - thickness)
    b.image()
    b._stop_z = 0.0
    b._stop_semi = pupil_semi
    return b.finalize(backend=backend)
