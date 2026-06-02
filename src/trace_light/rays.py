"""Core data structures for rays and surface parameters.

User-facing types: :class:`Rays`, :class:`Surface`, :class:`System`.
Implementation-internal: :class:`_Structure`, :class:`_Params`.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import os


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


# ---------------------------------------------------------------------------
# User-facing construction types (Phase 2)
# ---------------------------------------------------------------------------

_INF_SENTINEL = "__inf__"


class Surface(NamedTuple):
    """User-authored single optical surface.

    Surfaces are assembled into a :class:`System` via :class:`SystemBuilder`
    or the ``systems.*`` prefabs.  Use ``radius=math.inf`` to indicate a flat
    plane.

    Attributes
    ----------
    z : float
        Absolute z-position of the surface vertex (mm).
    radius : float
        Radius of curvature (mm).  ``math.inf`` → flat plane.
    conic : float
        Conic constant *k*.  Zero gives a sphere.
    n1 : float
        Refractive index of the medium **before** (incident side).
    n2 : float
        Refractive index of the medium **after** (transmitting side).
    semi_aperture : float
        Semi-aperture radius (mm).  ``math.inf`` disables clipping.
    reflective : bool
        When True the surface acts as a mirror (reflection instead of
        refraction).
    """

    z: float
    radius: float
    conic: float = 0.0
    n1: float = 1.0
    n2: float = 1.0
    semi_aperture: float = math.inf
    reflective: bool = False


def _inf_to_sentinel(v: float) -> float | str:
    """Replace math.inf with a JSON-safe sentinel string."""
    return _INF_SENTINEL if math.isinf(v) else v


def _sentinel_to_inf(v: Any) -> float:
    """Restore math.inf from its sentinel string."""
    return math.inf if v == _INF_SENTINEL else float(v)


class System(NamedTuple):
    """Compiled optical system ready for ray tracing.

    Obtain a :class:`System` from :meth:`SystemBuilder.finalize` or a
    ``systems.*`` prefab.  Never construct directly.

    Attributes
    ----------
    structure : _Structure
        Static (hashable) surface geometry — controls JAX recompilation.
    params : _Params
        Traced numerical parameters (radii, conics, indices).
    pupil_z : float
        z-position of the entrance pupil (mm).
    pupil_semi : float
        Semi-aperture of the entrance pupil (mm).
    image_z : float
        z-position of the image plane (mm).
    wavelengths : tuple of float
        Design wavelengths (µm).
    backend : Backend
        Array-computation backend bound at construction.
    """

    structure: _Structure
    params: _Params
    pupil_z: float
    pupil_semi: float
    image_z: float
    wavelengths: tuple
    backend: Any  # Backend instance — excluded from serialization

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise this system to a plain-Python dictionary.

        Arrays are converted to plain lists via ``backend.to_numpy().tolist()``.
        ``math.inf`` values are replaced with the string ``"__inf__"`` so the
        result is JSON-safe.  The backend is **not** stored; supply it when
        calling :meth:`from_dict`.

        Returns
        -------
        dict
            JSON-serialisable representation of the system.
        """
        be = self.backend
        to_np = be.to_numpy

        structure_dict = {
            "n_surfaces": self.structure.n_surfaces,
            "is_plane": list(self.structure.is_plane),
            "reflective": list(self.structure.reflective),
            "semi_apertures": [
                _inf_to_sentinel(v) for v in self.structure.semi_apertures
            ],
            "z": list(self.structure.z),
        }
        params_dict = {
            "radii": [_inf_to_sentinel(v) for v in to_np(self.params.radii).tolist()],
            "conics": to_np(self.params.conics).tolist(),
            "n1": to_np(self.params.n1).tolist(),
            "n2": to_np(self.params.n2).tolist(),
        }
        return {
            "schema_version": 1,
            "structure": structure_dict,
            "params": params_dict,
            "pupil_z": self.pupil_z,
            "pupil_semi": _inf_to_sentinel(self.pupil_semi),
            "image_z": self.image_z,
            "wavelengths": list(self.wavelengths),
        }

    @classmethod
    def from_dict(cls, data: dict, *, backend: Any = None) -> System:
        """Reconstruct a :class:`System` from a serialised dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as returned by :meth:`to_dict` or parsed from a JSON
            file written by :func:`save_system`.
        backend : Backend, optional
            Backend to bind.  Defaults to :class:`~trace_light.backends.NumpyBackend`.

        Returns
        -------
        System
            Reconstructed system ready for ray tracing.
        """
        if backend is None:
            from trace_light.backends._numpy import NumpyBackend

            backend = NumpyBackend()

        sd = data["structure"]
        pd = data["params"]

        structure = _Structure(
            n_surfaces=sd["n_surfaces"],
            is_plane=tuple(sd["is_plane"]),
            reflective=tuple(sd["reflective"]),
            semi_apertures=tuple(_sentinel_to_inf(v) for v in sd["semi_apertures"]),
            z=tuple(sd["z"]),
        )
        import numpy as _np

        params = _Params(
            radii=backend.asarray(
                _np.array([_sentinel_to_inf(v) for v in pd["radii"]])
            ),
            conics=backend.asarray(_np.array(pd["conics"])),
            n1=backend.asarray(_np.array(pd["n1"])),
            n2=backend.asarray(_np.array(pd["n2"])),
        )
        return cls(
            structure=structure,
            params=params,
            pupil_z=float(data["pupil_z"]),
            pupil_semi=_sentinel_to_inf(data["pupil_semi"]),
            image_z=float(data["image_z"]),
            wavelengths=tuple(data["wavelengths"]),
            backend=backend,
        )


def save_system(system: System, path: str | os.PathLike[str]) -> None:
    """Serialise *system* to a JSON file at *path*.

    Parameters
    ----------
    system : System
        The compiled optical system to save.
    path : str or path-like
        Destination file path.  The file is created or overwritten.

    Returns
    -------
    None
    """

    class _Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            """Encode numpy arrays as plain lists."""
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w") as fh:
        json.dump(system.to_dict(), fh, cls=_Encoder, indent=2)


def load_system(
    path: str | os.PathLike[str],
    *,
    backend: Any = None,
) -> System:
    """Load a :class:`System` from a JSON file written by :func:`save_system`.

    Parameters
    ----------
    path : str or path-like
        Path to the JSON file.
    backend : Backend, optional
        Backend to bind to the loaded system.  Defaults to
        :class:`~trace_light.backends.NumpyBackend`.

    Returns
    -------
    System
        Reconstructed system.
    """
    with open(path) as fh:
        data = json.load(fh)
    return System.from_dict(data, backend=backend)
