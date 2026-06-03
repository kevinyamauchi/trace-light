"""Phase 7 fluorescence-composition tests (IMPLEMENTATION_PLAN §9 assembly)."""

from __future__ import annotations

import numpy as np

from trace_light.fluorescence import emission_volume, widefield
from trace_light.systems import four_f


def _blob(ny=24, nx=24):
    """Return a (ny, nx) fluorophore map with a centred bright block."""
    f = np.zeros((ny, nx), dtype=np.float64)
    f[ny // 2 - 2 : ny // 2 + 2, nx // 2 - 2 : nx // 2 + 2] = 1.0
    return f


def test_fluorescence_widefield_shapes(backend):
    """Widefield: emission volume → image stack with correct shapes."""
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be)

    # 2-D fluorophore → 2-D image
    fluor2d = be.asarray(_blob(24, 24))
    img2d = widefield(
        sys, fluor2d, extent=0.3, focus=6.0, psf="single", psf_grid=(15, 15), n_rays=200
    )
    assert tuple(be.to_numpy(img2d).shape) == (24, 24)

    # 3-D fluorophore + focal stack
    fluor3d = be.asarray(np.stack([_blob(20, 20) for _ in range(2)], axis=0))
    img3d = widefield(
        sys,
        fluor3d,
        extent=0.3,
        focus=np.array([4.0, 8.0]),
        psf="single",
        psf_grid=(15, 15),
        n_rays=200,
        depth_extent=1.0,
    )
    assert tuple(be.to_numpy(img3d).shape) == (2, 20, 20)


def test_fluorescence_emission_two_photon():
    """Two-photon emission scales as the square of the excitation."""
    fluor = np.ones((4, 4))
    exc = np.full((4, 4), 3.0)
    one = emission_volume(fluor, exc, two_photon=False)
    two = emission_volume(fluor, exc, two_photon=True)
    np.testing.assert_allclose(one, 3.0)
    np.testing.assert_allclose(two, 9.0)


def test_fluorescence_no_spurious_gain(backend):
    """The imaged fluorescence energy never exceeds the emission energy.

    The PSF is normalised to unit sum and ``fftconvolve(mode='same')`` only
    drops energy off the array borders, so the pipeline cannot manufacture
    signal above the input.
    """
    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=3.0, backend=be)
    fluor_np = _blob(24, 24)
    fluor = be.asarray(fluor_np)
    img = be.to_numpy(
        widefield(
            sys,
            fluor,
            extent=0.3,
            focus=6.0,
            psf="single",
            psf_grid=(15, 15),
            n_rays=400,
        )
    )
    emission_energy = float(fluor_np.sum())
    assert img.sum() <= emission_energy * (1.0 + 1e-6)
