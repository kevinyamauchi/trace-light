"""Phase 7 viz tests (IMPLEMENTATION_PLAN §12) — smoke tests only."""

from __future__ import annotations

import pytest

from trace_light.sources import point_source
from trace_light.systems import four_f
from trace_light.viz import layout


def test_layout_2d_returns_figure(backend):
    """layout(dim=2) returns a matplotlib Figure without error."""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=9, pupil_pattern="fan")
    fig = layout(sys, src, dim=2)
    assert isinstance(fig, Figure)


def test_layout_3d_returns_figure(backend):
    """layout(dim=3) returns a matplotlib Figure without error."""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    be = backend
    sys = four_f(f1=100.0, f2=100.0, pupil_semi=5.0, backend=be)
    src = point_source((0.0, 0.0), z_object=-100.0, n_samples=7)
    fig = layout(sys, src, dim=3)
    assert isinstance(fig, Figure)
