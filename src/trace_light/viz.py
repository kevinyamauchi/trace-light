"""Visualization helpers (Phase 7).

:func:`layout` renders the ray paths of a traced system as a 2-D y-z fan or a
3-D path plot, reading positions from the trace history. It is smoke-tested
only; matplotlib is imported lazily so the package never requires it at import
time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from trace_light.kernels import _trace_surfaces
from trace_light.sources import emit

if TYPE_CHECKING:
    from trace_light.rays import System
    from trace_light.sources import Source


def layout(system: System, source: Source, dim: int = 2) -> Any:
    """Render a ray-trace layout diagram for *system* illuminated by *source*.

    The source is emitted and traced, and the per-surface position history is
    drawn as connected ray paths. In 2-D the y-z meridional plane is shown; in
    3-D the full ``(x, y, z)`` paths are drawn.

    Parameters
    ----------
    system : System
        Optical system to visualise.
    source : Source
        Ray source to emit and trace.
    dim : int, optional
        ``2`` for a meridional y-z plot (default) or ``3`` for a 3-D plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the layout.

    Raises
    ------
    ValueError
        If *dim* is not 2 or 3.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}.")

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    be = system.backend
    rays = emit(source, system)
    _, history = _trace_surfaces(rays, system.structure, system.params, be)

    # history: list of (n_rays, 3) arrays → stack to (n_steps, n_rays, 3)
    pts = np.stack([be.to_numpy(h) for h in history], axis=0)
    _n_steps, n_rays, _ = pts.shape

    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
        for r in range(n_rays):
            ax.plot(pts[:, r, 2], pts[:, r, 1], "-", lw=0.5)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Ray layout (meridional y-z)")
    else:
        ax = fig.add_subplot(111, projection="3d")
        for r in range(n_rays):
            ax.plot(pts[:, r, 2], pts[:, r, 0], pts[:, r, 1], "-", lw=0.5)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_zlabel("y (mm)")
        ax.set_title("Ray layout (3-D)")

    return fig
