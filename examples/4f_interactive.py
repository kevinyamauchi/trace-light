# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.8",
# ]
#
# [tool.marimo.runtime]
# auto_instantiate = true
# ///
"""Interactive demonstration of interactive raytracing.

This is a marimo notebook that can be rendered to HTML + WASM.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive notebook demo

    This is a demo of an interactive Python notebook that can be run in a web browser.
    The first demo is an interactive ray tracing simulation that is controlled entirely
    through the GUI. The second demo shows how the user can update and execute Python
    code.

    **To start the demo**, press the big play button in the
    bottom right corner of the window.

    ## 4f relay — thick-lens defocus explorer

    A **4f relay** consists of two lenses separated by the sum of their focal lengths.
    For an ideal (thin) lens the image forms at a predictable plane **z_thin**.
    Real thick lenses introduce a small **defocus**: the true best-focus plane
    **z_actual** differs from position predicted by the thin lens approximation
    **z_thin**.

    Use the slider below to move the measurement plane and observe how the spot
    diagram changes. More focused points would result in a sharper image.
    The **green** line and border mark z_actual; the **magenta**
    line and border mark the thin-lens prediction z_thin. Dashed magenta lines on
    the spot diagrams show the y-extent of the z_thin spot for easy comparison.

    > System: biconvex lenses, f = 100 mm, centre thickness = 10 mm, n = 1.5
    """)
    return


@app.cell(hide_code=True)
async def _():
    import sys

    import marimo as mo
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    import traitlets
    from matplotlib.patches import Rectangle

    if sys.platform == "emscripten":
        import micropip

        # anywidget and optisketch are not in Pyodide; install before importing.
        # deps=False: numpy/scipy/matplotlib are already provided by Pyodide.
        await micropip.install("anywidget")
        await micropip.install("optisketch", deps=False)

    import anywidget

    from optisketch.kernels import _propagate_to_plane as propagate_to_plane
    from optisketch.kernels import _trace_surfaces as trace_surfaces
    from optisketch.sources import emit, point_source
    from optisketch.systems import four_f

    return (
        Rectangle,
        anywidget,
        cm,
        emit,
        four_f,
        mo,
        np,
        plt,
        point_source,
        propagate_to_plane,
        trace_surfaces,
        traitlets,
    )


@app.cell(hide_code=True)
def _(emit, four_f, np, point_source, propagate_to_plane, trace_surfaces):
    F = 100.0
    THICKNESS = 10.0
    PUPIL_SEMI = 10.0
    N_RAYS = 21
    Z_OBJECT = -F

    system = four_f(f1=F, f2=F, n=1.5, thickness=THICKNESS, pupil_semi=PUPIL_SEMI)
    be = system.backend
    z_thin = system.image_z

    src = point_source(
        field_xy=(0.0, 0.0),
        z_object=Z_OBJECT,
        pupil_pattern="fan",
        n_samples=N_RAYS,
    )
    rays_init = emit(src, system)
    final_rays, history = trace_surfaces(rays_init, system.structure, system.params, be)

    # pts shape: (n_surfaces+1, n_rays, 3)
    pts = np.stack([be.to_numpy(h) for h in history], axis=0)

    # Source origin for ray diagram (all rays start at same point)
    source_pts = np.zeros((1, N_RAYS, 3))
    source_pts[0, :, 2] = Z_OBJECT

    # Lens geometry for drawing
    surf_z = be.to_numpy(system.params.z)
    lens_spans = [(surf_z[0], surf_z[1]), (surf_z[2], surf_z[3])]
    lens_half_h = PUPIL_SEMI * 1.15

    # Ray colors: pupil y-position mapped to colormap
    M_vals = be.to_numpy(rays_init.M)
    M_max = float(np.abs(M_vals).max())

    # Start slider at the thin-lens prediction; user explores to find true best focus
    z_best = float(z_thin)

    # Static: propagate to z_thin once
    rays_at_thin = propagate_to_plane(final_rays, float(z_thin), be)
    x_thin = be.to_numpy(rays_at_thin.x)
    y_thin = be.to_numpy(rays_at_thin.y)
    valid_thin = be.to_numpy(rays_at_thin.valid).astype(bool)
    return (
        M_max,
        M_vals,
        N_RAYS,
        be,
        final_rays,
        lens_half_h,
        lens_spans,
        pts,
        source_pts,
        valid_thin,
        x_thin,
        y_thin,
        z_best,
        z_thin,
    )


@app.cell(hide_code=True)
def _(anywidget, mo, traitlets, z_best, z_thin):
    class ZSlider(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const wrap = document.createElement("div");
            wrap.style.cssText = [
                "display:flex",
                "align-items:center",
                "gap:10px",
                "padding:6px 12px",
                "font-family:monospace",
                "font-size:13px",
            ].join(";");

            const lbl = document.createElement("span");
            lbl.textContent = "z_actual";
            lbl.style.cssText = "font-weight:bold;white-space:nowrap";

            const minLbl = document.createElement("span");
            minLbl.style.color = "#888";

            const slider = document.createElement("input");
            slider.type = "range";
            slider.style.cssText = "flex:1;cursor:pointer";

            const maxLbl = document.createElement("span");
            maxLbl.style.color = "#888";

            const valLbl = document.createElement("span");
            valLbl.style.cssText = "min-width:72px;text-align:right"
                                 + ";font-variant-numeric:tabular-nums";

            function sync() {
                const min  = model.get("min");
                const max  = model.get("max");
                const step = model.get("step");
                const val  = model.get("value");
                slider.min   = min;
                slider.max   = max;
                slider.step  = step;
                slider.value = val;
                minLbl.textContent = min.toFixed(1);
                maxLbl.textContent = max.toFixed(1);
                valLbl.textContent = val.toFixed(2) + " mm";
            }
            sync();

            slider.addEventListener("input", () => {
                const v = parseFloat(slider.value);
                valLbl.textContent = v.toFixed(2) + " mm";
                model.set("value", v);
                model.save_changes();
            });

            model.on("change:value", sync);
            model.on("change:min",   sync);
            model.on("change:max",   sync);
            model.on("change:step",  sync);

            wrap.append(lbl, minLbl, slider, maxLbl, valLbl);
            el.appendChild(wrap);
        }
        export default { render };
        """

        value = traitlets.Float(0.0).tag(sync=True)
        min = traitlets.Float(0.0).tag(sync=True)
        max = traitlets.Float(1.0).tag(sync=True)
        step = traitlets.Float(0.5).tag(sync=True)

    raw_slider = ZSlider(
        value=z_best,
        min=z_thin - 20.0,
        max=z_thin + 20.0,
        step=0.5,
    )
    slider = mo.ui.anywidget(raw_slider)
    return (slider,)


@app.cell(hide_code=True)
def _(
    M_max,
    M_vals,
    N_RAYS,
    Rectangle,
    be,
    cm,
    final_rays,
    lens_half_h,
    lens_spans,
    mo,
    np,
    plt,
    propagate_to_plane,
    pts,
    slider,
    source_pts,
    valid_thin,
    x_thin,
    y_thin,
    z_thin,
):
    Z_ACTUAL_COLOR = "green"
    Z_THIN_COLOR = "magenta"
    ZOOM_COLOR = "darkorange"
    SPOT_LIM = 2.0  # fixed half-range for spot panels (mm)
    RAY_XLIM = (-110, 330)  # fixed x range for full ray diagram
    ZOOM_XLIM = (270, 330)  # fixed x range for zoomed ray diagram
    ZOOM_YLIM = (-3, 3)  # fixed y range for zoomed ray diagram

    z_val = slider.value["value"]

    # Propagate to current z_actual
    rays_at_z = propagate_to_plane(final_rays, float(z_val), be)
    x_spot = be.to_numpy(rays_at_z.x)
    y_spot = be.to_numpy(rays_at_z.y)
    valid_spot = be.to_numpy(rays_at_z.valid).astype(bool)

    # Full ray paths: source → history points → z_actual
    final_pts = np.stack(
        [
            be.to_numpy(rays_at_z.x),
            be.to_numpy(rays_at_z.y),
            be.to_numpy(rays_at_z.z),
        ],
        axis=-1,
    )[np.newaxis]
    paths = np.concatenate([source_pts, pts, final_pts], axis=0)

    cmap = cm.RdBu_r
    colors = cmap(0.5 + M_vals / (2.0 * M_max))

    # y-extent of z_thin spot — reference lines drawn on both spot panels
    y_thin_min = float(y_thin[valid_thin].min())
    y_thin_max = float(y_thin[valid_thin].max())

    def draw_spot_panel(ax, xs, ys, valid, title, line_color):
        ax.scatter(
            xs[valid],
            ys[valid],
            c=colors[valid],
            s=40,
            linewidths=0.4,
            edgecolors="k",
            zorder=3,
        )
        ax.axhline(0, color="lightgray", lw=0.8, zorder=1)
        ax.axvline(0, color="lightgray", lw=0.8, zorder=1)
        ax.axhline(y_thin_min, color=Z_THIN_COLOR, lw=1.0, ls="--", zorder=2)
        ax.axhline(y_thin_max, color=Z_THIN_COLOR, lw=1.0, ls="--", zorder=2)
        ax.set_xlim(-SPOT_LIM, SPOT_LIM)
        ax.set_ylim(-SPOT_LIM, SPOT_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
            spine.set_linewidth(2.5)

    def draw_ray_diagram(ax, xlim, ylim=None):
        for r in range(N_RAYS):
            ax.plot(
                paths[:, r, 2],
                paths[:, r, 1],
                color=colors[r],
                lw=0.9,
                alpha=0.85,
                solid_capstyle="round",
            )
        for z0, z1 in lens_spans:
            ax.add_patch(
                Rectangle(
                    xy=(z0, -lens_half_h),
                    width=z1 - z0,
                    height=2 * lens_half_h,
                    facecolor="steelblue",
                    edgecolor="steelblue",
                    alpha=0.20,
                    lw=0,
                )
            )
            ax.plot([z0, z0], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)
            ax.plot([z1, z1], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)
        ax.axvline(
            z_val,
            color=Z_ACTUAL_COLOR,
            lw=1.5,
            ls="-",
            label=f"z_actual = {z_val:.1f} mm",
        )
        ax.axvline(
            z_thin,
            color=Z_THIN_COLOR,
            lw=1.2,
            ls="--",
            label=f"z_thin = {z_thin:.1f} mm",
        )
        ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")

    fig, (ax_ray, ax_zoom, ax_actual, ax_thin_ax) = plt.subplots(1, 4, figsize=(19, 5))

    # ── Full ray diagram ──────────────────────────────────────────────────────
    draw_ray_diagram(ax_ray, xlim=RAY_XLIM)
    ax_ray.set_title("Ray diagram (meridional)")
    ax_ray.legend(loc="upper left", fontsize=8)

    # Box on the full diagram showing the zoomed region
    zoom_rect = Rectangle(
        xy=(ZOOM_XLIM[0], ZOOM_YLIM[0]),
        width=ZOOM_XLIM[1] - ZOOM_XLIM[0],
        height=ZOOM_YLIM[1] - ZOOM_YLIM[0],
        facecolor="none",
        edgecolor=ZOOM_COLOR,
        lw=1.5,
        zorder=5,
    )
    ax_ray.add_patch(zoom_rect)

    # ── Zoomed ray diagram ────────────────────────────────────────────────────
    draw_ray_diagram(ax_zoom, xlim=ZOOM_XLIM, ylim=ZOOM_YLIM)
    ax_zoom.set_title("Zoom: focusing region")
    ax_zoom.legend(loc="upper left", fontsize=8)
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(ZOOM_COLOR)
        spine.set_linewidth(2.5)

    # ── Spot at z_actual ─────────────────────────────────────────────────────
    draw_spot_panel(
        ax_actual,
        x_spot,
        y_spot,
        valid_spot,
        f"Spot at z_actual = {z_val:.1f} mm",
        Z_ACTUAL_COLOR,
    )

    # ── Spot at z_thin (static reference) ────────────────────────────────────
    draw_spot_panel(
        ax_thin_ax,
        x_thin,
        y_thin,
        valid_thin,
        f"Spot at z_thin = {z_thin:.1f} mm",
        Z_THIN_COLOR,
    )

    plt.tight_layout()
    mo.vstack([fig, slider])  # cell output — figure above, slider below
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical aperture and resolution

    The **aperture stop** placed at the Fourier plane of a 4f relay controls
    which spatial frequencies pass through the system. Its radius sets the
    **numerical aperture**. Using the small-angle approximation,

    $$\mathrm{NA} = \frac{r_\text{stop}}{F}$$

    A larger NA admits higher spatial frequencies, producing a smaller Airy
    disk and finer resolving power. The Rayleigh resolution limit is:

    $$d = \frac{0.61\,\lambda}{\mathrm{NA}}$$

    **Left** — meridional ray diagram. Blue rays pass the aperture stop;
    red rays are blocked. **Middle** — a bar-pattern target (periods 4-32 µm)
    convolved with the Airy PSF at the chosen NA. **Right** — the ground-truth
    pattern with no blurring. Reduce NA to watch fine bars merge into gray.

    > System: biconvex lenses, f = 100 mm, centre thickness = 10 mm, n = 1.5, λ = 550 nm

    Edit `numerical_aperture` in the cell below and re-run it to see the effect.
    You can run the code cell by pressing the ▶ button or Ctrl/Cmd+Enter.
    """)
    return


@app.cell(hide_code=True)
def _():
    import matplotlib.patches as _mpatches
    import matplotlib.patheffects as _pe
    import matplotlib.pyplot as _plt
    import numpy as _np
    from matplotlib.patches import Rectangle as _Rectangle
    from scipy.signal import fftconvolve as _fftconvolve
    from scipy.special import j1 as _j1

    from optisketch.kernels import _propagate_to_plane as _prop
    from optisketch.kernels import _trace_surfaces as _trace
    from optisketch.lenses import aperture as _aperture_surf
    from optisketch.lenses import biconvex as _biconvex
    from optisketch.sources import emit as _emit
    from optisketch.sources import point_source as _point_source
    from optisketch.systems import SystemBuilder as _SystemBuilder

    _F = 100.0
    _THICKNESS = 10.0
    _N_GLASS = 1.5
    _PUPIL_SEMI = 10.0
    _N_RAYS = 31
    _Z_OBJECT = -_F
    _WAVELENGTH = 5.5e-4
    _PIXEL_MM = 1e-3
    _IMG_SIZE = 256

    def _airy_psf(na, wavelength_mm, pixel_mm, half_size):
        y, x = _np.ogrid[-half_size : half_size + 1, -half_size : half_size + 1]
        r_mm = _np.hypot(x, y) * pixel_mm
        v = 2.0 * _np.pi * na * r_mm / wavelength_mm
        v_safe = _np.where(v < 1e-10, 1.0, v)
        kernel = _np.where(v < 1e-10, 1.0, (2.0 * _j1(v_safe) / v_safe) ** 2)
        return kernel / kernel.sum()

    def _bar_pattern(size):
        img = _np.zeros((size, size), dtype=float)
        periods_px = [4, 8, 16, 32]
        band_h = size // len(periods_px)
        for i, p in enumerate(periods_px):
            y0, y1 = i * band_h, (i + 1) * band_h
            for col in range(size):
                if (col % p) < p // 2:
                    img[y0:y1, col] = 1.0
        return img

    _OUTLINE = [_pe.withStroke(linewidth=2, foreground="black")]

    def _scale_bar(ax, bar_um=20.0):
        bar_px = bar_um / (_PIXEL_MM * 1e3)
        x0 = _IMG_SIZE * 0.05
        y0 = _IMG_SIZE * 0.92
        (line,) = ax.plot(
            [x0, x0 + bar_px], [y0, y0], color="cyan", lw=2, solid_capstyle="butt"
        )
        line.set_path_effects(_OUTLINE)
        ax.text(
            x0 + bar_px / 2,
            y0 - 2,
            f"{bar_um:.0f} µm",
            color="cyan",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            path_effects=_OUTLINE,
        )

    def simulate_image(numerical_aperture: float) -> None:
        """Render the 3-panel NA demo figure for the given numerical aperture."""
        na = float(numerical_aperture)
        if not (0.001 <= na <= 0.1):
            raise ValueError(
                f"numerical_aperture must be between 0.001 and 0.1, got {na}"
            )
        r_stop = na * _F
        half_gap = (_F + _F - _THICKNESS) / 2

        b = _SystemBuilder()
        b.add(*_biconvex(R=_F, n=_N_GLASS, thickness=_THICKNESS))
        b.gap(half_gap)
        b.add(*_aperture_surf(semi=r_stop))
        b.gap(half_gap)
        b.add(*_biconvex(R=_F, n=_N_GLASS, thickness=_THICKNESS))
        b.gap(_F - _THICKNESS)
        b.image()
        b._stop_z = 0.0
        b._stop_semi = _PUPIL_SEMI
        system = b.finalize()
        be = system.backend

        surf_z = be.to_numpy(system.params.z)
        fourier_z = surf_z[2]

        src = _point_source(
            field_xy=(0.0, 0.0),
            z_object=_Z_OBJECT,
            pupil_pattern="fan",
            n_samples=_N_RAYS,
        )
        rays_init = _emit(src, system)
        final_rays, history = _trace(rays_init, system.structure, system.params, be)

        pts = _np.stack([be.to_numpy(h) for h in history], axis=0)
        passed = be.to_numpy(final_rays.valid).astype(bool)

        z_image = system.image_z
        rays_at_image = _prop(final_rays, z_image, be)
        final_pts = _np.stack(
            [
                be.to_numpy(rays_at_image.x),
                be.to_numpy(rays_at_image.y),
                _np.full(_N_RAYS, z_image),
            ],
            axis=-1,
        )[_np.newaxis]
        paths = _np.concatenate([pts, final_pts], axis=0)

        lens_spans = [(surf_z[0], surf_z[1]), (surf_z[3], surf_z[4])]
        lens_half_h = _PUPIL_SEMI * 1.15

        airy_r_px = 0.61 * _WAVELENGTH / (na * _PIXEL_MM)
        half_k = max(int(4.0 * airy_r_px), 16)
        psf = _airy_psf(na, _WAVELENGTH, _PIXEL_MM, half_k)

        test_image = _bar_pattern(_IMG_SIZE)
        img_demo = _fftconvolve(test_image, psf, mode="same")
        img_demo = _np.clip(img_demo, 0.0, None)
        img_demo /= img_demo.max()

        fig, (ax_ray, ax_demo, ax_gt) = _plt.subplots(1, 3, figsize=(15, 5))

        for r in range(_N_RAYS):
            if passed[r]:
                ax_ray.plot(
                    paths[:, r, 2],
                    paths[:, r, 1],
                    color="royalblue",
                    lw=0.9,
                    alpha=0.85,
                    solid_capstyle="round",
                )
            else:
                ax_ray.plot(
                    paths[:4, r, 2],
                    paths[:4, r, 1],
                    color="crimson",
                    lw=0.9,
                    alpha=0.85,
                    solid_capstyle="round",
                )

        for z0, z1 in lens_spans:
            ax_ray.add_patch(
                _Rectangle(
                    xy=(z0, -lens_half_h),
                    width=z1 - z0,
                    height=2 * lens_half_h,
                    facecolor="steelblue",
                    alpha=0.20,
                    lw=0,
                )
            )
            ax_ray.plot(
                [z0, z0], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3
            )
            ax_ray.plot(
                [z1, z1], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3
            )

        ax_ray.plot(
            [fourier_z, fourier_z],
            [r_stop, lens_half_h],
            color="black",
            lw=4,
            solid_capstyle="butt",
        )
        ax_ray.plot(
            [fourier_z, fourier_z],
            [-lens_half_h, -r_stop],
            color="black",
            lw=4,
            solid_capstyle="butt",
        )
        ax_ray.annotate(
            f"Aperture stop\nr = {r_stop:.1f} mm · NA = {na:.3f}",
            xy=(fourier_z, r_stop),
            xytext=(fourier_z + 12, r_stop + 4),
            fontsize=7,
            va="bottom",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
        )
        ax_ray.axvline(z_image, color="dimgray", lw=1.0, ls="--")
        ax_ray.text(
            z_image + 1,
            -lens_half_h + 0.5,
            f"Image\n{z_image:.0f} mm",
            fontsize=6,
            va="bottom",
        )
        ax_ray.set_xlim(_Z_OBJECT - 10, z_image + 18)
        ax_ray.set_ylim(-lens_half_h - 1, lens_half_h + 8)
        ax_ray.set_xlabel("z (mm)")
        ax_ray.set_ylabel("y (mm)")
        ax_ray.set_title(f"Ray diagram  |  NA = {na:.3f}  (r_stop = {r_stop:.1f} mm)")
        ax_ray.legend(
            handles=[
                _mpatches.Patch(facecolor="royalblue", label="Passed aperture"),
                _mpatches.Patch(facecolor="crimson", label="Blocked"),
            ],
            loc="upper left",
            fontsize=7,
        )

        _periods_um = [4, 8, 16, 32]
        _band_h = _IMG_SIZE // len(_periods_um)

        ax_demo.imshow(img_demo, cmap="gray", origin="upper", vmin=0, vmax=1)
        _scale_bar(ax_demo)
        ax_demo.set_title(f"Image  |  NA = {na:.3f}  (λ = 550 nm)")
        ax_demo.axis("off")
        for i, p in enumerate(_periods_um):
            ax_demo.text(
                _IMG_SIZE - 2,
                (i + 0.5) * _band_h,
                f"{p} µm",
                color="cyan",
                ha="right",
                va="center",
                fontsize=14,
                fontweight="bold",
                path_effects=_OUTLINE,
            )

        ax_gt.imshow(test_image, cmap="gray", origin="upper", vmin=0, vmax=1)
        _scale_bar(ax_gt)
        ax_gt.set_title("Ground truth")
        ax_gt.axis("off")
        for i, p in enumerate(_periods_um):
            ax_gt.text(
                _IMG_SIZE - 2,
                (i + 0.5) * _band_h,
                f"{p} µm",
                color="cyan",
                ha="right",
                va="center",
                fontsize=14,
                fontweight="bold",
                path_effects=_OUTLINE,
            )

        fig.suptitle(
            "4f relay  ·  Numerical aperture and resolution"
            f"  ·  F = {_F:.0f} mm  ·  λ = 550 nm",
            fontsize=10,
        )
        _plt.tight_layout()
        _plt.show()

    return (simulate_image,)


@app.cell
def _(simulate_image):
    # Set the numerical aperture below, then run this cell (▶ or Ctrl/Cmd+Enter).
    # Valid range: 0.001 - 0.1
    numerical_aperture = 0.05

    simulate_image(numerical_aperture=numerical_aperture)
    return


if __name__ == "__main__":
    app.run()
