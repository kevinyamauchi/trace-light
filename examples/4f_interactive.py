# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.8",
# ]
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
    # 4f relay — thick-lens defocus explorer

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

    from optisketch.analysis.spot import spot as compute_spot
    from optisketch.kernels import _propagate_to_plane as propagate_to_plane
    from optisketch.kernels import _trace_surfaces as trace_surfaces
    from optisketch.sources import emit, point_source
    from optisketch.systems import four_f

    return (
        Rectangle,
        anywidget,
        cm,
        compute_spot,
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
def _(
    compute_spot,
    emit,
    four_f,
    np,
    point_source,
    propagate_to_plane,
    trace_surfaces,
):
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


if __name__ == "__main__":
    app.run()
