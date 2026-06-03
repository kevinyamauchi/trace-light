"""4f relay: thick-lens defocus visualisation.

Left panel  — meridional ray diagram.
  Solid line  : z_actual (actual best-focus plane, where the spot is measured).
  Dashed line : z_thin   (thin-lens formula prediction, i.e. system.image_z).
Right panel — spot diagram at z_actual, rays colored to match.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from optisketch.analysis.spot import spot
from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.sources import emit, point_source
from optisketch.systems import four_f

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------
F = 100.0  # focal length (mm)
THICKNESS = 10.0  # lens centre thickness (mm)
PUPIL_SEMI = 10.0  # entrance-pupil semi-aperture (mm)
N_RAYS = 21  # fan rays (odd → chief ray at y=0)
Z_OBJECT = -F  # source at front focal plane of lens 1

system = four_f(f1=F, f2=F, n=1.5, thickness=THICKNESS, pupil_semi=PUPIL_SEMI)
be = system.backend

z_thin = system.image_z  # thin-lens prediction: f1 + 2*f2 = 300 mm

# ---------------------------------------------------------------------------
# Trace through the system
# ---------------------------------------------------------------------------
src = point_source(
    field_xy=(0.0, 0.0),
    z_object=Z_OBJECT,
    pupil_pattern="fan",
    n_samples=N_RAYS,
)
rays_init = emit(src, system)
final_rays, history = _trace_surfaces(rays_init, system.structure, system.params, be)

# pts: (n_steps, n_rays, 3)  — axis-2 is [x, y, z]
pts = np.stack([be.to_numpy(h) for h in history], axis=0)

# ---------------------------------------------------------------------------
# Find z_actual: sweep z and minimise RMS spot radius
# ---------------------------------------------------------------------------
z_scan = np.linspace(z_thin - 20.0, z_thin + 20.0, 201)
rms_vals = np.array(
    [
        spot(_propagate_to_plane(final_rays, float(z), be), backend=be).rms
        for z in z_scan
    ]
)
z_actual = float(z_scan[np.argmin(rms_vals)])

# Propagate to z_actual for the spot diagram
rays_actual = _propagate_to_plane(final_rays, z_actual, be)
x_spot = be.to_numpy(rays_actual.x)
y_spot = be.to_numpy(rays_actual.y)
valid = be.to_numpy(rays_actual.valid).astype(bool)

# ---------------------------------------------------------------------------
# Build full ray-path arrays for plotting
# source (single point) → entrance pupil → surfaces → z_actual
# ---------------------------------------------------------------------------
# Source point: all rays originate at (x=0, y=0, z=Z_OBJECT)
source_pts = np.zeros((1, N_RAYS, 3))
source_pts[0, :, 2] = Z_OBJECT

# Final positions at z_actual
final_pts = np.stack(
    [
        be.to_numpy(rays_actual.x),
        be.to_numpy(rays_actual.y),
        be.to_numpy(rays_actual.z),
    ],
    axis=-1,
)[np.newaxis]  # (1, n_rays, 3)

paths = np.concatenate([source_pts, pts, final_pts], axis=0)

# ---------------------------------------------------------------------------
# Colors: pupil y-position mapped to colormap (symmetric around 0)
# ---------------------------------------------------------------------------
M_vals = be.to_numpy(rays_init.M)
M_max = np.abs(M_vals).max()
cmap = plt.cm.berlin
colors = cmap(0.5 + M_vals / (2.0 * M_max))

# ---------------------------------------------------------------------------
# Lens element geometry for drawing
# ---------------------------------------------------------------------------
surf_z = be.to_numpy(system.params.z)
# four_f surfaces: [lens1_front, lens1_back, lens2_front, lens2_back]
lens_spans = [(surf_z[0], surf_z[1]), (surf_z[2], surf_z[3])]
lens_half_h = PUPIL_SEMI * 1.15  # draw lenses slightly taller than the beam

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_ray, ax_spot) = plt.subplots(1, 2, figsize=(13, 5))

# ---- Left: ray diagram ----
for r in range(N_RAYS):
    ax_ray.plot(
        paths[:, r, 2],  # z
        paths[:, r, 1],  # y
        color=colors[r],
        lw=0.9,
        alpha=0.85,
        solid_capstyle="round",
    )

for z0, z1 in lens_spans:
    ax_ray.add_patch(
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
    ax_ray.plot([z0, z0], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)
    ax_ray.plot([z1, z1], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)

ax_ray.axvline(
    z_actual,
    color="black",
    lw=1.5,
    ls="-",
    label=f"z_actual = {z_actual:.1f} mm",
)
ax_ray.axvline(
    z_thin,
    color="dimgray",
    lw=1.2,
    ls="--",
    label=f"z_thin = {z_thin:.1f} mm",
)

ax_ray.set_xlim(Z_OBJECT - 10, z_actual + 15)
ax_ray.set_xlabel("z (mm)")
ax_ray.set_ylabel("y (mm)")
ax_ray.set_title("Ray diagram (meridional)")
ax_ray.legend(loc="upper left", fontsize=8)

# ---- Right: spot diagram ----
ax_spot.scatter(
    x_spot[valid],
    y_spot[valid],
    c=colors[valid],
    s=40,
    linewidths=0.4,
    edgecolors="k",
    zorder=3,
)
ax_spot.axhline(0, color="lightgray", lw=0.8, zorder=1)
ax_spot.axvline(0, color="lightgray", lw=0.8, zorder=1)
ax_spot.set_aspect("equal")
ax_spot.set_xlabel("x (mm)")
ax_spot.set_ylabel("y (mm)")
ax_spot.set_title(f"Spot diagram  z = {z_actual:.1f} mm")

defocus_mm = z_actual - z_thin
fig.suptitle(
    f"4f relay (thick biconvex)  |  "
    f"defocus: {defocus_mm:+.2f} mm from thin-lens prediction",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("4f_raytrace.png", dpi=150, bbox_inches="tight")
plt.show()
