"""4f relay: numerical aperture and resolution.

Left panel   — meridional ray diagram with aperture stop at the Fourier plane.
               Blue rays pass the stop; red rays are blocked.
Middle panel — bar-pattern image convolved with the Airy PSF at the chosen NA.
Right panel  — ground-truth bar pattern (no blurring).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.signal import fftconvolve
from scipy.special import j1

from optisketch.kernels import _propagate_to_plane, _trace_surfaces
from optisketch.lenses import aperture, biconvex
from optisketch.sources import emit, point_source
from optisketch.systems import SystemBuilder

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
F = 100.0  # focal length (mm)
THICKNESS = 10.0  # lens centre thickness (mm)
N_GLASS = 1.5  # lens refractive index
PUPIL_SEMI = 10.0  # full-aperture semi-diameter (mm)
N_RAYS = 31  # fan rays (odd → chief ray at y=0)
Z_OBJECT = -F  # source at front focal plane of L1
WAVELENGTH = 5.5e-4  # 550 nm in mm

NA_DEMO = 0.05  # NA for the middle panel (r_stop = 5 mm)

PIXEL_MM = 1e-3  # 1 µm per pixel
IMG_SIZE = 256  # 256 x 256 pixels

# ---------------------------------------------------------------------------
# Build 4f system with aperture stop at the Fourier plane
# ---------------------------------------------------------------------------
r_stop = NA_DEMO * F  # aperture radius = 5.0 mm
half_gap = (F + F - THICKNESS) / 2  # 95 mm — half the inter-lens space

b = SystemBuilder()
b.add(*biconvex(R=F, n=N_GLASS, thickness=THICKNESS))  # L1 (z=0 … 10)
b.gap(half_gap)  # to Fourier plane
b.add(*aperture(semi=r_stop))  # aperture stop (z=105)
b.gap(half_gap)  # to L2
b.add(*biconvex(R=F, n=N_GLASS, thickness=THICKNESS))  # L2 (z=200 … 210)
b.gap(F - THICKNESS)  # to image (z=300)
b.image()
b._stop_z = 0.0  # entrance pupil at front of L1
b._stop_semi = PUPIL_SEMI

system = b.finalize()
be = system.backend

# Surface z-positions: [L1_front, L1_back, aperture, L2_front, L2_back]
surf_z = be.to_numpy(system.params.z)
fourier_z = surf_z[2]  # z of the aperture stop

# ---------------------------------------------------------------------------
# Trace fan rays through the full entrance aperture
# ---------------------------------------------------------------------------
src = point_source(
    field_xy=(0.0, 0.0),
    z_object=Z_OBJECT,
    pupil_pattern="fan",
    n_samples=N_RAYS,
)
rays_init = emit(src, system)
final_rays, history = _trace_surfaces(rays_init, system.structure, system.params, be)

# history: 6 entries — [source, L1f, L1b, aperture, L2f, L2b]
pts = np.stack([be.to_numpy(h) for h in history], axis=0)  # (6, N_RAYS, 3)

passed = be.to_numpy(final_rays.valid).astype(bool)

# Propagate (valid) rays to the image plane for the final segment
z_image = system.image_z
rays_at_image = _propagate_to_plane(final_rays, z_image, be)
final_pts = np.stack(
    [
        be.to_numpy(rays_at_image.x),
        be.to_numpy(rays_at_image.y),
        np.full(N_RAYS, z_image),
    ],
    axis=-1,
)[np.newaxis]  # (1, N_RAYS, 3)

# paths index: 0=source, 1=L1f, 2=L1b, 3=aperture(Fourier), 4=L2f, 5=L2b, 6=image
paths = np.concatenate([pts, final_pts], axis=0)  # (7, N_RAYS, 3)

# ---------------------------------------------------------------------------
# Lens geometry
# ---------------------------------------------------------------------------
lens_spans = [(surf_z[0], surf_z[1]), (surf_z[3], surf_z[4])]
lens_half_h = PUPIL_SEMI * 1.15  # draw lenses slightly taller than the beam


# ---------------------------------------------------------------------------
# Airy disk PSF
# ---------------------------------------------------------------------------
def _airy_psf(
    na: float, wavelength_mm: float, pixel_mm: float, half_size: int
) -> np.ndarray:
    y, x = np.ogrid[-half_size : half_size + 1, -half_size : half_size + 1]
    r_mm = np.hypot(x, y) * pixel_mm
    v = 2.0 * np.pi * na * r_mm / wavelength_mm
    v_safe = np.where(v < 1e-10, 1.0, v)  # avoid 0/0 in both np.where branches
    kernel = np.where(v < 1e-10, 1.0, (2.0 * j1(v_safe) / v_safe) ** 2)
    return kernel / kernel.sum()


airy_r_px = 0.61 * WAVELENGTH / (NA_DEMO * PIXEL_MM)  # Airy radius in pixels
half_k = max(int(4.0 * airy_r_px), 16)
psf = _airy_psf(NA_DEMO, WAVELENGTH, PIXEL_MM, half_k)


# ---------------------------------------------------------------------------
# Resolution test pattern — vertical bars in 4 horizontal bands
# ---------------------------------------------------------------------------
def _bar_pattern(size: int) -> np.ndarray:
    img = np.zeros((size, size), dtype=float)
    periods_px = [4, 8, 16, 32]
    band_h = size // len(periods_px)
    for i, p in enumerate(periods_px):
        y0, y1 = i * band_h, (i + 1) * band_h
        for col in range(size):
            if (col % p) < p // 2:
                img[y0:y1, col] = 1.0
    return img


test_image = _bar_pattern(IMG_SIZE)
img_demo = fftconvolve(test_image, psf, mode="same")
img_demo = np.clip(img_demo, 0.0, None)
img_demo /= img_demo.max()

# ---------------------------------------------------------------------------
# Scale bar overlay helper
# ---------------------------------------------------------------------------
_OUTLINE = [pe.withStroke(linewidth=2, foreground="black")]


def _scale_bar(ax: plt.Axes, bar_um: float = 20.0) -> None:
    bar_px = bar_um / (PIXEL_MM * 1e3)  # µm → pixels
    x0 = IMG_SIZE * 0.05
    y0 = IMG_SIZE * 0.92
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_ray, ax_demo, ax_gt) = plt.subplots(1, 3, figsize=(15, 5))

# ---- Left: ray diagram ----
for r in range(N_RAYS):
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
        # Draw only source → L1f → L1b → aperture (stop here)
        ax_ray.plot(
            paths[:4, r, 2],
            paths[:4, r, 1],
            color="crimson",
            lw=0.9,
            alpha=0.85,
            solid_capstyle="round",
        )

# Lens rectangles
for z0, z1 in lens_spans:
    ax_ray.add_patch(
        Rectangle(
            xy=(z0, -lens_half_h),
            width=z1 - z0,
            height=2 * lens_half_h,
            facecolor="steelblue",
            alpha=0.20,
            lw=0,
        )
    )
    ax_ray.plot([z0, z0], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)
    ax_ray.plot([z1, z1], [-lens_half_h, lens_half_h], color="steelblue", lw=1.3)

# Aperture stop blades (black bars covering the blocked region)
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

# Aperture stop label
ax_ray.annotate(
    f"Aperture stop\nr = {r_stop:.0f} mm · NA = {NA_DEMO:.2f}",
    xy=(fourier_z, r_stop),
    xytext=(fourier_z + 12, r_stop + 4),
    fontsize=7,
    va="bottom",
    arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
)

# Image plane marker
ax_ray.axvline(z_image, color="dimgray", lw=1.0, ls="--")
ax_ray.text(
    z_image + 1,
    -lens_half_h + 0.5,
    f"Image\n{z_image:.0f} mm",
    fontsize=6,
    va="bottom",
)

ax_ray.set_xlim(Z_OBJECT - 10, z_image + 18)
ax_ray.set_ylim(-lens_half_h - 1, lens_half_h + 8)
ax_ray.set_xlabel("z (mm)")
ax_ray.set_ylabel("y (mm)")
ax_ray.set_title(f"Ray diagram  |  NA = {NA_DEMO:.2f}  (r_stop = {r_stop:.0f} mm)")
ax_ray.legend(
    handles=[
        mpatches.Patch(facecolor="royalblue", label="Passed aperture"),
        mpatches.Patch(facecolor="crimson", label="Blocked"),
    ],
    loc="upper left",
    fontsize=7,
)

# ---- Middle: image at NA_DEMO ----
ax_demo.imshow(img_demo, cmap="gray", origin="upper", vmin=0, vmax=1)
_scale_bar(ax_demo)
ax_demo.set_title(f"Image  |  NA = {NA_DEMO:.2f}  (λ = 550 nm)")
ax_demo.axis("off")

# Bar-spacing labels
_periods_um = [4, 8, 16, 32]
_band_h = IMG_SIZE // len(_periods_um)
for i, p in enumerate(_periods_um):
    ax_demo.text(
        IMG_SIZE - 2,
        (i + 0.5) * _band_h,
        f"{p} µm",
        color="cyan",
        ha="right",
        va="center",
        fontsize=10,
        path_effects=_OUTLINE,
    )

# ---- Right: ground truth ----
ax_gt.imshow(test_image, cmap="gray", origin="upper", vmin=0, vmax=1)
_scale_bar(ax_gt)
ax_gt.set_title("Ground truth")
ax_gt.axis("off")

for i, p in enumerate(_periods_um):
    ax_gt.text(
        IMG_SIZE - 2,
        (i + 0.5) * _band_h,
        f"{p} µm",
        color="cyan",
        ha="right",
        va="center",
        fontsize=10,
        path_effects=_OUTLINE,
    )

fig.suptitle(
    f"4f relay  ·  Numerical aperture and resolution  ·  F = {F:.0f} mm  ·  λ = 550 nm",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("na_demo.png", dpi=150, bbox_inches="tight")
plt.show()
