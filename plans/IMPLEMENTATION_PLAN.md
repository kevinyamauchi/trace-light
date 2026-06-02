# Implementation Plan: `raytrace_jax`

**Purpose:** A coherent, multiphase plan for building `raytrace_jax` — a minimal,
differentiable, forward-only sequential ray tracer ported in spirit from Optiland.
This document is written to drive incremental implementation (e.g. with Claude Code):
each phase is a self-contained unit of work with a clear scope, dependency set, test
strategy, and a definition-of-done gate.

**Source documents:** `DESIGN.md` (proposed API) and `TEST_PLAN.md` (accuracy tests).
**Reference oracle:** the Optiland codebase (`https://github.com/optiland/optiland.git`).
Section references like "DESIGN §3" or "TEST_PLAN §1.1" point into those documents.

---

## 0. Ground rules that apply to every phase

These conventions are fixed up front so they never have to be retrofitted.

### 0.1 Bottom-up, with the dual backend locked in first

The dependency spine is: **backend → kernels → construction → sources → analysis →
optimize → assembly**. Each phase produces something testable against golden values
before the next layer depends on it. The NumPy/JAX abstraction (Phase 0) comes first
because every later test runs once per backend through it, and the lazy-import
discipline (no top-level `import jax`) is painful to bolt on later.

### 0.2 The four test-porting modes (apply per test)

Every ported test is tagged with how its reference is obtained. This keeps it
unambiguous which numbers are sacred and which are merely patterns.

| Mode | When | How |
|------|------|-----|
| **copy-value** | Optiland asserts against an analytic literal (convention-independent under the standard sag) | Paste the literal verbatim; add a provenance comment naming the Optiland test |
| **copy-math** | Optiland computes the reference inline (physics invariants, `‖d‖=1`, Snell residual) | Port the computation as Python; it is convention-bearing, so recompute don't freeze |
| **copy-array** | Optiland freezes an external-tool (Zemax) reference array | Optiland's own toroidal goldens are out of scope (toroids need the iterative solver); instead **regenerate** goldens by tracing the equivalent sphere/conic system *in the Optiland clone* and pasting the printed arrays, with a committed generator script |
| **model-pattern** | The cited Optiland test targets a subsystem we are not porting (torch optimizer, MTF) | Copy the assertion *structure*; the numbers come from our own system, not Optiland |

Tier-B cross-backend parity tests have **no Optiland analog** and are authored fresh:
run the kernel on both backends with shared input and assert
`|numpy − jax|` at `rtol=1e-11` (TEST_PLAN §0.2).

### 0.3 Provenance discipline

- Each copy-value constant carries a comment: `# optiland tests/<file>::<Class>::<test>`.
- Each copy-array golden has a committed generator script under `golden/` that
  reproduces it from the clone, so a reviewer can regenerate and diff.
- `schema_version` is recorded in any serialized artifact from day one.

### 0.4 Test harness (built in Phase 0, used everywhere)

Three fixtures in `tests/conftest.py` — **conftest is fixtures only; no helpers are
exported from it**:

| Fixture | When to use |
|---------|-------------|
| `numpy_backend` | NumPy-specific tests (flags, graceful degradation) |
| `jax_backend` | JAX-specific tests (jit, vmap, grad, transforms); auto-skips when JAX absent |
| `backend` | Cross-backend accuracy tests; parametrized over `["numpy", "jax"]`; skips jax variant when JAX absent |

Tier-B parity tests take `jax_backend` and instantiate `NumpyBackend()` inline (they
need both backends simultaneously, not parametrization).

Assertion helpers — a private `_ac(a, b, rtol, atol, be)` that funnels both operands
through `be.to_numpy()` before `np.testing.assert_allclose`, plus four one-line tier
wrappers — are defined **inline** at the top of each test file that needs them.  They
are not imported from conftest or any shared module.

The four tolerance tiers from TEST_PLAN §0.2:
  - **A** reference/golden: `rtol=1e-5, atol=1e-7`
  - **B** cross-backend parity: `rtol=1e-11, atol=1e-12`
  - **C** transform parity (jit vs eager, vmap vs loop): `rtol=1e-9, atol=1e-10`
  - **D** exact invariant: `atol=1e-12`

All tests are **module-level functions — no test classes**.  Naming convention:
`test_<kernel>_<case>` for accuracy tests and `test_<group>_<property>` for
capability/flag tests.

### 0.5 Float64 everywhere

OPD/path-length precision requires float64. NumPy is float64 natively; the JAX
backend enables `jax_enable_x64` inside `JaxBackend.__init__` (never at package
import, so NumPy-only/Pyodide environments never touch `jax`).

### 0.6 Dependency graph

```
Phase 0  backends + harness
   │
Phase 1  Rays, kernels, trace        ──┐
   │                                    │ (Optiland math is the oracle)
Phase 2  Surface/System, lenses,        │
         builder, prefabs, SERIALIZE  ──┘
   │
Phase 3  Source, sampling, emit
   │
Phase 4  analysis.spot / psf / irradiance
   │
Phase 5  analysis.image_sim (2-D/3-D, single/varying)
   │
Phase 6  optimize.best_focus / minimize
   │
Phase 7  viz.layout + fluorescence assembly + hardening
```

---

## Phase 0 — Backend abstraction + project skeleton ✅

**Objective.** Stand up the package, the `Backend` protocol, both backends, and the
test harness. Nothing optical happens here; this phase exists so the two-backend
promise is structurally guaranteed before any kernel is written.

**Deliverables.**
- Package skeleton `trace_light/` with the module layout from DESIGN §10
  (`backends`, plus empty stubs for `lenses`, `systems`, `sources`, `analysis`,
  `optimize`, `viz`).
- `Backend` abstract base class (`backends/_protocol.py`): the ~25 array ops the core
  uses (`sqrt, sin, cos, abs, sign, where, minimum/maximum, stack, concatenate,
  sum/mean/max, isfinite, isnan, zeros/full/zeros_like/ones_like/full_like/asarray,
  linspace`), the function transforms (`jit, vmap, grad`), `to_numpy`, and the
  capability flags (`name, is_differentiable, supports_jit`).
- `NumpyBackend` (always available): ops bind to `numpy`; `jit` is identity;
  `vmap` is a Python-loop-plus-`stack` fallback; `grad` raises `NotDifferentiable`
  (with an optional finite-difference path left as a stub for Phase 6);
  `is_differentiable=False, supports_jit=False`.
- `JaxBackend` (constructed only if `import jax` succeeds; lives in `backends/_jax.py`
  which imports `jax` *inside* `__init__` only): ops bind to `jax.numpy`; real
  `jit/vmap/grad`; enables x64 in `__init__`; `is_differentiable=True,
  supports_jit=True`.
- `backends.numpy()` / `backends.jax()` constructors; `jax()` exposed only when
  importable (Pyodide-safe).
- Test harness from §0.4 (three fixtures, inline tier helpers).

**Key decisions enforced here.** Explicit backend passing only — no global, no
contextvar (DESIGN §11.3). The functional, `where`-based style is mandated so the
abstraction stays thin (DESIGN §11.1).

**Tests (`tests/test_backends.py`).**

§2.1 op parity & correctness:
- `test_ops_sin/cos/sqrt/abs/sign/where/isfinite/isnan/stack/sum_mean/minimum_maximum/zeros_full/linspace(numpy_backend)` — copy-value (analytic literals).
- `test_all_ops_parity(jax_backend)` — Tier-B parity for `sqrt/sin/cos/abs/sign`.

§2.2 capability flags & graceful degradation:
- `test_numpy_name/not_differentiable/no_jit/grad_raises/jit_is_identity/vmap_equals_loop/float64_output(numpy_backend)`.
- `test_jax_name/differentiable/supports_jit/float64_output(jax_backend)`.

§2.3 transform parity (JAX only):
- `test_jax_jit_matches_eager(jax_backend)` — Tier C.
- `test_jax_vmap_matches_loop(jax_backend)` — Tier C.
- `test_recompile_only_on_structure` — **stubbed here, completed in Phase 7**
  (needs a real `System`).

Isolation:
- `test_import_without_jax(monkeypatch)` — shadows `jax` in `sys.modules`, reloads
  `trace_light`, asserts no `ImportError`.

**Definition of done.** Package imports with `jax` absent and present. All tests
above green on both backends. `import trace_light` never imports `jax` at top level.

---

## Phase 1 — Core kernels, `Rays`, and `_trace_surfaces` ✅

**Objective.** Build the fixed functional core. We do **not** have prototype trace
code; this phase reconstructs the kernels from the DESIGN spec + Optiland's formulas,
pinned to TEST_PLAN §1 goldens.

**Deliverables.**
- `Rays` NamedTuple (`rays.py`, DESIGN §4.1): `x,y,z,L,M,N,i,w,opd,valid`. Immutable
  — every kernel returns a new `Rays`.
- `_Structure` (static, hashable) and `_Params` (traced) NamedTuples (`rays.py`,
  DESIGN §4.2) — implementation-only, never user-facing.  `_Structure` contains only
  plain Python scalars and tuples (including an `is_plane: tuple[bool]` flag per
  surface); `_Params` contains traced arrays (`radii, conics, n1, n2`).
- Kernels (`kernels.py`, DESIGN §12), each reconstructed from a named Optiland source:
  - `_intersect` ← `StandardGeometry.distance()` (closed-form conic quadratic);
    plane special-case branched via Python `if is_plane` (not `backend.where`) since
    `is_plane` is a static Python bool, not a traced value.
  - `_normal` ← `StandardGeometry.surface_normal()` (near-verbatim; pure arithmetic).
    Convention matches Optiland: conic surfaces return `nz < 0`; planes return
    `(0, 0, +1)`.  `_align_normal` (shared by `_refract` and `_reflect`) flips the
    normal to oppose the incident ray.
  - `_refract` ← `RealRays.refract()` (vector Snell, `root = sqrt(1 − u²(1−dot²))`);
    TIR (imaginary root) returns a boolean `tir` mask and fills direction with the
    `sqrt(0)` fallback so no NaN propagates.  `valid` is updated by the caller.
  - `_reflect` ← `RealRays.reflect()` (`d − 2(d·n)n`), verbatim.
  - `_surface_step`: localize → intersect → propagate → normal → refract/reflect →
    aperture-clip (`r² > semi²`) → OPD update (`opd += n1 * t`) → globalize.
    `INF` semi-aperture disables clipping via Python `if math.isinf(semi)`.
  - `_trace_surfaces(rays, structure, params, backend)`: Python loop over the static
    surface list (unrolls under `jit`); prepends the initial position to the history
    list; returns `(final_rays, history)` where history is a list of `(n_rays, 3)`
    arrays.
- `valid` semantics (DESIGN §4.5): initialized all-True bool array, accumulated
  `valid &= isfinite(t) & ~outside & ~tir`, sticky once False.
- `golden/generate_ray_coords.py` — vendored generator script that traces equivalent
  systems in the Optiland clone and prints copy-pasteable golden arrays.

**Tests (`tests/test_kernels.py`).**

Assertion helpers defined inline at the top of the file:
`_ac / assert_tier_a / assert_tier_b / assert_tier_d`.

§1.1 `_intersect`:
- `test_intersect_sphere_single(backend)` — copy-value, `t=2.7888809636986154`
  (`optiland tests/test_geometries.py::TestStandardGeometry::test_distance`).
- `test_intersect_sphere_tilted(backend)` — copy-value, `t=10.201933401020467`.
- `test_intersect_plane_axial(backend)` — copy-value (analytic), `t=5.0`.
- `test_intersect_batch_equals_stacked(backend)` — Tier B.
- `test_intersect_parity(jax_backend)` — Tier B.

§1.2 `_normal`:
- `test_normal_conic_values(backend)` — copy-value, `(0.10127…, 0.20254…, −0.97402…)`
  (`optiland tests/test_geometries.py::TestStandardGeometry::test_surface_normal`).
- `test_normal_unit_length(backend)` — copy-math, `‖n‖=1`.
- `test_normal_plane(backend)` — copy-value, `(0, 0, 1)`.
- `test_normal_parity(jax_backend)` — Tier B.

§1.3 `_refract`:
- `test_refract_flat_30deg(backend)` — copy-value, `M=1/3, N=2√2/3`.
- `test_refract_snell_residual(backend)` — copy-math, `n1·sinθ1 = n2·sinθ2`.
- `test_refract_unit_direction(backend)` — copy-math, `‖d_out‖=1`.
- `test_refract_tir_sets_flag(backend)` — 45° glass→air sets `tir=True`.
- `test_refract_no_tir_below_critical(backend)` — 30° glass→air sets `tir=False`.

§1.4 `_reflect`:
- `test_reflect_canonical_z_normal(backend)` — copy-math
  (`optiland tests/test_rays.py::test_reflect`).
- `test_reflect_canonical_x_normal(backend)` — copy-math.
- `test_reflect_unit_direction(backend)` — copy-math, `‖d_out‖=1`.
- `test_reflect_coplanar(backend)` — copy-math, coplanarity via cross-product.

§1.5 `_trace_surfaces`:
- `test_trace_history_shape(backend)` — shape `(n_surf+1, n_rays, 3)`.
- `test_trace_ray_coords_vs_reference(backend)` — **copy-array**, plano-convex singlet
  at `z=101.6667`; golden from `golden/generate_ray_coords.py`.
- `test_trace_4f_imaging(backend)` — copy-value golden `y=−0.9663708617629430`;
  chief ray from `y=+1` through biconvex 4f; from `golden/generate_ray_coords.py`.
- `test_trace_opd_flat_plate(backend)` — copy-math, `OPD = n·d = 7.5mm`.
- `test_trace_spherical_aberration_scaling(backend)` — model-pattern, fit exponent ≈3.
- `test_trace_parity(jax_backend)` — Tier B.

§11 valid semantics:
- `test_valid_miss_sets_invalid(backend)` — ray outside `semi_aperture` → `valid=False`.
- `test_valid_tir_sets_invalid(backend)` — TIR → `valid=False`.
- `test_valid_clip_sets_invalid(backend)` — `semi=3.5`, rays at `y=[0..5]` →
  `[T,T,T,T,F,F]`.
- `test_valid_sticky(backend)` — invalid through second surface with `semi=inf`.
- `test_valid_inf_semi_disables_clip(backend)` — `semi=inf`, large-y rays stay valid.

JAX jit/vmap:
- `test_jit_runs(jax_backend)` — `_trace_surfaces` runs under `jax.jit` without error.
- `test_vmap_over_fields(jax_backend)` — `vmap(trace_one)` matches loop+stack, Tier C.

**Definition of done.** All §1 goldens green on both backends; the three DESIGN §2
behavioral checks pass; `_trace_surfaces` runs under `jax.jit` and `jax.vmap`
without error. Tier-B parity holds at `1e-11`.

---

## Phase 2 — Construction layer: `Surface`/`System`, lenses, builder, prefabs, **serialization**

**Objective.** Make the trace usable: the public authored types, lens factories, the
builder that does the bookkeeping, prefab systems, and save/load.

**Deliverables.**
- `Surface` NamedTuple (DESIGN §4.3): `z, radius, conic, n1, n2, semi_aperture,
  reflective`. The only user-authored surface type.
- `System` NamedTuple (DESIGN §4.4): holds `_structure`, `_params`, `pupil_z`,
  `pupil_semi`, `image_z`, `wavelengths`, `backend`. Obtained only from
  `SystemBuilder.finalize()` or a `systems.*` prefab.
- `lenses.*` factories (DESIGN §5), each returning `tuple[Surface, ...]`:
  `singlet, biconvex, plano_convex, doublet, thin_lens, mirror, aperture,
  objective, tube_lens`.
- `SystemBuilder` (DESIGN §6): `.add/.gap/.stop/.image/.finalize`; resolves absolute
  `z` from gaps + element spans, chains `n1/n2` from the preceding gap medium,
  locates the entrance pupil (first powered element by default, `.stop()` override).
  `.finalize()` compiles surfaces into the `_Structure`/`_Params` split and binds the
  backend (default `NumpyBackend`).
- `systems.*` prefabs: `four_f`, `microscope` (infinity-corrected), `relay`,
  `telescope`.
- **Serialization** (your addition; not in DESIGN — landed here because it rides on
  `System`):
  - `System.to_dict()` → `{schema_version, structure, params, pupil_z, pupil_semi,
    image_z, wavelengths}`. Arrays normalized via `backend.to_numpy().tolist()`
    (reuses the existing §11.2 `to_numpy`; no new backend surface).
  - `System.from_dict(data, *, backend=NumpyBackend())` — **backend supplied at load,
    not stored** (consistent with DESIGN §11.3: save a *design*, choose an *engine*
    at reload).
  - `save_system(sys, path)` / `load_system(path)` over `json.dump/load` with a
    JAX/NumPy-aware encoder modeled on `optiland/fileio/optiland_handler.py`
    (`OptilandEncoder`: `if hasattr(obj,'tolist'): return obj.tolist()`).
  - We serialize the **finalized `System`** (resolved), not the builder recipe — it
    is self-contained and round-trips to an identical trace. (Agreed.)

**Tests (`tests/test_lenses.py`, `tests/test_systems.py`).**

§3 lenses:
- `test_lens_biconvex_focal_length(backend)` — copy-value, `R1=+100, R2=−100`.
- `test_lens_plano_convex_geometry(backend)` — copy-value, `R=50`, flat second.
- `test_lens_doublet_index_chain(backend)` — copy-value, 3 surfaces with chained indices.
- `test_lens_thin_zero_thickness(backend)` — copy-value, single zero-thickness surface.
- `test_lens_mirror_reflective(backend)` — copy-value, `reflective=True`.
- `test_lens_aperture_no_power(backend)` — copy-value, no optical power.
- `test_lens_parity(jax_backend)` — Tier B.

§4 systems/builder:
- `test_system_builder_absolute_z(backend)` — copy-math, gap → absolute z.
- `test_system_builder_index_chain(backend)` — copy-math, n1/n2 chaining.
- `test_system_builder_pupil_default(backend)` — copy-math, first powered element.
- `test_system_builder_pupil_stop_override(backend)` — copy-math, `.stop()`.
- `test_system_four_f_magnification(backend)` — copy-math, mag ≈ −1.
- `test_system_telescope_afocal(backend)` — copy-math, afocal exit.
- `test_system_microscope_collimated(backend)` — copy-math, object→collimated→image.
- `test_system_parity(jax_backend)` — Tier B.

Serialization:
- `test_system_roundtrip_trace(backend)` — build → save → load → trace; Tier B
  same-backend.
- `test_system_roundtrip_cross_backend(jax_backend)` — save NumPy, load JAX; Tier B.
- `test_system_schema_version_present` — no backend fixture needed.

**Definition of done.** Every prefab traces correctly on both backends; lensmaker
goldens green; a saved `System` reloads (on either backend) and reproduces its trace
at `1e-11`. Serialized files become reusable fixtures for later phases.

---

## Phase 3 — Sources, pupil sampling, and `emit`

**Objective.** Close the loop so a system can be traced from a named source, with
field/wavelength sweeps via `vmap`.

**Deliverables.**
- `Source` NamedTuple (DESIGN §4.6): `kind, field, wavelength, pupil_pattern,
  n_samples, weights`. Field/angle/wavelength are traced leaves so `vmap(emit)`
  batches them; pattern/`n_samples` are static.
- `sources.*` factories (DESIGN §7): `point_source` (finite conjugate, default),
  `collimated_source` (infinite — needed for infinity-corrected microscopes),
  `extended_source` (2-D image or 3-D volume of incoherent emitters; a batch of
  point sources, conceptually `vmap(point_source)`).
- Pupil sampling patterns, computed once in NumPy (an input, not differentiable):
  `disk, hex, ring, random, fan`. Port the geometry from Optiland's
  `distribution.py` (`UniformDistribution`, `HexagonalDistribution` —
  `1 + 3·rings·(rings+1)` points, `RandomDistribution`, line/cross).
- `emit(Source, System) -> Rays`: reads pupil geometry from the `System` (source
  never hardcodes `pupil_z`); supports `vmap(emit)` over field/λ.
- Wire the public `trace(System, Source, *, backend=None)` from Phase 1 to take a
  `Source` via `emit`.

**Tests (`tests/test_sources.py`).**

- `test_source_point_n_rays_valid(backend)` — copy-math, `n_samples` rays, all
  `valid=True`.
- `test_source_point_unit_directions(backend)` — copy-math, `‖d‖=1`.
- `test_source_point_chief_ray(backend)` — copy-math, chief ray through pupil center.
- `test_source_collimated_direction(backend)` — copy-math, `angle=(0,θ)` →
  `M=sinθ, N=cosθ`.
- `test_pupil_disk_centroid(backend)` — copy-math, centroid ≈ (0,0).
- `test_pupil_disk_radius(backend)` — copy-math, all radii ≤ 1.
- `test_pupil_hex_count(backend)` — copy-value, `1 + 3·rings·(rings+1)` points
  (`optiland tests/test_distribution.py::test_hexapolar`).
- `test_pupil_ring_symmetry(backend)` — copy-math, all radii == 1.
- `test_emit_vmap_fields(jax_backend)` — `vmap(emit)` matches loop+stack, Tier C.
- `test_emit_parity(jax_backend)` — Tier B.

**Definition of done.** A full pipeline runs end-to-end:
`emit(point_source, four_f) → _trace_surfaces → Rays`, on both backends, with `vmap`
over a field/λ batch matching the looped result.

---

## Phase 4 — Read-only analysis: `spot`, `psf`, `irradiance`

**Objective.** Measurement-only helpers (no parameter mutation). `psf` is the
primitive that image-sim and fluorescence both build on, so it is established here.

**Deliverables.**
- `analysis.spot(Rays) -> SpotStats` (DESIGN §10): centroid, RMS, geometric radius
  on final `(x,y)`, masked on `valid`; chief-ray vs centroid reference modes.
- `analysis.psf(system, field, *, depth=0, focus=0, wavelength=None, n_rays,
  grid=(ny,nx), extent_px) -> Array` (DESIGN §8): trace a point emitter at lateral
  `field` and axial `depth`, refocus at detector `focus`, histogram image-plane hits
  into a normalized 2-D kernel. `vmap` over `depth`/`focus` yields a through-focus
  stack with no extra code.
- `analysis.irradiance(system, source, z, grid, *, extent=None) -> Array`
  (DESIGN §9.3): trace `source`, weighted `histogram2d` of valid hits at plane `z`.
  Thin wrapper over `_trace_surfaces` + the backend `histogram2d` (NumPy vs
  `jax.numpy`).

**Tests (`tests/test_analysis.py`).**

§6 spot:
- `test_spot_on_axis_centroid(backend)` — copy-math, centroid ≈ (0,0).
- `test_spot_ideal_rms_zero(backend)` — copy-math, ideal lens at focus RMS ≈ 0.
- `test_spot_excludes_invalid(backend)` — model-pattern, `valid=False` rays excluded.
- `test_spot_chief_vs_centroid(backend)` — model-pattern, off-axis reference modes
  differ.
- `test_spot_parity(jax_backend)` — Tier B.

§7 psf:
- `test_psf_sums_to_one(backend)` — copy-math, kernel sum ≈ 1.0.
- `test_psf_no_nan_inf(backend)` — model-pattern, shape `(ny,nx)`, no NaN/Inf.
- `test_psf_on_axis_centered(backend)` — model-pattern, on-axis in-focus peak centred.
- `test_psf_through_focus_symmetry(backend)` — model-pattern, symmetric about best
  focus.
- `test_psf_parity(jax_backend)` — Tier B.

§9 irradiance:
- `test_irradiance_uniform_flat(backend)` — model-pattern, uniform beam → ~flat
  histogram.
- `test_irradiance_valid_sum(backend)` — copy-math, sum = Σ weights of valid rays.
- `test_irradiance_parity(jax_backend)` — Tier B.

§11 hygiene:
- `test_nan_does_not_poison_reductions(backend)` — NaN in invalid ray must not corrupt
  centroid/RMS/histogram over valid rays.

**Definition of done.** `spot`, `psf`, `irradiance` correct on both backends; PSF
normalizes and centres; reductions are NaN-safe under masking.

---

## Phase 5 — Image simulation (2-D/3-D, single & field-varying PSF)

**Objective.** The most algorithmically involved analysis piece, isolated after the
PSF primitive exists.

**Deliverables.**
- `analysis.image_sim(system, obj, extent, *, psf="varying", field=(0,0),
  grid=(gy,gx), focus=0, wavelength=None) -> Array` (DESIGN §8):
  - `obj` is `(ny,nx)` or `(nz,ny,nx)`; `extent` lateral (+ axial for a volume).
  - Image formation is the incoherent sum over object depth slices, each convolved
    with the PSF for its `(field, depth, focus)`:
    `image[focus] = Σ_z conv(obj[z], psf(depth=z, focus))`.
  - `psf="single"` (shift-invariant): one lateral PSF per depth; FFT-convolve each
    slice (backend `fftconvolve`: SciPy vs `jax.scipy`).
  - `psf="varying"` (field-dependent): PSFs on a coarse `grid` of field points per
    depth (one `vmap` over grid × depths); convolve each region with its local PSF
    and blend across boundaries to avoid seams.
  - scalar `focus` → 2-D image; array `focus` → 3-D focal stack. A purely 2-D
    workflow drops the `nz`/`focus` axes (3-D adds no cost when unused).

**Tests (`tests/test_image_sim.py`).**

- `test_image_sim_2d_shape(backend)` — Tier D, 2-D `obj` + scalar `focus` → `(ny,nx)`.
- `test_image_sim_3d_shape(backend)` — Tier D, 3-D `obj` + `focus` array `(nf,)` →
  `(nf,ny,nx)`.
- `test_image_sim_no_nan(backend)` — Tier D, `max > 0`, no NaN.
- `test_image_sim_single_matches_varying(backend)` — copy-math, shift-invariant system
  `psf="single"` ≈ `psf="varying"`.
- `test_image_sim_energy_conserved(backend)` — copy-math, total image energy tracks
  object energy × throughput.
- `test_image_sim_parity(jax_backend)` — Tier B.

**Definition of done.** Both PSF modes agree on a shift-invariant system; 2-D and
3-D shapes correct; energy conserved; parity holds on both backends.

---

## Phase 6 — Optimization: `best_focus`, `minimize`

**Objective.** Differentiable parameter optimization. Separated from `analysis` by
design (DESIGN §13.1) because it *mutates* parameters toward an objective.

**Deliverables.**
- `optimize.best_focus(System) -> float` (DESIGN §3, §10): gradient descent on spot
  variance over the traced `image_z`. Reproduces the verified autofocus:
  408.000 mm / 90.93 µm → ≈405.874 mm / 11.81 µm.
- `optimize.minimize(System, objective, params, ...) -> System`: general grad-based
  parameter optimization (radii, spacings, indices as traced leaves).
- Differentiability story: analytic `jax.grad` on the JAX backend; on NumPy,
  `backend.is_differentiable=False` gates this — either the finite-difference fallback
  (stubbed in Phase 0, completed here) or a clear `NotDifferentiable` raise.

**Tests (`tests/test_optimize.py`).**

- `test_best_focus_reduces_rms(jax_backend)` — model-pattern, RMS strictly decreases.
- `test_best_focus_converges(jax_backend)` — copy-value, converged `image_z` within
  tolerance of 405.874 mm.
- `test_best_focus_residual_is_aberration(jax_backend)` — model-pattern, converged
  RMS bounded below by spherical-aberration floor.
- `test_grad_vs_finite_diff(jax_backend)` — copy-math, `jax.grad` ≈ central FD.
- `test_optimize_no_nan(jax_backend)` — model-pattern.
- `test_optimize_numpy_behavior(numpy_backend)` — capability, raises
  `NotDifferentiable` (or matches FD fallback if shipped).

**Definition of done.** Autofocus reproduces the DESIGN §3 numbers on JAX; analytic
gradient matches finite difference; NumPy path behaves per its capability flag.

---

## Phase 7 — Visualization, fluorescence assembly, hardening

**Objective.** The low-priority and compositional remainder, plus the cross-cutting
hardening backlog.

**Deliverables.**
- `viz.layout(System, Source, dim) -> figure` (DESIGN §10): 2-D y–z ray fan and 3-D
  paths from `_trace_surfaces` history. Smoke-tested only (not accuracy-critical).
- **Fluorescence pipeline** (DESIGN §9) — assembled as a documented *composition* of
  existing pieces, not a new subsystem:
  - Pass 1 (optional): excitation irradiance `I_exc` via `analysis.irradiance`
    (or uniform `1.0` for widefield/Köhler).
  - Build the incoherent emission volume `emission = I_exc · fluorophore` (one-photon)
    or `I_exc² · fluorophore` (two-photon).
  - Pass 2: `analysis.image_sim(collection_sys, emission, extent, psf="varying",
    focus=z_planes, wavelength=λ_em)` — imaged at the **emission** wavelength.
  - Document the §9.4 modality table (widefield / confocal / two-photon / light-sheet
    / 3-D) as excitation-side variations layered on this pipeline.
- **Hardening** (DESIGN §14 / TEST_PLAN §11 remainder): finalize
  `test_recompile_only_on_structure` (parameter change → no recompile; structure
  change → recompile) now that a real `System` exists; confirm misses/TIR set `valid`
  and don't poison downstream; document the nominal-vs-real spacing defocus.

**Tests (`tests/test_viz.py`, `tests/test_backends.py` §2.3 close-out,
`tests/test_fluorescence.py`).**

§12 viz:
- `test_layout_2d_returns_figure(backend)` — smoke, `dim=2` returns a figure without
  error.
- `test_layout_3d_returns_figure(backend)` — smoke, `dim=3` returns a figure without
  error.

§2.3 close-out (back in `tests/test_backends.py`):
- `test_recompile_only_on_structure(jax_backend)` — parameter change → no recompile;
  structure change → recompile; trace-count instrumentation.

Fluorescence end-to-end:
- `test_fluorescence_widefield_shapes(backend)` — smoke, widefield → emission volume
  → image stack; assert shapes correct.
- `test_fluorescence_no_spurious_gain(backend)` — copy-math, no gain above input
  energy.

**Definition of done.** Layout renders in 2-D/3-D; the fluorescence pipeline runs
end-to-end as a composition with no new core code; recompilation occurs only on
structure change; the §14 limitations are documented.

---

## Sequencing notes and risks

- **Phases 0–1 are the spine.** If schedule slips, slip later phases — the backend +
  kernels + their goldens are what make every subsequent claim verifiable.
- **Optiland clone stays available** throughout as the comparison oracle, specifically
  for regenerating the Phase-1 copy-array ray-coordinate goldens and for cross-checking
  paraxial/lensmaker values in Phase 2.
- **Fluorescence is composition, not a subsystem.** It is Phase-7 assembly. If it turns
  out to be the *primary* near-term driver, pull a thin vertical slice forward (a single
  widefield end-to-end run) right after Phase 5 to validate the pipeline earlier — this
  is a known, low-cost adjustment to the plan.
- **Out of scope for v1** (DESIGN §1, §15): polarization/coatings/scatter, aspheres/
  freeforms needing Newton–Raphson intersection, physical-optics PSF, dispersive
  materials. The architecture leaves the door open (traced parameters, the `Backend`
  protocol, `lax.while_loop` for future iterative solvers) without building them now.
