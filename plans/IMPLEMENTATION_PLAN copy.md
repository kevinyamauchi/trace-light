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

- A backend-parametrized fixture analogous to Optiland's
  `tests/conftest.py::set_test_backend` — every accuracy test runs once per backend.
- An `assert_allclose(a, b, rtol, atol)` helper that funnels **both** operands
  through `backend.to_numpy()` before `np.allclose`, mirroring
  `optiland/tests/utils.py`.
- The four tolerance tiers from TEST_PLAN §0.2:
  - **A** reference/golden: `rtol=1e-5, atol=1e-7`
  - **B** cross-backend parity: `rtol=1e-11, atol=1e-12`
  - **C** transform parity (jit vs eager, vmap vs loop): `rtol=1e-9, atol=1e-10`
  - **D** exact invariant: `atol=1e-12`

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

## Phase 0 — Backend abstraction + project skeleton

**Objective.** Stand up the package, the `Backend` protocol, both backends, and the
test harness. Nothing optical happens here; this phase exists so the two-backend
promise is structurally guaranteed before any kernel is written.

**Deliverables.**
- Package skeleton `raytrace_jax/` with the module layout from DESIGN §10
  (`backends`, plus empty stubs for `lenses`, `systems`, `sources`, `analysis`,
  `optimize`, `viz`).
- `Backend` protocol (DESIGN §11.2): the ~25 array ops the core uses
  (`sqrt, sin, cos, abs, sign, where, minimum/maximum, stack, concatenate,
  sum/mean/max, isfinite, isnan, zeros/full/asarray`, …), the function transforms
  (`jit, vmap, grad`), `to_numpy`, and the capability flags
  (`name, is_differentiable, supports_jit`).
- `NumpyBackend` (always available): ops bind to `numpy`; `jit` is identity;
  `vmap` is a Python-loop-plus-`stack` fallback; `grad` raises `NotDifferentiable`
  (with an optional finite-difference path left as a stub for Phase 6);
  `is_differentiable=False, supports_jit=False`.
- `JaxBackend` (constructed only if `import jax` succeeds; lives in its own module
  that imports `jax` *inside* it): ops bind to `jax.numpy`; real `jit/vmap/grad`;
  enables x64 in `__init__`; `is_differentiable=True, supports_jit=True`.
- `backends.numpy()` / `backends.jax()` constructors; `jax()` exposed only when
  importable (Pyodide-safe).
- Test harness from §0.4 (fixture, `assert_allclose`, tiers).

**Key decisions enforced here.** Explicit backend passing only — no global, no
contextvar (DESIGN §11.3). The functional, `where`-based style is mandated so the
abstraction stays thin (DESIGN §11.1).

**Tests (TEST_PLAN §2).**
- §2.1 op parity & correctness — copy-value for the analytic ones
  (`sin([0,π/2,π])=[0,1,0]`), Tier-B parity authored fresh for `test_all_ops_parity`.
- §2.2 capability flags & graceful degradation — `NumpyBackend` flags False,
  `grad` raises `NotDifferentiable`, `jit` is identity, `vmap` equals loop+stack,
  float64 outputs. Model-pattern from `test_backend_contract.py`.
- §2.3 transform parity (JAX only) — `jit` matches eager (Tier C), `vmap` matches
  loop (Tier C). `test_recompile_only_on_structure` is stubbed here and completed
  in Phase 7 (needs a real `System`).

**Definition of done.** Package imports with `jax` absent (simulate by hiding it)
and present. All §2 tests green on both backends. `import raytrace_jax` never
imports `jax` at top level (assert via a subprocess with `jax` shadowed).

---

## Phase 1 — Core kernels, `Rays`, and `trace` (reconstruct from Optiland math)

**Objective.** Build the fixed functional core. We do **not** have prototype trace
code; this phase reconstructs the kernels from the DESIGN spec + Optiland's formulas,
pinned to TEST_PLAN §1 goldens.

**Deliverables.**
- `Rays` NamedTuple (DESIGN §4.1): `x,y,z,L,M,N,i,w,opd,valid`. Immutable — every
  kernel returns a new `Rays`.
- Internal `_Structure` (static, hashable) and `_Params` (traced) NamedTuples
  (DESIGN §4.2) — implementation-only, never user-facing.
- Kernels (DESIGN §12), each reconstructed from a named Optiland source:
  - `_intersect` ← `StandardGeometry.distance()` (closed-form conic quadratic),
    rewritten functional; plane special-case folded via `backend.where` instead of
    the Python `_is_radius_infinite` branch.
  - `_normal` ← `StandardGeometry.surface_normal()` (near-verbatim; pure arithmetic).
  - `_refract` ← `RealRays.refract()` (vector Snell, `root = sqrt(1 − u²(1−dot²))`);
    TIR (imaginary root) sets `valid=False` rather than Optiland's NaN-and-continue.
  - `_reflect` ← `RealRays.reflect()` (`d − 2(d·n)n`), verbatim.
  - `_surface_step`: localize → intersect → propagate → refract/reflect →
    aperture-clip (`backend.where`) → update `valid` → globalize.
  - `trace(rays, params, structure, backend)`: Python loop over the static surface
    list (unrolls under `jit`). Public entry `trace(System, Source, *, backend=None)`
    is wired in Phase 3 once `Source`/`emit` exist; here it is exercised with
    hand-built `_Params`/`_Structure`.
- `valid` semantics (DESIGN §4.5): initialized all-True, accumulated
  `valid &= inside_aperture & isfinite(t) & ~tir`, sticky once False, distinct from
  `i`. `INF` semi-aperture disables clipping.

**Tests (TEST_PLAN §1 + the §11 items that are kernel-level).**
- §1.1 `_intersect` — copy-value (`t=2.7888809636986154`, `10.201933401020467`;
  plane axial `t=5.0`); batch equals stacked per-ray; `test_intersect_parity` Tier B.
  *(All three sampled goldens already verified to reproduce against Optiland's math.)*
- §1.2 `_normal` — copy-value (conic normal triple); copy-math for `‖n‖=1`; Tier-B parity.
- §1.3 `_refract` — copy-value (flat 30° → `(0,0.33333,0.94280904)`); copy-math for
  Snell residual and `‖d_out‖=1`; `test_refract_tir_sets_invalid` (45°>θc → valid False).
- §1.4 `_reflect` — copy-math (canonical normals, coplanarity, unit direction).
- §1.5 `_surface_step` & `trace` — `test_trace_4f_imaging` (y=+1.0 → ≈−0.96, DESIGN §2),
  `test_trace_spherical_aberration_scaling` (RMS ~ aperture³, fit exponent ≈3),
  `test_trace_opd_flat_plate` (OPD increment = n·d), `test_trace_ray_coords_vs_reference`
  (**copy-array**, regenerated from the clone on a sphere/conic singlet/doublet),
  `test_trace_history_shape` `(n_surf+1, n_rays, 3)`, `test_trace_parity` Tier B.
- §11 kernel-level: `test_miss_sets_invalid`, `test_tir_sets_invalid`,
  `test_clip_sets_invalid` (radial `r_min=2,r_max=5` on `[0..5]` → `[F,F,T,T,F,F]`),
  `test_valid_sticky`.

**Definition of done.** All §1 goldens green on both backends; the three DESIGN §2
behavioral checks pass; `trace` runs under `jax.jit` and `jax.vmap` (fields) without
error. Tier-B parity holds at `1e-11`.

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

**Tests (TEST_PLAN §3, §4 + roundtrip).**
- §3 lenses — copy-value via lensmaker relations: `biconvex(f=100,n=1.5)→R1=+100,
  R2=−100`; `plano_convex(f=100,n=1.5)→R=50`, flat second surface; doublet returns 3
  surfaces with chained indices; `thin_lens` single zero-thickness surface;
  `mirror.reflective=True`; `aperture` no power; `test_factory_parity` Tier B.
- §4 systems/builder — copy-math/copy-value: absolute-z resolution, index chaining,
  pupil location + `.stop()` override; `four_f(f=100)` magnification ≈ −1;
  `telescope` afocal; `microscope` object→collimated→image at tube focal plane;
  `test_system_parity` Tier B.
- **Serialization** (new): `test_system_roundtrip` (build → save → load → trace →
  identical final coords, Tier B same-backend); `test_roundtrip_cross_backend`
  (save on NumPy, load on JAX, trace, Tier B parity); `test_schema_version_present`.

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

**Tests (TEST_PLAN §5).**
- copy-math: `point_source` emits `n_samples` rays, all `valid=True`, all `‖d‖=1`;
  chief ray through pupil center; `collimated_source(angle=(0,θ))` → `M=sinθ, N=cosθ`.
- copy-value/copy-math: `disk` centroid ≈ (0,0), radius ≤ 1; `hex`/`ring` expected
  counts/symmetry (port `test_distribution.py::test_hexapolar` counts).
- Tier C: `test_emit_vmap_fields` (`vmap(emit)` == loop+stack).
- Tier B: `test_emit_parity`.

**Definition of done.** A full pipeline runs end-to-end:
`emit(point_source, four_f) → trace → Rays`, on both backends, with `vmap` over a
field/λ batch matching the looped result.

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
  Thin wrapper over `trace` + the backend `histogram2d` (NumPy vs `jax.numpy`).

**Tests (TEST_PLAN §6, §7, §9 + the NaN-hygiene §11 items).**
- §6 spot — copy-math/model-pattern: on-axis centroid ≈ (0,0); ideal lens at focus
  RMS ≈ 0; `valid=False` rays excluded; chief vs centroid differ off-axis; Tier B parity.
- §7 psf — copy-math/model-pattern: kernel sums ≈ 1.0; shape `(ny,nx)` no NaN/Inf;
  on-axis in-focus peak centered; through-focus symmetry about best focus; Tier B parity.
- §9 irradiance — model-pattern: uniform beam → ~flat histogram; sum = Σ weights of
  valid rays; Tier B parity.
- §11 hygiene: `test_nan_does_not_poison_reductions` (a NaN in an invalid ray must
  not corrupt centroid/RMS/histogram over valid rays); `test_inf_semi_disables_clip`.

**Definition of done.** `spot`, `psf`, `irradiance` correct on both backends; PSF
normalizes and centers; reductions are NaN-safe under masking.

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

**Tests (TEST_PLAN §8).**
- Shape/sanity (Tier D): 2-D `obj` scalar `focus` → `(ny,nx)`; 3-D `obj` + `focus`
  array `(nf,)` → `(nf,ny,nx)`; no NaN, `max>0`.
- copy-math/model-pattern: on a shift-invariant system `psf="single"` ≈
  `psf="varying"`; total image energy tracks object energy × throughput (no spurious
  gain); Tier B parity.

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

**Tests (TEST_PLAN §10).**
- model-pattern (numbers are ours, structure from `test_torch_optimization.py`):
  `test_best_focus_reduces_rms` (RMS strictly decreases; converged `image_z` within
  tolerance of 405.874); `test_best_focus_residual_is_aberration` (converged RMS
  bounded below by the spherical-aberration floor — refocus can't remove it);
  `test_grad_vs_finite_diff` (`jax.grad(objective)` ≈ central FD); `test_optimize_no_nan`.
- capability: `test_optimize_numpy_behavior` — on NumPy, raises `NotDifferentiable`
  (or matches the FD fallback if shipped).

**Definition of done.** Autofocus reproduces the DESIGN §3 numbers on JAX; analytic
gradient matches finite difference; NumPy path behaves per its capability flag.

---

## Phase 7 — Visualization, fluorescence assembly, hardening

**Objective.** The low-priority and compositional remainder, plus the cross-cutting
hardening backlog.

**Deliverables.**
- `viz.layout(System, Source, dim) -> figure` (DESIGN §10): 2-D y–z ray fan and 3-D
  paths from `trace` history. Smoke-tested only (not accuracy-critical).
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

**Tests (TEST_PLAN §12 + §2.3 / §11 closeout).**
- §12 viz — smoke only: history shape `(n_surf+1, n_rays, 3)`; `dim=2` and `dim=3`
  return a figure without error.
- §2.3: `test_recompile_only_on_structure` (JAX-specific; trace-count instrumentation).
- A fluorescence end-to-end smoke/energy test composing the pipeline (widefield →
  emission volume → image stack; assert shapes + no spurious gain).

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
