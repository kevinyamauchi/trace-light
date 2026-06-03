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

### 0.3a Code style: docstrings and imports

These two conventions apply to every function and module in `src/`, including private
helpers.  They were retrofitted onto Phases 0–1 and must be followed in all future
phases.

**Docstrings — NumPy style, always complete.**

Every function (public or private, including inner closures such as `vmapped`) must
have a NumPy-style docstring with, at minimum:

- A one-line summary.
- A `Parameters` section listing every argument with its type and a short description.
- A `Returns` section (or a `Raises` section for functions that always raise).

Stub functions whose API is not yet defined use `*args: Any, **kwargs: Any`; their
`Parameters` section documents both as `Any` with a note that the signature is pending,
and their `Raises` section cites `NotImplementedError`.

**Type annotations — all arguments and return values.**

Every function signature must carry type annotations on every argument and the return.
The type conventions in use:

| Context | Type to use |
|---------|-------------|
| Backend-agnostic array (kernel inputs/outputs) | `Any` (from `typing`) |
| NumPy-specific array | `np.ndarray` |
| JAX array | `Any` (JAX types require a runtime import) |
| Callable passed to `jit`/`vmap`/`grad` | `Callable[..., Any]` (under `TYPE_CHECKING`) |
| Shape arguments | `int \| tuple[int, ...]` |
| Optional dtype | `Any` (covers NumPy and JAX dtype objects) |

`Callable` is imported under `TYPE_CHECKING` to keep it off the runtime import path
(matches the lazy-`jax` discipline in §0.1).

**Imports — absolute only.**

All intra-package imports use absolute paths (`from trace_light.backends._protocol
import Backend`, not `from ._protocol import Backend`).  The sole exception is a
deferred `import` statement inside a function body used to avoid a circular import or
to keep an optional dependency off the load path; that import may remain bare
(`import jax`) but must not use the relative-dot form.

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
Phase 0  backends + harness                              ✅
   │
Phase 1  Rays, kernels, trace        ──┐                 ✅
   │                                    │ (Optiland math is the oracle)
Phase 2  Surface/System, lenses,        │                ✅
         builder, prefabs, SERIALIZE  ──┘
   │
Phase 3  Source, sampling, emit                          ✅
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

## Phase 2 — Construction layer: `Surface`/`System`, lenses, builder, prefabs, **serialization** ✅

**Objective.** Make the trace usable: the public authored types, lens factories, the
builder that does the bookkeeping, prefab systems, and save/load.

**Deliverables.**
- `Surface` NamedTuple (`rays.py`, DESIGN §4.3): `z, radius, conic, n1, n2,
  semi_aperture, reflective`.  `radius=math.inf` indicates a flat plane.
- `System` NamedTuple (`rays.py`, DESIGN §4.4): holds `structure` (`_Structure`),
  `params` (`_Params`), `pupil_z`, `pupil_semi`, `image_z`, `wavelengths`, `backend`.
  Obtained only from `SystemBuilder.finalize()` or a `systems.*` prefab.
- `lenses.*` factories (`lenses.py`, DESIGN §5), each returning `tuple[Surface, ...]`:
  `singlet, biconvex, plano_convex, doublet, thin_lens, mirror, aperture`.
  (`objective` and `tube_lens` deferred — not needed for current phases.)
- `SystemBuilder` (`systems.py`, DESIGN §6): `.add/.gap/.stop/.image/.finalize`.
  `.finalize()` compiles surfaces into the `_Structure`/`_Params` split and binds the
  backend (default `NumpyBackend`).
- `systems.*` prefabs: `four_f`, `microscope` (infinity-corrected), `relay`,
  `telescope`.
- **Serialization** (in `rays.py`, rides on `System`):
  - `System.to_dict()` → `{schema_version, structure, params, pupil_z, pupil_semi,
    image_z, wavelengths}`. Arrays via `backend.to_numpy().tolist()`; `math.inf`
    replaced by the `"__inf__"` sentinel for JSON safety.
  - `System.from_dict(data, *, backend=NumpyBackend())` — backend supplied at load.
  - `save_system(sys, path)` / `load_system(path, *, backend=None)` over
    `json.dump/load` with a `hasattr(obj,'tolist')` encoder.

**Implementation notes (deviations / decisions made).**

- **z-relative factory convention.** Factories return surfaces with z-coordinates
  relative to 0 (first surface at `z=0`). `SystemBuilder.add()` offsets them by the
  current cursor and advances the cursor to `max(s.z for s in surfaces)`. `gap()`
  advances without adding surfaces. This makes factory output self-contained and the
  builder responsible only for placement.

- **`gap(n=...)` is informational only.** The `n` kwarg is accepted for API symmetry
  but not used; n1/n2 come from the Surface objects produced by the factories. The
  "chains n1/n2" language in the plan refers to the factory producing correct chained
  indices, not the builder re-writing them.

- **`thin_lens` returns 2 surfaces at same z.** Implemented as a zero-thickness
  symmetric biconvex (R = 2*(n-1)*f) so that `s[0].z == s[1].z`. The test
  `test_lens_thin_zero_thickness` verifies both surfaces are co-located and the
  computed R matches the lensmaker formula.

- **Pupil location defaults to first non-plane surface.** If `.stop()` was not called,
  `finalize()` finds the first surface with a finite radius and places the pupil there.
  `pupil_semi` is taken from that surface's `semi_aperture` (may be `math.inf` for
  unconstrained systems; the prefabs always call `.stop()` explicitly).

- **Prefab magnification tolerance.** `test_system_four_f_magnification` uses a 2%
  tolerance (not Tier-A) because thick lenses shift principal planes; the paraxial
  approximation `mag = -1` is not exact at finite thickness.

**Tests (`tests/test_lenses.py`, `tests/test_systems.py`).**

§3 lenses (all green):
- `test_lens_biconvex_focal_length(backend)` — copy-math, lensmaker gives `f≈100mm`.
- `test_lens_plano_convex_geometry(backend)` — copy-value, flat back `radius=inf`.
- `test_lens_doublet_index_chain(backend)` — copy-value, 3 surfaces, chained indices.
- `test_lens_thin_zero_thickness(backend)` — copy-value, both surfaces at `z=0`,
  `R=100mm` for `f=100,n=1.5`.
- `test_lens_mirror_reflective(backend)` — copy-value, `reflective=True`.
- `test_lens_aperture_no_power(backend)` — copy-value, plane surface, finite semi.
- `test_lens_parity(jax_backend)` — Tier B (plain Python float consistency).

§4 systems/builder (all green):
- `test_system_builder_absolute_z(backend)` — copy-math, gap→absolute z (Tier D).
- `test_system_builder_index_chain(backend)` — copy-math, n1/n2 per surface (Tier D).
- `test_system_builder_pupil_default(backend)` — copy-math, first powered element.
- `test_system_builder_pupil_stop_override(backend)` — copy-math, `.stop()`.
- `test_system_four_f_magnification(backend)` — model-pattern, |mag|≈1, sign negative;
  2% tolerance to accommodate thick-lens principal-plane shift.
- `test_system_telescope_afocal(backend)` — model-pattern, |M_exit|<5e-3 (nearly
  parallel exit for an axial input ray through a thick Keplerian telescope).
- `test_system_microscope_collimated(backend)` — model-pattern, on-axis ray stays near
  axis through an infinity-corrected microscope.
- `test_system_parity(jax_backend)` — Tier B.

Serialization (all green):
- `test_system_roundtrip_trace(backend)` — build → save → load → trace; Tier B.
- `test_system_roundtrip_cross_backend(jax_backend)` — save NumPy, load JAX; Tier B.
- `test_system_schema_version_present` — `schema_version == 1` always present.

**Definition of done.** ✅ Every prefab traces on both backends; lensmaker goldens green;
a saved `System` reloads on either backend and reproduces its trace at `1e-11`.

---

## Phase 3 — Sources, pupil sampling, and `emit` ✅

**Objective.** Close the loop so a system can be traced from a named source, with
field/wavelength sweeps via `vmap`.

**Deliverables.**
- `Source` NamedTuple (`sources.py`, DESIGN §4.6): `kind, field, wavelength,
  pupil_pattern, n_samples, weights`. Field/angle/wavelength are traced leaves so
  `vmap(emit)` batches them; pattern/`n_samples` are static Python values.
- `sources.*` factories: `point_source`, `collimated_source`, `extended_source`.
- Pupil sampling patterns in NumPy (never traced): `disk` (Fibonacci spiral),
  `hex` (hexapolar, `1 + 3·rings·(rings+1)` pts), `ring`, `random`, `fan`.
- `emit(Source, System) -> Rays`: pupil sampling computed once in NumPy; direction
  arithmetic uses the backend so JAX can vmap/jit through it.

**Implementation notes (deviations / decisions made).**

- **`field` encoding.** `point_source` encodes field as a length-3 NumPy array
  `[x, y, z_object]`; `collimated_source` encodes it as `[theta_x, theta_y]` (rad).
  These become JAX traced arrays inside `vmap`.

- **`extended_source` returns a Python list.** It is `[point_source(p, ...) for p in
  field_points]` — not a single batched Source. Users loop or `vmap(emit)` over the
  list manually.

- **`emit` two-phase structure.** Pupil samples (px, py) are computed in NumPy as
  static constants before any backend math. Only the direction arithmetic (subtraction,
  `sqrt`, `full`) uses the backend, so JAX sees a pure function of `source.field` and
  can vmap/jit through it.

- **`trace(System, Source)` public wrapper** was not wired up in this phase. Users
  call `emit(src, sys)` then `_trace_surfaces(rays, sys.structure, sys.params, be)`
  directly. A convenience wrapper is deferred to Phase 4 or later.

- **`for "point"` kind**: rays are placed at the source position `(fx, fy, fz)` with
  directions aimed at the pupil samples. Directions are unit vectors by construction
  (normalized by magnitude).

- **`for "collimated"` kind**: rays are placed at the pupil plane `(px, py, pupil_z)`
  with the same direction `(sin θ_x, sin θ_y, cos θ)` for all rays.

**Tests (`tests/test_sources.py`, all green).**

- `test_source_point_n_rays_valid(backend)` — copy-math, exactly 13 rays, all valid.
- `test_source_point_unit_directions(backend)` — copy-math, `‖d‖=1` to 1e-12.
- `test_source_point_chief_ray(backend)` — copy-math, ray starts at source position,
  N>0 (pointing toward pupil).
- `test_source_collimated_direction(backend)` — copy-math, `M=sinθ, N=cosθ` (Tier D).
- `test_pupil_disk_centroid(backend)` — copy-math, centroid ≈ (0,0) for n=500.
- `test_pupil_disk_radius(backend)` — copy-math, all radii ≤ 1.
- `test_pupil_hex_count(backend)` — copy-value, `1+3·rings·(rings+1)` for rings 1–5
  (matches `optiland tests/test_distribution.py::test_hexapolar`).
- `test_pupil_ring_symmetry(backend)` — copy-math, all radii == 1 (Tier D).
- `test_emit_vmap_fields(jax_backend)` — `jax.vmap(emit_one)` over 3 field points
  matches loop+stack; Tier C.
- `test_emit_parity(jax_backend)` — NumPy vs JAX ray bundle identical at Tier B.
- `test_emit_trace_end_to_end(backend)` — full pipeline: `point_source → emit →
  _trace_surfaces`; all rays valid, history length correct.

**Definition of done.** ✅ Full pipeline runs end-to-end on both backends; `vmap(emit)`
over fields matches the looped result at Tier-C tolerance.

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
