# Audit, September 2026

A full read of every source file, a bug hunt, and an optimisation pass.
Baseline: 60 tests passing. After: **74 passing** (14 new regression tests, one
per fix). Every physical quantity checked before and after agrees to float32
round-off; every integer vortex count is bit-identical.

Three sections: **A** things that were fine (so nobody re-checks them), **B**
bugs fixed, **C** optimisations, **D** what is still open.

---

## A. Verified correct — do not "fix" these

* **The kinetic factor** `exp(-i*0.25*dt*K_sq)` for a half step. Re-derived and
  re-checked; the previous fix from `0.5` was right.
* **Yoshida-4 composition and merging.** Expanded the merged loop back to
  `K(th dt/2) N(th dt) K((1-th)dt/2) N((1-2th)dt) K((1-th)dt/2) N(th dt) K(th dt/2)`
  by hand — correct, including the `kin_edge^2` merge between outer steps.
* **Strang block merging.** `K(dt/2)[N K]^(n-1) N K(dt/2)` is exactly `n` steps.
* **TW noise normalisation.** `std = sqrt(variance/2)` per quadrature, and
  `delta_psi = (N^3/sqrt(V)) * ifftn(alpha)` — both re-derived, both right.
  `offset_density = sum_k <|alpha_k|^2> / V` likewise.
* **The plaquette winding loop** is the correct counter-clockwise circuit, and
  `_phase_diff`'s real/imag expansion of `arg(psi_q conj(psi_p))` is right.
* **`winding.slice_axes()`** — the 2D/3D sign agreement. Checked all three
  normals by hand.
* **`dilate`/`erode`** roll the pass's *input*, not the partly-grown result, so
  one pass is exactly one cell. (Looks like a bug; is not. Now pinned by a
  test so it does not get "fixed".)
* **Angular momentum** `Lx = <Y jz - Z jy>` and cyclic — correct.
* **Chunked ensemble accumulation** — the `/chunk_size` on extensive
  quantities and the un-divided `phys_*` are both right.
* **`line_length_from_crossings`** — the isotropy-averaged factor `2 dx/3` is
  right.
* **`units.py`.** The `a = 2.75 nm` and `g = 4*pi*a/l_phys` derivation carried
  a "best effort, verify" flag. **Now verified**: the source paper's own
  dimensionless coupling is `U0~ = 8*pi*a/x0` with `x0 = l_phys/sqrt(2)`, i.e.
  `U0~ = 2*sqrt(2)*g`. This project's `g = 3.5233e-3` gives `9.965e-3` against
  the paper's stated `1e-2` — 0.35%. The flag is closed. See `PHYSICS.md` §2.1.

---

## B. Bugs fixed

### B1. `run_block(0)` was not a no-op — `gpe3d/physics.py`
Both compositions applied their leading half/edge kinetic factor before
discovering there was nothing to do. At order 2 a zero-step call advanced psi
by a **full** step; at order 4 it returned psi with an **unpaired**
`exp(-i*th*dt*K_sq/4)` on it — a silently corrupted state from a call that
should not have touched it. Latent (production always passes `n_steps >= 1`),
but a trap for any new caller. Guarded, and negative `n_steps` now raises.
*Test: `test_run_block_zero_steps_is_a_no_op`.*

### B2. Two different `dx` — `run.py` vs `gpe3d/physics.py`
The engine's grid keeps both endpoints, so `dx = L/(N-1)`. `run.py` computed
`dx = L/N` for its pre-flight aliasing gate, making the Nyquist wavenumber
`pi/dx` a factor `N/(N-1)` too high — a gate that was systematically too
generous on **the one failure mode no conservation gate can catch**. Also
printed a `dx` that disagreed with the one written into `run_metadata.json`.
Added `physics.grid_spacing(N, L)` as the single definition; `run.py` uses it,
and now also prints the true FFT box `N*dx`.
*Test: `test_grid_spacing_matches_the_engine`.*

### B3. Two different healing lengths — `nucleation/runner.py`
`healing_length()` returned `1/sqrt(g n)` while `gpe3d/noise.py` builds its UV
cutoff on `xi = 1/sqrt(2 g n)`. The reported `dx/xi` — and the "cores
resolved" verdict `run.py` prints before a long run — were therefore
**optimistic by sqrt(2)**: a grid at `dx = 0.42 xi` reported `0.30` and passed
the `dx <= xi/3` line. Now `1/sqrt(2 g n)` project-wide, which is also the
convention under which `E_k(1/xi) = g n` exactly.
*Test: `test_healing_length_uses_the_same_convention_as_the_noise_cutoff`.*

### B4. `vortex_map.gif` interleaved every trajectory — `nucleation/vortex_map.py`
The observer stores one frame per **(trajectory, time)**, so an 8-trajectory
run's `pierce_points.npz` holds each time eight times over in block-major
order. `load_run()` returned all of them, so the movie played
`t0(traj0), t0(traj1), ... t1(traj0), ...` — eight different realisations
cutting between each other at every frame — and the count-vs-time strip plotted
each time eight times on its x axis. Since vortices nucleate in different
places in every trajectory, that is not a noisier version of the right picture;
it is a different picture at every frame. **The `vortex_map.gif` currently in
`results/` was produced this way.** `load_run(path, trajectory=...)` now
selects one realisation and sorts by time; `runner.write_outputs()` passes
`cfg.slice_trajectory`, so both movies follow the same trajectory; the CLI
gained `--trajectory`.
*Test: `test_vortex_map_loads_a_single_trajectory`.*

### B5. `aggregate()` raised on any observable but `n_lines` — `nucleation/statistics.py`
It read `out["n_lines"]` unconditionally, so `aggregate(rows,
observables=("L_total",))` — and `compare_to_control(key="L_total")` through
it — raised `KeyError`. Since `L_total` is the observable the project's own
documentation tells you to do the statistics on, the documented path was the
broken one. Now falls back to the first observable and reports which one it
used as `count_key`.
*Test: `test_aggregate_works_without_n_lines`.*

### B6. The movie masked against the wrong peak — `nucleation/observer.py`
`_record_slices()` masked each 2D panel with `mask_threshold * max(this plane)`
while `detect_frame()` masks with `mask_threshold * max(the volume)`. On every
plane that does not cut the densest part of the cloud the panel's threshold was
**strictly lower** than the counter's, so the movie could ring cores the count
excluded — precisely what that function's docstring promises cannot happen.
Now passed the trajectory's 3D `n_peak` (reused from the block's detection when
there was one, otherwise one reduction; `max(d - c) == max(d) - c`, so the
offset costs nothing). The docstring now also states the one remaining,
deliberate difference: the panel does not apply the 3D density-dip filter, so
it is an *upper bound* on what the counter accepts.

### B7. `dx` carried float32 rounding — `gpe3d/physics.py`
`self.dx = float(x[1] - x[0])` on a float32 grid was ~4e-6 relative off the
spacing it describes, and fed `dV` (3x that), every norm and energy, and the
`k` grid. Now the exact `L/(N-1)` in float64.

### B8. `max_component` could not fire before the allocation it guarded — `nucleation/detect.py`
`trace_lines(max_component=20000)` skips the dense distance matrix for huge
components. But the matrix is `4 P^2` bytes and the old one-liner
`norm(pts[:,None,:] - pts[None,:,:], axis=-1)` first materialises a `(P,P,3)`
difference — at `P=20000`, 1.6 GB plus 4.8 GB of temporary. The guard was
three orders of magnitude above what it was guarding. Distances are now built
in row chunks (bit-identical output, `(chunk,P,3)` temporary), and the default
is `4000`, which caps the matrix at 64 MB.
*Test: `test_chunked_distances_match_the_dense_form`.*

### B9. `tw_collision` put twice the atoms in each cloud — `config.py`
`n_clouds=4.0` with `n_per_cloud=5.0e3`, but `two_cloud_collision_psi()` builds
**two** clouds and renormalises the pair to `n_clouds * n_per_cloud`. So each
5e3 cloud was silently scaled to 1e4 atoms, and `depletion_fraction` was
measured against a target twice the real one — i.e. the TWA validity gate read
**half** the depletion the run actually had. Fixed to `2.0`.
*Test: `test_collision_rows_are_self_consistent`.*

### B10. `v_rel_real` was not the relative velocity — `config.py`
Each cloud's speed was `v_rel_real / v_split`, with `v_split=4` on the
collision rows and `2` on the nucleation ones — so on the `/4` rows the field
named "relative velocity" was **twice** the relative velocity actually
simulated. `v_split` is gone; each cloud always carries half of `v_rel_real`,
and the rows were halved to match, so **every row's dynamics are unchanged**
(verified: identical natural-unit half-velocities to the last digit). Only the
label is now true. See §D2 for what this reveals about the paper comparison.

### B11. `N=248` was an FFT-hostile grid size — `config.py`
`248 = 2^3 * 31`. FFT cost is set by `N`'s **largest prime factor**, and a
factor of 31 pushes cuFFT onto its Bluestein path. Measured on CPU: 11.2 ns per
point at 248 against 8.9 at 250 — ~25% wasted, worse on GPU — for a 0.8%
change in `dx`. The project's own `check_fft_grid_size()` warns about this and
suggests exactly 250; the config ignored it. Now `N=250 = 2 * 5^3`.
*Test: `test_configured_grid_sizes_are_fft_friendly`.*

### B12. `run_metadata.json` omitted the seed
The file is documented as "every knob that could change the answer, next to the
answer", and three scenarios seed from `int(time())` — so a run could not be
reproduced at all. Now records `seed`, the whole cutoff row,
`n_cutoff_modes`, `temperature_natural`, `n_particles`, `batch_size`,
`order4_dt_multiplier`, the collision geometry, `dt_accuracy_limit`, the FFT
box, and `healing_length`/`dx_over_xi`.

### B13. A regression test was passing by luck — `tests/test_diagnostics_memory_opt.py`
It compared every diagnostic against its own value at `rtol=1e-5`, including
`Lz`, which on the deliberately generic random field it uses comes out `~0.39`
against per-term sums of order `1e3` — **3.2 digits of cancellation out of
float32's ~7**. Measured against a float64 evaluation of the same integral,
the pre-optimisation reference (8.1e-5 absolute) and the current code (9.9e-5)
are equally far from the truth: the test was pinning a round-off pattern, not a
formula. Momentum and angular momentum are now compared against the scale of
their own triple — the same rule `gates.py` already applies to exactly these
quantities. Any real algebraic error is still O(1) against that scale and still
fails by four orders of magnitude.

---

## C. Optimisations

All verified to reproduce the previous results (see the table at the end).

### C1. Energy from one FFT instead of six — `gpe3d/physics.py`
`_energy()` differentiated `psi` along each axis in real space: three forward
plus three inverse FFTs and a complex64 `(N,N,N)` derivative. By Parseval,
`E_kin = (dV/N^3) sum_k (K^2/2)|psi_k|^2`, and `K^2` separates, so `|psi_k|^2`
reduces onto three length-`N` marginals. **One forward FFT, no inverse, no
derivative array.** Measured **6.3x faster** at `N=48`, 4.9x at `N=96`; agrees
to 2e-7. This sits on the imaginary-time convergence check and every gate's
critical path — ground-state prep is ~1.17x faster overall.

### C2. One shared forward FFT in `compute_diagnostics()`
The full report needs the current in real space (angular momentum weights `j`
by position), but it was transforming the *identical* `psi` once per axis.
Now transformed once and reused: **4 transforms instead of 6**, same peak
memory, arithmetic byte-identical to the reference.

### C3. Density slices no longer build the whole density — `gpe3d/evolve.py`, `gpe3d/ensemble.py`
`get_density_numpy()[n]` computed an `N^3` float32 array **and copied all of it
off the GPU** to keep one plane — 226 MB per movie frame at `N=384`, across
PCIe, once per recorded frame. Same in the ensemble runner
(`_density(state.psi)[:, n]`), inside the diagnostics peak that runner exists
to avoid. Both now slice `psi` first; added
`GPEPhysics3D.get_density_slice_numpy()`.
*Test: `test_density_slice_matches_the_full_density`.*

### C4. Half the phase work in the detector — `nucleation/winding.py`, `nucleation/detect.py`
Each plane family needs the phase differences along the two axes transverse to
its normal, so across all three families **every axis is needed exactly twice**
— and was computed twice. `arctan2` over the full grid is the detector's single
most expensive operation. `phase_differences()` computes the three once and
`winding_from_differences()` consumes them. With the smaller wins below,
`detect_frame()` on three plane families is **~1.5x faster**.
*Test: `test_shared_phase_differences_match_the_per_family_form`.*

### C5. Fewer full-grid temporaries in the detector
* `w / _TWO_PI` -> `w *= 1/_TWO_PI` (one `(N,N,N)` float32 saved per family).
* `rint(w)` was computed twice — once for `integrality_error`, once for the
  charge. Computed once and shared.
* `integrality_error` used `max(abs(w - rint(w)))`; now `max(max(d), -min(d))`
  on one array — two reductions instead of another full grid.
* `dilate`/`erode` accumulate in place (`|=`, `&=`). At `close_passes=4` the
  out-of-place form allocated **24 full boolean grids per call**, twice.
* `local_density_dip` builds its box filter in place and folds `factor` in.
*Tests: `test_morphology_is_unchanged_by_the_in_place_rewrite`,
`test_local_density_dip_does_not_mutate_its_input`,
`test_integrality_error_accepts_a_precomputed_rounding`.*

### C6. An FFT-friendly production grid
See B11 — ~25% of the FFT time in the default nucleation scenario, for free.

### Measured (CPU, N=96 unless stated)

| | before | after | |
|---|---|---|---|
| `_energy` | 69.2 ms | 14.2 ms | **4.9x** |
| `_energy` (N=48, x20) | 264 ms | 42 ms | **6.3x** |
| `compute_diagnostics` | 102 ms | 97 ms | 1.05x (4/6 transforms; larger on GPU) |
| `detect_frame`, 3 planes | 93.9 ms | 64.9 ms | **1.45x** |
| `winding_volume` | 21.4 ms | 13.5 ms | 1.58x |
| imaginary-time ground state (N=48) | 6.75 s | 5.77 s | 1.17x |
| FFT per point, N=248 -> 250 | 11.2 ns | 8.9 ns | **1.25x** |

Correctness: 65 physical quantities compared across a trapped run, a
collision, a TW ensemble and four imprinted detector fields. Every one agrees
to `<= 2e-6` of its own scale (float32 round-off from B7's exact `dx`); all
integer counts — `n_lines`, `n_pierce`, `n_pierce_{x,y,z}`, `n_lines_closed` —
are **bit-identical**. The three entries that look large in a naive relative
comparison are `Px` and `Lz` on a centred, non-rotating cloud, whose true value
is exactly zero (`1e-6` and `1e-9` absolute).

---

## D. Still open — decisions for you, not code

### D1. The trap anisotropy is inverted (unchanged, now confirmed)
The source paper has **z tight**: `w_x = 2*pi*4.57 Hz`,
`lambda_z/lambda_{x,y} = sqrt(8)`, colliding along the **wide** x.
`config.ANISOTROPY` encodes the opposite — `_WIDE = (1, 1/sqrt(8), 1/sqrt(8))`
makes x the *tightest* axis and collides along it. The correct target is
`OMEGA = (1.0, 1.0, ANISOTROPY)`. Not changed: every collision row's `L`, `N`
and velocity gate was tuned against the current geometry, so this needs a full
re-tune and re-gate, not a one-line swap.

### D2. The collision runs at half the paper's velocity
Newly visible once B10 removed `v_split`. The paper's wavepackets **separate at
4.0 mm/s**; the `collision` row runs 2.0 mm/s relative (it said `4.0e-3` but
`v_split=4` halved it), and `tw_collision` runs 1.0 mm/s. Doubling
`collision`'s `v_rel_real` to `4.0e-3` to match is **safe on resolution
grounds** — pre-computed: `k_char` goes 3.55 -> 7.10 against
`k_nyquist = pi/dx = 75.2` at `N=384, L=16`, so the 2x-margin gate still passes
with 5x to spare — but it changes the dynamics and `dt`, so it is your call and
needs a re-gate. Left as-is; the label is now honest either way.

### D3. The shipped default is the offset collision
`SCENARIO = "nucleation_collision"` ships with `impact_offset=(0.0, 3.0, 0.0)`.
A transverse offset breaks the symmetry by itself, so the **mean-field control
nucleates too** and the clean TW-vs-control A/B is lost. That is a "does
anything happen" probe, not a headline result. Set it to `(0,0,0)` for the A/B.

### D4. `n_traj=8` is a gate ensemble, not a statistics ensemble
Counts are small integers and roughly Poisson; 20–50 independent seeds are what
a quotable number needs. The bootstrap machinery is in place and correct.

### D5. Finite-`T` uses bare plane-wave energies
The largest open *physics* limitation — see `PHYSICS.md` §6.6. Fine while
`T* < mu*` (few-percent `T/Tc`); a typical 30% `T/Tc` experiment needs a
Bogoliubov-quasiparticle `mode_variance()`, not a bigger cutoff.

### D6. The Wigner ordering correction is an ensemble statement
`n_phys = <|psi_W|^2> - offset` holds **in the ensemble mean**. The detector
subtracts the same constant from a *single* trajectory, where it is only a
sensible bias correction, not an identity — individual cells can go negative.
It is the right thing to do (it stops the uniform noise floor dragging
`n_peak`), but it is a heuristic and worth stating in a write-up.
