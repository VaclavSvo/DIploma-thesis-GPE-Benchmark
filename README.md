# 3D GPE / Truncated-Wigner solver with vortex-nucleation diagnostics

Natural units (`hbar = m = 1`):

    i dpsi/dt = ( -1/2 grad^2 + V(r) + g|psi|^2 ) psi

Spectral split-step Fourier propagation (Strang or Yoshida), imaginary-time
ground states, Truncated-Wigner noise as an initial condition, and
phase-winding vortex detection on top. Runs on CuPy (GPU) when it is installed
and NumPy (CPU) otherwise, with no code changes either way.

**Three documents.** This one is *how to run it and where everything lives*.
[`PHYSICS.md`](PHYSICS.md) is *what it computes and which paper each number
comes from* — read it before quoting anything. [`AUDIT.md`](AUDIT.md) is the
September 2026 correctness pass: what was broken, what was verified correct,
what is still an open decision.

---

## Run it

```bash
python run.py          # reads config.py's SCENARIO
```

Everything is set in `config.py`: pick `SCENARIO`, edit that row of
`SCENARIOS`, run. No command-line flags. Nothing else in the project needs
editing to change a run.

For vortex work, run the control too — `SCENARIO = "nucleation_control"`,
nothing else touched. **The control is what makes the TW numbers mean
anything**: identical parameters with the noise off must give zero vortices. If
it does not, the detector is firing on something that is not a vortex.

```bash
python -m pytest tests/ -q     # ~2 min on CPU, no GPU needed
```

---

## Layout

| file | what |
|---|---|
| `config.py` | every parameter, as one scenario table |
| `run.py` | scenario switch: pre-flight, prepare, evolve, gate, write outputs |
| `gpe3d/backend.py` | `xp` = cupy or numpy, and the project's only FFT entry points |
| `gpe3d/base.py` | the 4-slot solver contract (`units`, `init_state`, `call`, `get_derived_quantities`) |
| `gpe3d/physics.py` | `GPEPhysics3D`: grid, imaginary time, propagators, diagnostics |
| `gpe3d/potentials.py` | potential builders (harmonic trap) |
| `gpe3d/solver.py` | classical solvers: trapped condensate, colliding condensates |
| `gpe3d/tw_solver.py` | the same two geometries plus TW noise |
| `gpe3d/noise.py` | TW noise sampler, cutoff mask, ordering-correction offset |
| `gpe3d/ensemble.py` | chunked TW ensemble runner |
| `gpe3d/evolve.py` | the block-evolution + recording loop |
| `gpe3d/gates.py` | conservation and resolution gates, pass/fail |
| `gpe3d/units.py` | SI -> natural units |
| `nucleation/` | vortex detection, statistics, movies (reads `psi`, never changes it) |
| `tests/` | see [Tests](#tests) |

The solver contract and the gates are fixed infrastructure — extend them, don't
rewrite them per scenario. A new scenario is a solver class plus one row in
`config.SCENARIOS` and one in `run.SOLVERS`.

Inside `nucleation/`:

| file | what |
|---|---|
| `winding.py` | plaquette phase winding, masks, morphology — the primitive everything is built on |
| `detect.py` | winding fields -> pierce points -> vortex lines, counts, line length |
| `observer.py` | the per-frame recorder that plugs into both evolution loops |
| `settings.py` | `DetectorConfig`, and how `config.py` maps onto it |
| `statistics.py` | aggregation, bootstrap CIs, onset times, control comparison |
| `visualise.py` | the density+winding movie, and the frame-rate/writer helpers |
| `vortex_map.py` | three-view core map from the saved points alone, with a CLI |
| `synthetic.py` | analytically imprinted fields with known vortex content (the test ground truth) |
| `io.py` | CSV / npz / metadata writing and round-tripping |
| `runner.py` | glue between `run.py` and the above |

---

## Scenarios

| SCENARIO | what |
|---|---|
| `trapped_dipole` | one condensate, dimensionless `g=1`, offset/kicked. Fast propagator sanity check (Kohn mode). |
| `collision` | two condensates released from a trap and collided in free space, scaled to Norrie, Ballagh & Gardiner, PRA **73**, 043617 (2006) |
| `tw_trapped_dipole` | trapped condensate + TW vacuum noise |
| `tw_collision` | the collision geometry + TW vacuum noise |
| `tw_trapped_thermal` | the trapped scenario at finite (low) temperature |
| `nucleation_collision` | `tw_collision` tuned to nucleate vortices, with the detector attached |
| `nucleation_control` | the same parameters, noise off, same detector — must give zero |

Collision parameters are physically scaled (`23-Na`, the paper's real trap
frequency, real scattering length, real collision velocity) but with the atom
number scaled down from the paper's ~1e6/cloud to something that fits a laptop
GPU. Expect the same qualitative physics — colliding clouds, matter-wave
interference fringes on overlap — not quantitative parity until both atom count
and resolution go up.

The unit derivation itself is now **verified against the source paper**: this
project's `g = 3.5233e-3` reproduces the paper's dimensionless `U0~ = 1e-2` to
0.35% once the different length-unit convention is accounted for
(`PHYSICS.md` §2.1). What is *not* matched is the trap geometry and the
collision speed — see [Known flags](#known-flags).

---

## Configuration reference

Everything below is in `config.py`. The scenario table row comes first; the
shared settings below it apply to every run.

### Scenario row

| key | meaning |
|---|---|
| `solver` | `"trapped"` or `"collision"` — which geometry |
| `tw` | `True` adds Truncated-Wigner noise (selects a TW solver class) |
| `physical` | `True` derives `g` from `23-Na` + `OMEGA_REF_HZ`; `False` uses a bare `G` |
| `omega` | trap frequencies `(wx, wy, wz)` in units of `OMEGA_REF_HZ` |
| `n_particles` / `n_per_cloud`, `n_clouds` | atom number (total, or per cloud x cloud count) |
| `v_rel_real` | the clouds' **relative** closing speed in m/s. Each cloud gets half, always. |
| `separation` | `(x,y,z)` between cloud centres; half displaces each cloud, opposite signs |
| `impact_offset` | `(x,y,z)` impact parameter, added to `separation` the same way. **A transverse component breaks the symmetry by itself, so the mean-field control nucleates too.** |
| `N`, `L`, `t_total` | grid points per axis, nominal box edge, simulated time |
| `n_traj`, `batch_size` | TW ensemble size, and how many trajectories are resident at once |
| `cutoff`, `cutoff_n_ref`, `cutoff_exponent`, `cutoff_nyquist_safety` | the noise UV cutoff and its grid scaling (`PHYSICS.md` §6.4) |
| `temperature_fraction_tc` | `T/Tc`; absent or 0 means vacuum noise only |
| `seed` | RNG seed. Several rows use `int(time())` — recorded in `run_metadata.json` so a run stays reproducible. |
| `order4_dt_multiplier` | this row's order-4 `dt` factor (see [Integrator and dt](#integrator-and-dt)) |

### Shared settings

| key | meaning |
|---|---|
| `IMAG_TIME_STEPS` | ground-state relaxation cap; raise if "converged at step ..." looks premature |
| `SPLITSTEP_ORDER` | `2` = Strang `O(dt^2)`, `4` = Yoshida `O(dt^4)` at 3x work per step |
| `TOL` | pass/fail tolerance on the conserved quantities |
| `SAVE_GIF` | mid-plane density movie for non-nucleation runs |
| `GIF_SECONDS`, `GIF_FPS` | movie playback length and rate — independent of `T_TOTAL` |
| `SNAPSHOT_ROWS` | report frames when `SAVE_GIF` is off (`3 *` this) |
| `OUT_DIR`, `VERBOSE` | output directory, log verbosity |

### Detector (`NUCLEATION_*`, nucleation scenarios only)

| key | meaning |
|---|---|
| `MASK_THRESHOLD` | `\|psi\|^2 > f * n_peak` is "inside the cloud". **A convergence parameter — sweep it.** |
| `MASK_CLOSE_PASSES` | morphological closing of that mask, in cells. Must exceed the core radius `~xi/dx`. |
| `REQUIRE_DENSITY_DIP`, `DIP_FACTOR` | also require a local density minimum at the core |
| `NORMALS` | which plane families to sweep. All three, or rings in one plane are missed. |
| `INTEGRALITY_TOL` | abort if windings are not integer to within this — the grid's own resolution check |
| `LINK_CUTOFF_DX`, `CLOSE_CUTOFF_DX` | pierce points this close are one line; traced ends this close close a ring |
| `DETECT_STRIDE`, `SLICE_STRIDE` | detect every N-th recorded frame / record a movie frame every N-th. **`DETECT_STRIDE` is the cost knob**, not the frame count. |
| `SLICE_PLANES`, `SLICE_TRAJECTORY` | which mid-planes the movie shows, and which trajectory both movies follow |
| `VORTEX_MAP` | also write `vortex_map.gif` from the saved points |

---

## Movie length

Movie length is set by `GIF_SECONDS` and `GIF_FPS` and is completely
independent of how much physical time the run covers. `run.py` works out the
snapshot cadence and reports it before evolving:

```
dt=1.3966e-03 (accuracy limit 1.5656e-03, shrunk onto 716 whole steps)  total_steps=716
-- Movie schedule --
6s at 30fps = 180 frames, showing T_TOTAL=1 of simulated time (0.167x speed)
recording 180 snapshots, one every 4 of the 716 computed sub-steps
(dt_frame=5.5866e-03, last frame at t=1)
```

Every sub-step is computed as usual; the recorder samples 180 of them.

**dt is shrunk onto the frame grid.** `choose_dt` returns an accuracy *ceiling*,
and `T_TOTAL/dt` is generally not divisible by the number of frames. Rather
than rounding the steps-per-frame — which rounds *down* to 1 whenever the ratio
is under 1.5, silently ending the run short of `T_TOTAL` (a `T=1` run at
`dt=3.7e-3` asking for 180 frames used to stop at `t=0.669`, a third short, and
report success) — the schedule rounds steps-per-frame **up** and then sets
`dt = T_TOTAL/total_steps`. That is always at or below the accuracy ceiling,
lands the last frame exactly on `T_TOTAL`, and makes the frames exactly
equidistant in sub-steps.

**The ceiling is one snapshot per sub-step.** The run only passes through
`T_TOTAL/dt` distinct states. Ask for more and `run.py` records every state
there is and says so:

```
NOTE: only 57 distinct states exist at this dt (T_TOTAL/dt = 56 sub-steps) --
the renderer cross-fades them up to 180, so the movie still runs 6s but only 57
frames carry new data. Lower SPLITSTEP_ORDER to 2, or shorten GIF_SECONDS, for
more real frames.
```

This bites hardest at `SPLITSTEP_ORDER = 4`, where the whole point of the
Yoshida composition is a much larger `dt`. Cross-fading is a blend, not a
physical interpolation: every frame carrying real data is a recorded one.

Timing accuracy: all three movies go through ffmpeg when it is installed, which
writes exact frame delays. Without it the Pillow fallback quantises each delay
to a whole centisecond, so rates that do not divide 100 (30 fps among them)
come out a few percent off. Install ffmpeg, or use 10/20/25/50 fps.

For a nucleation run the frame count is **not** the cost driver — detection is,
at three full-grid winding sweeps per trajectory per frame.
`NUCLEATION_DETECT_STRIDE` decouples the two:

```
detecting on 31/180 recorded frames (stride 6) x 4 trajectories = 124 detections
```

---

## Correctness notes worth knowing

Full derivations in [`PHYSICS.md`](PHYSICS.md); these are the traps.

**The kinetic factor.** `T = -1/2 grad^2` has Fourier eigenvalue `0.5*K_sq`, so
the half-dt propagator is `exp(-1j*0.25*dt*K_sq)`. A coefficient of `0.5` there
— the convention in the 2D CuPy reference notebook this was ported from —
silently implements `T = -grad^2`, i.e. half the intended mass, despite that
notebook's own text stating `-1/2 grad^2`. Fixed here; the noninteracting
ground state now matches the analytic QHO result to four significant figures.

**Aliasing is invisible to the conservation gates.** A velocity kick is a plane
wave `exp(i v.x)`; if `|v|` exceeds `pi/dx` the FFT grid aliases it — and
aliasing is still exactly unitary, so norm and energy stay green while the
dynamics are silently wrong. The first attempt at the collision scenario hit
exactly this: all gates passed while the clouds barely moved. `run.py` runs
`gates.resolution_gate()` as a pre-flight check before building anything
expensive, and aborts. **This is the one failure mode no conservation gate can
catch**, which is why the gate's `dx` must be the engine's `dx` — see below.

**`dx = L/(N-1)`, not `L/N`.** The grid is `linspace(-L/2, +L/2, N)` with both
endpoints kept, so the FFT's periodic box is `N*dx = L*N/(N-1)` — 0.4% larger
than the nominal `L` at `N=250`. Everything downstream is built from `dx` and
`dV = dx^3` and is self-consistent. Use `gpe3d.physics.grid_spacing(N, L)`;
recomputing it as `L/N` anywhere makes the aliasing gate above too generous.

**Momentum gates only apply where they are meaningful.** Free space (`V=0`) is
translation invariant, so a noiseless collision genuinely conserves momentum.
A trapped cloud does so only at rest and centred; displace or kick it and
momentum oscillates (Ehrenfest, and Kohn for the dipole mode), which is correct
physics, not drift. `run.py` disables those gates automatically for displaced,
kicked and TW runs; norm and energy are always checked.

**Getting actual dynamics.** A ground state is stationary — `|psi|^2` does not
move, only its global phase does. `INITIAL_OFFSET` displaces it (Kohn's
theorem: it then sloshes at exactly `OMEGA` regardless of `g`, a clean analytic
check); `INITIAL_VELOCITY` gives it a uniform push. Both are exact spectral
operations (Fourier shift theorem, Galilean phase boost), no interpolation
error.

**One healing length.** `xi = 1/sqrt(2*g*n)` project-wide — the convention the
noise cutoff is built on, so `E_k(1/xi) = g*n` exactly. `run.py`'s printed
`dx/xi` and its "cores resolved" verdict used to use `1/sqrt(g*n)` and were
optimistic by `sqrt(2)` (`AUDIT.md` §B3).

---

## Truncated Wigner

Every TW trajectory obeys the *same deterministic GPE* — the propagator is
untouched. All the stochasticity is in the initial condition: the mean-field
ground state plus one draw of noise per trajectory. Theory, derivation and
sources in [`PHYSICS.md`](PHYSICS.md) §6; the operational summary:

**Noise model.** Each `k`-mode below an energy cutoff carries an independent
complex Gaussian amplitude with `<|alpha_k|^2> = n_k(T) + 1/2 =
(1/2)coth(E_k/2T)`, `E_k = 0.5*K_sq`. `T=0` gives the flat vacuum `1/2`
exactly; `k=0` is always the flat `1/2` at any `T`.

**The UV cutoff is the point.** Without one, total noise energy grows with the
number of grid modes — unphysical and grid-dependent. `E_cut` is tied to the
interaction energy scale (`k_cut ~ 1/xi`), and the grid must then be fine
enough to resolve everything below it. The multiplier is the one free
literature parameter here; it is meant to be swept, not trusted.

**The ordering correction is the classic TW bug.** Raw `<|psi_W|^2>` is
Wigner-ordered and contaminated by the noise's own particle content:

    n_phys(r) = <|psi_W(r)|^2>_ensemble - offset_density

`offset_density = sum_k <|alpha_k|^2> / V`, which at `T=0` is
`N_cutoff_modes/(2V)`. It is **not** the flat `1/(2*dV)` you will see quoted —
that holds only when the cutoff spans every grid mode. For a genuine partial
cutoff the flat value over-subtracts by `N^3/N_cutoff_modes`, making a healthy
run look like it is losing particles it never had.

**Depletion is the validity gate.** `depletion_fraction = n_vacuum_offset /
n_particles_target` must be percent-level; order-1 means the run is outside
TWA's regime regardless of what norm and energy say. `run.py` prints it and
fails the run above 10%. At the dimensionless `g=1, N_PARTICLES=1` test
convention every gate passed with 950% depletion — that convention was only
ever meant to validate the noiseless propagator.

Measured depletion against grid and temperature, and the reasoning behind
`cutoff_exponent` and `cutoff_n_ref`, are tabulated in `PHYSICS.md` §6.5–6.6.
Short version: at `N_ref=128, p=0.85, g~3.5e-3, N=1e4` the 10% gate starts
failing around `N=224`, and finite `T/Tc` past a few percent leaves the
sampler's validity range. Both are TWA's boundary, not bugs.

---

## Integrator and dt

`SPLITSTEP_ORDER = 2` is Strang, `O(dt^2)`. `4` is Yoshida's composition of
three Strang sub-steps, `O(dt^4)` at 3x the work per outer step — so it pays
only if it affords more than 3x the `dt`. The middle sub-step has a *negative*
tau, which is unavoidable above 2nd order and harmless here.
`tests/test_existence.py::test_yoshida4_convergence_order` checks the
convergence *rate*, which is what a coefficient or sign error would break while
still looking numerically plausible.

`choose_dt()` sizes `dt` so the phase accumulated per step stays within ~0.2 rad
for both the fastest Fourier mode and the interaction term at peak density.
This is an accuracy heuristic, not a stability condition — split-step is
norm-preserving for any `dt`.

**The order=4 multiplier needs care.** Two separate numbers, deliberately:

- `physics._ORDER4_DT_MULTIPLIER = 400`, calibrated on the smooth noiseless
  collision scenario. Swept on real hardware: `N=128` passed up to 24, `N=256`
  up to 248, `N=512` passed at 400 and FAILED at 1000 (3.3% energy drift — the
  gate caught it as designed). 400 is the largest tested value that passed,
  with no deliberate headroom.
- `tw_solver._TW_ORDER4_DT_MULTIPLIER_DEFAULT = 50`, because that calibration
  does not transfer. Vacuum noise puts real amplitude on modes right against
  the cutoff — exactly the high-`k` content a large `dt` resolves worst. At
  `N=32, order=4`, the global 400 fails the energy gate with 18–51% drift,
  *non-monotonically* in the multiplier, while <=25 passes with large margin.

And a caution that cost a production run: `tests/sweep_order4_dt.py` with its
short `T_SWEEP=1.0` window is **not** sufficient certification. A multiplier of
400 passed that sweep cleanly for `tw_trapped_thermal` and then blew up on a
full `T_TOTAL=5` run — 24,394% energy drift, a genuine instability the short
window never reached. Binary-search a multiplier with full-length runs only.

---

## Vortex nucleation

`gpe3d/` evolves the field exactly as before; `nucleation/` only reads `psi`.
Algorithm and sources in [`PHYSICS.md`](PHYSICS.md) §7.

### How detection works

A vortex is a line about which the phase winds by `2*pi`. For each grid
plaquette, sum the four wrapped phase differences `arg(psi_q * conj(psi_p))` —
always the product form, never `arg(psi_q) - arg(psi_p)`, which needs branch
handling and silently produces garbage. The result is an integer for a resolved
core, which is a free resolution check: `detect.pierce_points()` raises if it
is not.

All three plane families are swept. A line is found by the planes it *pierces*,
so `z`-planes alone would miss a ring lying in the `y-z` plane entirely —
`tests/test_detect.py` pins exactly that failure. The three per-axis phase
differences are computed once and shared across the families.

Pierce points from all three families merge into one 3D cloud, are grouped into
connected components (= vortex lines), and each line is traced, measured and
classified as open or closed. Both `N_vortex(t)` and total line length `L(t)`
are reported, but **do the statistics on `L(t)`**: integer counts are small and
jumpy, line length varies smoothly and is the standard quantum-turbulence
observable.

One non-obvious trap in the masking: a resolved core has `|psi|^2 -> 0` at its
centre, so a raw density mask deletes exactly the plaquettes worth counting.
The mask is morphologically *closed* (`NUCLEATION_MASK_CLOSE_PASSES`, in cells)
to fill those holes without moving the cloud boundary outward. In the other
direction, masking is not optional: outside the condensate the phase is pure
noise, and in a TW run the entire box carries vacuum noise, so an unmasked
count returns thousands of spurious vortices.

Detection runs **per trajectory**, before any averaging. Vortices nucleate in
different places in every trajectory, so the ensemble-mean density is a smooth
blob with no cores in it at all. This is the failure mode that silently
produces a null result that looks like physics.

### Outputs

Everything lands in `outputs/<scenario>/`:

| file | what |
|---|---|
| `density_winding.gif` | the close-up. One column per plane: `\|psi\|^2` on top, phase winding below, cores ringed and sign-coloured with a live count per panel. **One trajectory** (`NUCLEATION_SLICE_TRAJECTORY`), never an ensemble average. |
| `vortex_map.gif` | the overview. Every detected core in the whole box, viewed down each axis, plus core count against time. Also one trajectory — the same one. What stays readable once a tangle has hundreds of lines in it. |
| `observables.csv` | one row per (trajectory, frame): counts, line lengths, integrality error |
| `pierce_points.npz` | every core's `(x, y, z, charge, plane)` plus its trajectory and time, ~kB/frame |
| `run_metadata.json` | every knob that could change the answer — including the RNG seed and the whole noise-cutoff row — next to the answer |

Both movies obey `GIF_SECONDS` / `GIF_FPS`. The winding panels are *not*
cross-faded: a half-integer winding is meaningless, so the nearest recorded
frame is held, and cores step while the density flows.

The winding panel masks against the same threshold the counter uses, but does
**not** apply the counter's 3D density-dip filter — so it is an *upper bound*
on what the count accepts. `observables.csv` is the number; the panel is where
it is happening.

`vortex_map.gif` needs only `pierce_points.npz`, so any finished run re-renders
afterwards — different frame rate, different views, a different trajectory, no
GPU:

```bash
python -m nucleation.vortex_map outputs/nucleation_collision/pierce_points.npz \
       better.gif --half-box 12 --fps 60 --views xz --bins 192 --trajectory 3
```

That is why points are stored instead of fields: at `N=384` one complex field
is ~450MB, and a whole run's cores are a few MB.

Compare two finished runs without re-running anything:

```python
from nucleation.runner import compare_runs
print(compare_runs("outputs/nucleation_collision/observables.csv",
                   "outputs/nucleation_control/observables.csv"))
print(compare_runs(..., ..., key="L_total"))   # works on any observable
```

### What a count needs before it means anything

Three knobs can manufacture vortices. A count that is not stable under all
three is not a result:

1. **The noise cutoff** — raise it far enough and vortices always appear, out
   of unphysically large noise. Watch the `depletion_fraction` `run.py` prints;
   percent-level is required, order-1 is outside TWA.
2. **`NUCLEATION_MASK_THRESHOLD`** — sweep it.
   `nucleation.runner.sweep_mask_threshold()` does this on one stored frame, so
   it costs no GPU time and there is no excuse for skipping it.
3. **Grid `N`** — cores need `dx` well below the healing length; `dx <~ xi/3`.
   `run.py` prints `dx/xi`.

Plus the protocol: **20–50 independent seeds**, not the 8 the configs ship with
(fine for conservation gates, useless for nucleation statistics). Counts are
small integers and roughly Poisson, so quote bootstrap intervals, not `+-SEM`
from a handful of trajectories. Record onset `t_nuc` with a sustain requirement
— with `sustain=1` the onset histogram is dominated by the earliest false
positive in each trajectory. And run the mean-field control, which must give
zero.

### If nothing nucleates

Two clouds closing at `+-v_half` interfere with fringe spacing `pi/v_half`; for
the resulting grey solitons to be snake-unstable (and decay into vortex rings)
that spacing wants to be a few healing lengths. Scan `v_rel_real` upward by
factors of 2 — the resolution gate catches the point where the grid can no
longer represent the kick, so the scan fails loudly rather than aliasing
silently. And check `T_TOTAL`: the instability needs several `xi/c_s` *after*
the clouds overlap, so a run that stops at overlap shows fringes and no
vortices, which reads as a null result but is just an early stop.

An offset collision — `impact_offset=(0.0, 3.0, 0.0)`, which is what
`nucleation_collision` currently ships — shears the clouds past each other and
nucleates lines rather than rings, more robustly. But the offset breaks the
symmetry by itself, so **the mean-field control nucleates too and the clean
TW-vs-control A/B is lost**. Set it to `(0,0,0)` for the headline result.

---

## Tests

No GPU needed; the whole suite runs on CPU in about two minutes.

```bash
python -m pytest tests/ -q          # or run any file directly
python tests/test_winding.py        # imprinted vortex -> ±1; soliton/sound/noise -> 0
python tests/test_detect.py         # ring -> one closed line of the right circumference
python tests/test_observer.py       # detection while the GPE actually propagates
python tests/test_statistics.py     # bootstrap CIs, sustained-onset rule, control check
python tests/test_existence.py      # gates + analytic QHO ground state + Yoshida order
python tests/test_regressions.py    # one test per bug fixed in AUDIT.md
python tests/smoke_nucleation.py    # both nucleation scenarios end to end at N=32
```

Most of the risk in the vortex work lives in `test_winding.py` and
`test_detect.py`, and neither needs GPU time — run them before any long run.
`nucleation/synthetic.py` builds the imprinted fields they check against:
straight vortices, rings, and the nulls that matter (a grey soliton looks
exactly like a vortex in a density slice, and a count that cannot tell them
apart is worthless).

`tests/test_diagnostics_memory_opt.py` reimplements the pre-optimisation
diagnostics formula byte-for-byte and checks the current one against it —
stronger than a gate test, which would pass a subtly wrong rewrite that still
conserved energy to 1%.

`tests/` also holds instruments rather than tests: `sweep_order4_dt.py`,
`bench_perf.py`, `bench_orders_collision.py`, `probe_thermal_depletion.py`,
`tw_collision_noise_diff.py`.

---

## Performance and memory

See [`performance.md`](performance.md) for the measured numbers, the
per-grid-point memory budget, and what the September 2026 optimisation pass
changed. Headlines:

- Written for a 6 GB GTX 1660 Ti; measured **~45.5 B/point** for a TW scenario
  at order 4 with `batch_size=1`.
- Prefer grid sizes whose prime factors are all small (2, 3, 5, 7, 11, 13).
  FFT cost is dominated by `N`'s largest prime factor, not its size; `run.py`
  warns (does not abort) on a bad `N`.
- `_energy()` is ~5-6x faster than before (one FFT instead of six, by
  Parseval); the vortex detector is ~1.5x faster (shared phase differences);
  movie frames no longer copy the whole `N^3` density off the GPU.

---

## Known flags

Open decisions, not bugs — each needs a re-tune or a judgement call, so none
was changed silently. Full detail in [`AUDIT.md`](AUDIT.md) §D.

- **Trap anisotropy is inverted.** The source paper has z as the tight axis
  (`w_z = sqrt(8) w_x`) and collides along the wide x; `config.ANISOTROPY`
  encodes the opposite, making x the tightest axis and colliding along it. The
  correct target is `OMEGA = (1.0, 1.0, ANISOTROPY)`. Every collision row's
  `L`/`N`/velocity gate was tuned against the current geometry, so fixing it
  needs a full re-tune and re-gate.
- **The collision runs at half the paper's velocity.** The paper's wavepackets
  separate at 4.0 mm/s; the `collision` row runs 2.0 mm/s relative and
  `tw_collision` 1.0 mm/s. (It used to *say* 4.0e-3, but a `v_split=4` divisor
  halved it; `v_split` is gone and the labels are now honest, with identical
  dynamics.) Doubling `collision`'s `v_rel_real` to `4.0e-3` is safe on
  resolution grounds — `k_char` 3.55 -> 7.10 against `k_nyquist = 75.2` — but
  changes the dynamics and needs a re-gate.
- **`nucleation_collision` ships with a transverse `impact_offset`.** Good as a
  "does anything happen" probe; not a headline result, because the control
  nucleates too.
- **`tw_trapped_dipole` at `N=384`** is past its validated range — depletion is
  already 13% at `N=256`, over the gate.
- **Finite `T` uses bare plane-wave energies**, not Bogoliubov quasiparticles.
  Fine while `T* < mu*`; a typical 30% `T/Tc` experiment needs a different
  `mode_variance()`. This is the largest open physics limitation.
- **A single trajectory's ordering correction is a bias correction, not an
  identity.** `n_phys = <|psi_W|^2> - offset` holds in the ensemble mean; the
  detector subtracts it per trajectory, which is the right thing to do but is a
  heuristic worth stating in a write-up.
