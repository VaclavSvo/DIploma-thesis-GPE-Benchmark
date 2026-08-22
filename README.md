# 3D GPE / Truncated-Wigner solver with vortex nucleation diagnostics

Natural units (hbar = m = 1):

    i dpsi/dt = (-1/2 grad^2 + V(r) + g|psi|^2) psi

Spectral split-step Fourier propagation (Strang or Yoshida), imaginary-time
ground states, Truncated-Wigner noise as an initial condition, and phase-winding
vortex detection on top. Runs on CuPy (GPU) when it is installed and NumPy (CPU)
otherwise, with no code changes either way.

## Run it

```bash
python run.py          # reads config.py's SCENARIO
```

Everything is set in `config.py`: pick `SCENARIO`, edit that row of `SCENARIOS`,
run. No command-line flags. Nothing else in the project needs editing to change a
run.

For vortex work, run the control too — `SCENARIO = "nucleation_control"`, nothing
else touched. **The control is what makes the TW numbers mean anything**:
identical parameters with the noise off must give zero vortices. If it does not,
the detector is firing on something that is not a vortex.

## Layout

| file | what |
|---|---|
| `config.py` | every parameter, as one scenario table |
| `run.py` | scenario switch: prepare, evolve, gate, write outputs |
| `gpe3d/backend.py` | `xp` = cupy or numpy, and the project's only FFT entry points |
| `gpe3d/base.py` | the 4-slot solver contract (`units`, `init_state`, `call`, `get_derived_quantities`) |
| `gpe3d/physics.py` | `GPEPhysics3D`: grid, imaginary time, propagators, diagnostics |
| `gpe3d/potentials.py` | potential builders (harmonic trap) |
| `gpe3d/solver.py` | classical solvers: trapped condensate, colliding condensates |
| `gpe3d/tw_solver.py` | the same two geometries plus TW noise |
| `gpe3d/noise.py` | TW noise sampler, cutoff mask, ordering-correction offset |
| `gpe3d/ensemble.py` | chunked TW ensemble runner |
| `gpe3d/evolve.py` | the block-evolution + recording loop |
| `gpe3d/gates.py` | conservation gates, pass/fail |
| `gpe3d/units.py` | SI -> natural units |
| `nucleation/` | vortex detection, statistics, movies (reads psi, never changes it) |
| `tests/` | see [Tests](#tests) |

The solver contract and the gates are fixed infrastructure — extend them, don't
rewrite them per scenario. A new scenario is a solver class plus one row in
`config.SCENARIOS` and one in `run.SOLVERS`.

## Scenarios

| SCENARIO | what |
|---|---|
| `trapped_dipole` | one condensate, dimensionless g=1, offset/kicked. Fast propagator sanity check. |
| `collision` | two condensates released from a trap and collided in free space, scaled to Norrie, Ballagh & Gardiner, PRA **73**, 043617 (2006) |
| `tw_trapped_dipole` | trapped condensate + TW vacuum noise |
| `tw_collision` | the collision geometry + TW vacuum noise |
| `tw_trapped_thermal` | the trapped scenario at finite (low) temperature |
| `nucleation_collision` | `tw_collision` tuned to nucleate vortices, with the detector attached |
| `nucleation_control` | the same parameters, noise off, same detector — must give zero |

Collision parameters are physically scaled (Na-23, real trap frequency, real
scattering length, real collision velocity) but with the atom number scaled down
from the paper's ~1e6/cloud to something that fits a laptop GPU. Expect the same
qualitative physics — colliding clouds, matter-wave interference fringes on
overlap — not quantitative parity until both atom count and resolution go up.

The paper's numbers were extracted by an automated summariser, not by a careful
manual read. Treat the scattering length, the trap anisotropy convention and the
interaction prefactor as "best effort, cross-check yourself"; the atom number,
reference trap frequency and collision velocity are more directly stated.

## Correctness notes worth knowing

**The kinetic factor.** `T = -1/2 grad^2` has Fourier eigenvalue `0.5*K_sq`, so
the half-dt propagator is `exp(-1j*0.25*dt*K_sq)`. A coefficient of `0.5` there
— the convention in the 2D CuPy reference notebook this was ported from —
silently implements `T = -grad^2`, i.e. half the intended mass, despite that
notebook's own markdown stating `-1/2 grad^2`. Confirmed numerically: at 0.5 a
free-particle Fourier mode matches `exp(-i*k^2*dt)` to 1e-16, not
`exp(-i*0.5*k^2*dt)`. Fixed here; the noninteracting ground state now matches the
analytic QHO result (E0 = 1.5*omega, sigma = 1/sqrt(2*omega)) to 4 significant
figures, and the gates drift ~0.02% against a 1% tolerance. The 2D notebook's own
conserved-quantity checks would not have caught this wherever kinetic and
potential energy do not exchange much, which is likely why it went unnoticed —
but any 2D run where a wavepacket exchanges energy with a potential is running at
the wrong effective mass.

**Aliasing is invisible to the conservation gates.** A velocity kick is a plane
wave `exp(i*v*x)`; if `|v|` exceeds the grid's Nyquist wavenumber `pi/dx`, the FFT
grid aliases it to a different, generally wrong value — and aliasing is still
exactly unitary, so norm and energy stay green while the dynamics are silently
wrong. The first attempt at the collision scenario hit exactly this: all gates
passed while the clouds barely moved, aliased to nearly zero velocity. `run.py`
now runs `gates.resolution_gate()` as a pre-flight check before building anything
expensive, and aborts with a clear message.

**Momentum gates only apply where they are meaningful.** Free space (V=0) is
translation invariant, so a noiseless collision genuinely conserves momentum and
angular momentum. A trapped cloud does so only at rest and centred; displace or
kick it and momentum oscillates (Ehrenfest, and Kohn for the dipole mode), which
is correct physics, not drift. `run.py` disables those gates automatically for
displaced, kicked and TW runs; norm and energy are always checked.

**Getting actual dynamics.** A ground state is stationary — `|psi|^2` does not
move, only its global phase does. `INITIAL_OFFSET` displaces it (Kohn's theorem:
it then sloshes at exactly `OMEGA` regardless of `g`, a clean analytic check);
`INITIAL_VELOCITY` gives it a uniform push. Both are exact spectral operations
(Fourier shift theorem, Galilean phase boost), no interpolation error.

## Truncated Wigner

Every TW trajectory obeys the *same deterministic GPE* — the propagator is
untouched. All the stochasticity is in the initial condition: the mean-field
ground state plus one draw of noise per trajectory.

**Noise model** (Sinatra, Lobo & Castin, J. Phys. B **35**, 3599 (2002); the same
grid/plane-wave method as Norrie/Ballagh/Gardiner). Each k-mode below an energy
cutoff carries an independent complex Gaussian amplitude with

    <|alpha_k|^2> = n_k(T) + 1/2 = (1/2)*coth(E_k / 2T),   E_k = 0.5*K_sq

T = 0 gives the flat vacuum 1/2 exactly. `k = 0` is always the flat 1/2 at any T:
it is the condensate's own mode, already normalised into the mean field, and
`n_k(T)` diverges as `E_k -> 0`.

**The UV cutoff is the point.** Without one, total noise energy grows with the
number of grid modes, which is unphysical and grid-dependent — refining the grid
would change the answer. `E_cut` is instead tied to the interaction energy scale
(the healing length `xi = 1/sqrt(2*g*n_peak)`), and the grid must then be fine
enough to resolve everything below it, not the other way around. The multiplier
is the one free literature parameter here (papers use O(1–few)); it is meant to
be swept, not trusted.

**The ordering correction is the classic TW bug.** Raw `<|psi_W|^2>` is
Wigner-ordered and contaminated by the noise's own particle content. The physical
density is

    n_phys(r) = <|psi_W(r)|^2>_ensemble - offset_density

where `offset_density = sum_k <|alpha_k|^2> / V`, which at T=0 is
`N_cutoff_modes/(2V)`. It is **not** the flat `1/(2*dV)` you will see quoted —
that only holds when the cutoff spans every grid mode. For a genuine partial
cutoff the flat value over-subtracts by `N^3/N_cutoff_modes`, making a healthy run
look like it is losing particles it never had.

**Depletion is the validity gate.** `depletion_fraction = n_vacuum_offset /
n_particles_target` must be percent-level; order-1 means the run is outside TWA's
regime regardless of what norm and energy say. `run.py` prints it and fails the
run above 10%. This matters: at the dimensionless `g=1, N_PARTICLES=1` test
convention, every gate passed with 950% depletion — that convention was only ever
meant to validate the noiseless propagator.

### Cutoff scaling with resolution

`scale_cutoff_to_grid` lets a finer grid admit more modes instead of leaving the
extra grid points unused above a fixed cutoff:

    E_cut = max(E_floor, min(E_floor * (N/N_ref)**exponent, E_nyquist/safety**2))

The growth law is a small *power* of N/N_ref on purpose. Mode count scales as
`E_cut**1.5` in 3D, so a cutoff parametrised as any sizeable fraction of
`E_nyquist` explodes: measured at g~3.5e-3, N=96, L=8, `E_nyquist` is ~260x
`E_floor`, and filling to it admitted 34% of all grid modes for a 1503%
depletion fraction. `N == N_ref` reproduces the fixed cutoff exactly, which is why
`N_ref` is a fixed shared constant (128) and not each scenario's own N — pegging
it to N would make the whole mechanism silently inert.

More admitted modes still means more vacuum "particles", so depletion still climbs
with N, gently. Measured at `N_ref=128`, `exponent=0.85`, g~3.5e-3,
`N_PARTICLES=1e4`:

| N | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|
| depletion | 5.10% | 6.83% | 8.96% | 11.03% ✗ | 13.00% ✗ |

So the <10% gate starts failing around N=224 at those parameters. That is TWA's
validity regime breaking down, not a bug. Levers, in order of preference: raise
`n_particles`, lower `cutoff_exponent`, raise `cutoff_n_ref`.

### Finite temperature

The sampler uses **bare plane-wave** energies, not the trap's Bogoliubov
quasiparticle spectrum. That is fine for modes at or above the healing-length
cutoff the mask already imposes (`E_k >~ mu`), but it over-populates the near-IR
sector once `k_B*T` approaches `mu`: the continuum Rayleigh-Jeans tail
`n_k ~ T/E_k` has no analogue of the trap's real level spacing to regularise it.

Measured at N=48, L=12 with the thermal scenario's g and atom number:

| T/Tc | 0 | 2% | 5% | 10% | 15% |
|---|---|---|---|---|---|
| depletion | 3.27% | 3.66% | 5.02% | 8.16% | 11.63% ✗ |

Smooth and monotonic, not explosive, as long as T/Tc stays in the few-percent
range where `T* < mu*`. T/Tc of 30%+ — where a casually typical BEC experiment
sits — blows the method up (22%+ depletion at N=48 alone). That regime needs a
Bogoliubov-quasiparticle variant of `mode_variance()`, not a bigger cutoff.

This is also why `tw_trapped_thermal` keeps the weak 4.57 Hz reference trap rather
than a "standard" 100–300 Hz lab trap: a stronger `g` pushes `E_floor = 2*g*n_peak`
up until it swallows a huge swath of thermally saturated low-k modes at any
nonzero T (30–40% depletion even at modest T/Tc, checked directly at 200 Hz). The
weak trap keeps `mu*/Tc* ~ 0.1`, leaving real room for a genuinely low-T regime.

## Integrator and dt

`SPLITSTEP_ORDER = 2` is Strang, `O(dt^2)`. `4` is Yoshida's composition of three
Strang sub-steps, `O(dt^4)` at 3x the FFT/kick work per outer step — so it pays
only if it affords more than 3x the dt. The middle sub-step has a *negative* tau,
which is unavoidable above 2nd order and harmless: `exp(-1j*tau*(...))` is unitary
either way. `tests/test_existence.py::test_yoshida4_convergence_order` checks the
convergence *rate*, which is what a coefficient or sign error would break while
still looking numerically plausible.

`choose_dt()` sizes dt so the phase accumulated per step stays within ~0.2 rad for
both the fastest Fourier mode and the interaction term at peak density. This is an
accuracy heuristic, not a stability condition — split-step is norm-preserving for
any dt.

**The order=4 multiplier needs care.** Two separate numbers, deliberately:

- `physics._ORDER4_DT_MULTIPLIER = 400`, calibrated on the smooth noiseless
  collision scenario. Swept on real hardware: N=128 passed up to 24, N=256 up to
  248, N=512 passed at 400 and FAILED at 1000 (3.3% energy drift — the gate caught
  it as designed). 400 is the largest tested value that passed, with no deliberate
  headroom below it.
- `tw_solver._TW_ORDER4_DT_MULTIPLIER_DEFAULT = 50`, because that calibration does
  not transfer. Vacuum noise puts real amplitude on modes right against the
  cutoff — exactly the high-k content a large dt resolves worst. At N=32,
  order=4, the global 400 fails the energy gate with 18–51% drift,
  *non-monotonically* in the multiplier, while ≤25 passes with large margin.

And a caution that cost a production run: `tests/sweep_order4_dt.py` with its short
`T_SWEEP=1.0` window is **not** sufficient certification. A multiplier of 400
passed that sweep cleanly for `tw_trapped_thermal` and then blew up on a full
`T_TOTAL=5` run — 24,394% energy drift, a genuine instability the short window
never reached. Binary-search a multiplier with full-length runs only.

## Vortex nucleation

`gpe3d/` evolves the field exactly as before; `nucleation/` only reads `psi`.

### How detection works

A vortex is a line about which the phase winds by 2*pi. For each grid plaquette,
sum the four wrapped phase differences `arg(psi_q * conj(psi_p))` — always the
product form, never `arg(psi_q) - arg(psi_p)`, which needs branch handling and
silently produces garbage. The result is an integer for a resolved core, which is
a free resolution check: `nucleation/detect.py` raises if it is not.

All three plane families are swept. A line is found by the planes it *pierces*, so
z-planes alone would miss a ring lying in the y-z plane entirely —
`tests/test_detect.py` pins exactly that failure.

Pierce points from all three families are merged into one 3D cloud, grouped into
connected components (= vortex lines), and each line is traced, measured and
classified as open or closed. Both `N_vortex(t)` and total line length `L(t)` are
reported, but **do the statistics on `L(t)`**: integer counts are small and jumpy,
line length varies smoothly and is the standard quantum-turbulence observable.

One non-obvious trap in the masking: a resolved core has `|psi|^2 -> 0` at its
centre, so a raw density mask deletes exactly the plaquettes worth counting. The
mask is morphologically *closed* (`NUCLEATION_MASK_CLOSE_PASSES`, in cells) to fill
those holes without moving the cloud boundary outward. In the other direction,
masking is not optional: outside the condensate the phase is pure noise, and in a
TW run the entire box carries vacuum noise, so an unmasked count returns thousands
of spurious vortices.

Detection runs **per trajectory**, before any averaging. Vortices nucleate in
different places in every trajectory, so the ensemble-mean density is a smooth blob
with no cores in it at all. This is the failure mode that silently produces a null
result that looks like physics.

### Outputs

Everything lands in `outputs/<scenario>/`:

| file | what |
|---|---|
| `density_winding.gif` | the close-up. One column per plane: `\|psi\|^2` on top, phase winding below, cores ringed and sign-coloured with a live count per panel. One trajectory, never an ensemble average. |
| `vortex_map.gif` | the overview. Every detected core in the whole box, viewed down each axis, plus core count against time. What stays readable once a tangle has hundreds of lines in it. |
| `observables.csv` | one row per (trajectory, frame): counts, line lengths, integrality error |
| `pierce_points.npz` | every core's `(x, y, z, charge, plane)`, ~kB/frame |
| `run_metadata.json` | every knob that could change the answer, next to the answer |

Both movies play at 1x (playback length = `T_TOTAL`) at `NUCLEATION_GIF_FPS`. The
detector cannot run 60 times a second — it is three full-grid winding sweeps per
trajectory per frame — so recorded frames are cross-faded up to the target rate at
render time. Every frame carrying real data is a recorded one. The winding panels
are *not* blended: a half-integer winding is meaningless, so the nearest recorded
frame is held.

`vortex_map.gif` needs only `pierce_points.npz`, so any finished run re-renders
afterwards — different frame rate, different views, no GPU:

```bash
python -m nucleation.vortex_map outputs/nucleation_collision/pierce_points.npz \
       better.gif --half-box 8 --fps 60 --views xz --bins 192
```

That is why points are stored instead of fields: at N=384 one complex field is
~450MB, and a whole run's cores are a few MB.

Compare two finished runs without re-running anything:

```python
from nucleation.runner import compare_runs
print(compare_runs("outputs/nucleation_collision/observables.csv",
                   "outputs/nucleation_control/observables.csv"))
```

### What a count needs before it means anything

Three knobs can manufacture vortices. A count that is not stable under all three
is not a result:

1. **The noise cutoff** — raise it far enough and vortices always appear, out of
   unphysically large noise. Watch the `depletion_fraction` `run.py` prints;
   percent-level is required, order-1 is outside TWA.
2. **`NUCLEATION_MASK_THRESHOLD`** — sweep it.
   `nucleation/runner.sweep_mask_threshold()` does this on one stored frame, so it
   costs no GPU time and there is no excuse for skipping it.
3. **Grid `N`** — cores need `dx` well below the healing length; `dx <~ xi/3`.
   `run.py` prints `dx/xi` before evolving.

Plus the protocol itself: **20–50 independent seeds**, not the 4 the configs ship
with (fine for conservation gates, useless for nucleation statistics). Counts are
small integers and roughly Poisson, so quote bootstrap intervals, not ±SEM from a
handful of trajectories. Record onset `t_nuc` with a sustain requirement — with
sustain=1 the onset histogram is dominated by the earliest false positive in each
trajectory. And run the mean-field control, which must give zero.

### If nothing nucleates

Two clouds closing at ±v_half interfere with fringe spacing `pi/v_half`; for the
resulting grey solitons to be snake-unstable (and decay into vortex rings) that
spacing wants to be a few healing lengths. Scan `v_rel_real` upward by factors of
2 — the resolution gate catches the point where the grid can no longer represent
the kick, so the scan fails loudly rather than aliasing silently. And check
`T_TOTAL`: the instability needs several `xi/c_s` *after* the clouds overlap, so a
run that stops at overlap shows fringes and no vortices, which reads as a null
result but is just an early stop.

An offset collision (`separation=(6.0, 1.5, 0.0)`) shears the clouds past each
other and nucleates lines rather than rings, more robustly — but the offset breaks
the symmetry by itself, so the mean-field control nucleates too and the clean A/B
is lost. A first "does anything happen" shot, not a headline result.

## Tests

No GPU needed; the whole suite runs on CPU in a couple of minutes.

```bash
python -m pytest tests/ -q          # or run any file directly
python tests/test_winding.py        # imprinted vortex -> ±1; soliton/sound/noise -> 0
python tests/test_detect.py         # ring -> one closed line of the right circumference
python tests/test_observer.py       # detection while the GPE actually propagates
python tests/test_statistics.py     # bootstrap CIs, sustained-onset rule, control check
python tests/test_existence.py      # gates + analytic QHO ground state + Yoshida order
python tests/smoke_nucleation.py    # both nucleation scenarios end to end at N=32
```

Most of the risk in the vortex work lives in `test_winding.py` and
`test_detect.py`, and neither needs GPU time — run them before any long run.
`nucleation/synthetic.py` builds the imprinted fields they check against: straight
vortices, rings, and the nulls that matter (a grey soliton looks exactly like a
vortex in a density slice, and a count that cannot tell them apart is worthless).

`tests/` also holds instruments rather than tests: `sweep_order4_dt.py`,
`bench_perf.py`, `bench_orders_collision.py`, `probe_thermal_depletion.py`,
`tw_collision_noise_diff.py`.

## Performance and memory

Written for a 6 GB GTX 1660 Ti. Per grid point, TW scenario at order=4:

- persistent: `K_sq` 4 B + `V` 4 B + `mask` 1 B + `psi_mean` 8 B ≈ 17 B
- one resident trajectory: `psi` 8 B
- `compute_diagnostics()` transient: ~20–24 B

**Measured on real hardware: ~45.5 B/point.** `tw_collision` at N=512, order=4,
`batch_size=1`, 8 trajectories, T_TOTAL=5 peaked at 5821 MiB of 6144 — it works,
but at ~95% utilisation, so treat N=512 TW as "works for this config on this card"
rather than "safe with margin". Projections at that coefficient: N=384 ≈ 2460 MiB,
N=448 ≈ 3900 MiB, both with real headroom.

That 45.5 B/point matches the *logical* estimate almost exactly, meaning the ~1.9x
CuPy pool-overhead multiplier this project used to assume no longer applies. It was
measured under the old allocation pattern (order=4 kinetic operators persisted for
the whole run, diagnostics holding all three derivative arrays at once,
`batch_size=2`) — lots of large arrays alive concurrently, which is what drives
pool fragmentation. What removed it:

- `batch_size=1`, so only one trajectory's working set is ever resident (chunking
  is exact, not an approximation — it only trades memory for wall-clock).
- order=4's kinetic operators built per `run_block()` call instead of persisted,
  so 16 B/point is not resident during the diagnostics peak between blocks.
- `compute_diagnostics()` streamed one derivative axis at a time, down from a
  40 B/point transient. This is the function with the OOM history and it sits on
  every gate's critical path, so it has a dedicated regression test
  (`tests/test_diagnostics_memory_opt.py`) that reimplements the old formula and
  checks the new one against it — stronger than a gate test, which would pass a
  subtly wrong rewrite that still conserved energy to 1%.
- `ensemble.run_tw_ensemble(diagnostics_every=N)` recomputes diagnostics every N
  blocks instead of every block, keeping chunk endpoints fresh. Available, not
  wired into `config.py` — pick a cadence if you want it.

CPU-only scaling, measured the same way (peak RSS, real production pipeline):
~113 B/point at N=48 falling to ~98 B/point by N=128, i.e. roughly 2x the GPU
figure and flattening with N.

Prefer grid sizes whose prime factors are all small (2, 3, 5, 7, 11, 13). FFT cost
is dominated by N's largest prime factor, not just its size; `run.py` warns (does
not abort) on a bad N.

Reference run, `tw_collision` at N=384 (56.6M points), L=16, order=4, GTX 1660 Ti:
ground state 68 s; 1246 real-time steps in 866 s (1.4 steps/s); gates passed at
0.34% norm and energy drift; depletion 7.80%.

## Known flags

- **Trap anisotropy geometry.** The source paper has z as the tight axis
  (`omega_z = omega_x*sqrt(8)`) and collides along the wide x; `config.ANISOTROPY`
  currently encodes the opposite. Not silently flipped — every collision row's
  L/N/velocity gate was tuned against the current geometry, so fixing it needs a
  full re-tune and re-gate, not a one-line swap.
- **`tw_collision`'s `n_clouds=4.0`** is inconsistent with its own
  `n_per_cloud` (there are two clouds), and `depletion_fraction` is measured
  against that number. The nucleation rows use 2.0.
- **`v_split`.** The collision rows put each cloud at `v_rel/4`, the nucleation
  rows at `v_rel/2`. `/2` is the correct "each cloud carries half the relative
  velocity" convention; the `/4` rows are kept as-is because their velocity
  resolution gate was tuned against them. Settle which you mean before quoting a
  collision velocity in a report.
- **`tw_trapped_dipole` at N=384** is past its validated range — depletion is
  already 13% at N=256, over the gate. See the depletion table above.
