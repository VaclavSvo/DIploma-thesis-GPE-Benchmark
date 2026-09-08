# The physics and the numerics

Everything this code computes, where it comes from, and which paper to read
when a number here looks wrong. `README.md` is how to *run* it; this is what
it *is*. Section numbers are referenced from the source files.

Notation: `hbar = m = 1` throughout unless a formula carries SI symbols.
`psi` is the classical (c-number) field, `n = |psi|^2` the density, `g` the
dimensionless coupling, `N` the grid points per axis, `L` the nominal box
edge, `dx` the grid spacing.

---

## 1. The model

### 1.1 Gross–Pitaevskii equation

A dilute Bose gas at `T << Tc`, with all `N` atoms in one mode, obeys

    i * dpsi/dt = ( -1/2 * grad^2 + V(r) + g * |psi|^2 ) * psi                (1)

with `psi` normalised to the atom number, `integral |psi|^2 dV = N_atoms`.
This is the mean-field limit of the many-body problem: the field operator is
replaced by a c-number, valid when the condensate is dilute (`n a^3 << 1`) and
depletion is small. Derivation and validity: **Dalfovo et al. (1999)** §II,
**Pethick & Smith (2008)** ch. 6, **Pitaevskii & Stringari (2016)** ch. 5.

Implemented in `gpe3d/physics.py`. The trap is
`V = 1/2 (wx^2 x^2 + wy^2 y^2 + wz^2 z^2)` (`gpe3d/potentials.py`); the
collision scenarios switch to `V = 0` after preparation.

### 1.2 The energy functional and what is conserved

    E[psi] = integral [ 1/2 |grad psi|^2  +  V |psi|^2  +  g/2 |psi|^4 ] dV   (2)

and equation (1) is `i dpsi/dt = dE/dpsi*`. Because (1) is a Hamiltonian flow
of (2) with a time-independent `V`, the following are **exactly** conserved and
are what `gpe3d/gates.py` checks:

| quantity | expression | conserved when |
|---|---|---|
| norm | `integral n dV` | always (unitary evolution) |
| energy | equation (2) | `V` time-independent |
| momentum | `P = integral Im(psi* grad psi) dV` | `V` translation-invariant (i.e. `V=0`) |
| angular momentum | `L = integral r x j dV`, `j = Im(psi* grad psi)` | `V` rotation-invariant |

Note the `g/2` in the energy against the `g` in the equation of motion: the
interaction term is quadratic in the density, so the functional derivative
doubles it. A `g` in equation (2) is a common and silent error — it shows up
only as an energy that fails to conserve at large `g`.

**Momentum in a trap is not conserved, and that is correct physics.** A
displaced or kicked cloud oscillates (Ehrenfest's theorem; for the dipole mode
the frequency is exactly the bare trap frequency regardless of `g`, by the
generalised **Kohn theorem** — *Kohn (1961)*, *Dobson (1994)*). `run.py`
therefore disables the momentum gates for displaced, kicked and TW runs, and
`tests/test_existence.py` uses the Kohn mode as an analytic check.

### 1.3 Derived scales

| scale | formula | where used |
|---|---|---|
| chemical potential (TF) | `mu = g * n_peak` | the noise cutoff floor |
| healing length | `xi = 1/sqrt(2 g n)` | `nucleation/runner.healing_length` |
| speed of sound | `c = sqrt(g n)` | timescale of the snake instability |
| harmonic length | `sigma = 1/sqrt(2 w)` | analytic ground-state width |
| QHO ground energy | `E0 = (wx+wy+wz)/2` | `tests/test_existence.py` |

**One healing-length convention, project-wide.** `xi = 1/sqrt(2 g n)` follows
from balancing quantum pressure against interaction energy,
`hbar^2/(2 m xi^2) = g n` (**Pethick & Smith** §6.2, **Pitaevskii & Stringari**
§5.2). Equivalently `E_k = g*n` at `k = 1/xi`, which is exactly the energy
scale `gpe3d/noise.py` measures its UV cutoff against — so the two modules now
agree by construction. The other convention in circulation, `1/sqrt(g n)`, is
larger by `sqrt(2)`; `nucleation/runner.py` used to use it while `noise.py`
used this one, which made the printed `dx/xi` optimistic by that factor (see
`AUDIT.md` §B3).

---

## 2. Units (`gpe3d/units.py`)

A real experiment is mapped to `hbar = m = 1` by choosing the harmonic
oscillator length of a *reference* trap frequency as the length unit:

    l_phys = sqrt( hbar / (m * w_ref) )        t_phys = 1 / w_ref
    g      = 4*pi*a / l_phys                   v_nat  = v_SI / (l_phys * w_ref)
    T*     = k_B T / (hbar * w_ref)

For `23-Na` (`a = 2.75 nm`, `m = 22.9897693 u`) at `w_ref = 2*pi*4.57 Hz`:

    l_phys = 9.8084 um     t_phys = 34.826 ms     g = 3.5233e-3

### 2.1 Cross-check against the source paper (this validates `a` and `g`)

**Norrie, Ballagh & Gardiner, PRA 73, 043617 (2006)** use a *different* length
unit, `x0 = sqrt(hbar / (2 m w_x)) = l_phys / sqrt(2)`, and quote a
dimensionless coupling

    U0~ = 8*pi*a*sqrt(2 m w_x / hbar) = 8*pi*a / x0 = 2*sqrt(2) * g

Putting this project's numbers in: `2*sqrt(2) * 3.5233e-3 = 9.9653e-3`, against
the paper's stated `U0~ = 1e-2` — agreement to 0.35%. That closes the
"scattering length: best effort, verify" flag that used to sit in `units.py`:
`a = 2.75 nm` is right (independently the value quoted for `23-Na` in the
sodium-lattice literature), `w_ref = 4.57 Hz` is the paper's `w_x`, and the
`g = 4*pi*a/l_phys` derivation is right.

**Do not compare `config.G` against the paper's `U0~` directly** — they differ
by `2*sqrt(2)` purely from the length-unit convention.

### 2.2 Critical temperature

`condensate_critical_temperature_natural()` implements the ideal-gas result for
a 3D harmonic trap,

    k_B Tc = hbar * w_bar * (N / zeta(3))^(1/3),   w_bar = (wx wy wz)^(1/3)

(**Dalfovo et al. (1999)** Eq. 2). The interaction shift is a further few-percent
correction, deliberately not applied: this is a reference scale for picking a
"low but nonzero" `T/Tc`, not a prediction.

---

## 3. The grid, and the one failure mode gates cannot see

### 3.1 Grid convention

`GPEPhysics3D._build_grid()` builds `x = linspace(-L/2, +L/2, N)` — **both
endpoints included**. Consequences, all of them deliberate and all of them
worth knowing:

* `dx = L/(N-1)`, not `L/N`. Use `gpe3d.physics.grid_spacing(N, L)`; never
  recompute it as `L/N` (that bug is `AUDIT.md` §B2).
* The FFT's periodic box is `N*dx = L*N/(N-1)`, a fraction `1/(N-1)` larger
  than the nominal `L` (0.4% at `N=250`). `dV = dx^3` and `V = N^3 dV` are
  exact by construction, so nothing downstream is inconsistent — but "the box
  is L" is true only to `1/N`.
* The grid is symmetric under `r -> -r`, which is convenient for a symmetric
  trap, and for even `N` there is no sample exactly at the origin. That is why
  `nucleation/observer.plane_index()` exists instead of `N//2`.
* `dx` is computed from the exact `L/(N-1)` in float64, not from a float32
  `x[1]-x[0]` (which was ~4e-6 relative off, and fed `dV`, every norm and
  energy, and the `k` grid — `AUDIT.md` §B7).

Array axis order is `(z, y, x)` == `(-3, -2, -1)` everywhere, so a plain
`(N,N,N)` field and a batched `(B,N,N,N)` TW ensemble go through identical
code. `X`, `Y`, `Z`, `KX`, `KY`, `KZ` are broadcastable `(1,1,N)`-style views,
not dense arrays; only `K_sq` is dense.

### 3.2 The k grid and aliasing

`k = 2*pi*fftfreq(N, dx)`, so `k_nyquist = pi/dx`. A velocity kick is a plane
wave `exp(i v.r)`, and with `hbar = m = 1` a velocity *is* a wavenumber. If
`|v| > pi/dx` the grid represents it as a different, aliased wavenumber.

**Aliasing is exactly unitary.** Norm and energy stay green while the dynamics
are silently wrong — the first attempt at the collision scenario passed every
gate with the clouds barely moving, aliased to nearly zero velocity. This is
the one failure mode no conservation gate can catch, which is why
`gates.resolution_gate()` runs as a pre-flight check in `run.py` before
anything expensive is built, and aborts.

Nonlinear mixing generates harmonics of the kick, so the gate demands a factor
`safety_factor` (default 2) of headroom, not bare representability.

---

## 4. Time integration (`gpe3d/physics.py`)

### 4.1 Split-step Fourier

Write (1) as `i dpsi/dt = (T + W)psi` with `T = -1/2 grad^2` diagonal in `k`
and `W = V + g|psi|^2` diagonal in `r`. Neither `exp(-i dt T)` nor
`exp(-i dt W)` is hard on its own; the composition is where the order lives.

**Strang splitting** (*Strang 1968*; for NLSE *Weideman & Herbst 1986*):

    S2(dt) = K(dt/2) N(dt) K(dt/2),      local error O(dt^3), global O(dt^2)

`K(tau) = exp(-i tau T)` applied in `k` space, `N(tau) = exp(-i tau W)` applied
in `r` space. Both factors are pure phases, so **the scheme is norm-preserving
for any `dt`** — there is no stability limit, only an accuracy one.

**The kinetic factor is the classic trap.** `T = -1/2 grad^2` has Fourier
eigenvalue `K_sq/2`, so the *half*-step propagator is

    K(dt/2) = exp(-i * 0.25 * dt * K_sq)

A coefficient of `0.5` there silently implements `T = -grad^2`, i.e. half the
intended mass. It was in the 2D reference notebook this project was ported
from, despite that notebook's own text stating `-1/2 grad^2`. Verified
numerically here: at `0.5` a free-particle Fourier mode matches
`exp(-i k^2 dt)` to 1e-16, not `exp(-i k^2 dt/2)`. With `0.25` the
noninteracting ground state matches the analytic QHO result
(`E0 = 1.5 w`, `sigma = 1/sqrt(2w)`) to four significant figures.

Consecutive half-kicks are merged across sub-steps, so a block of `n` steps is
`K(dt/2) [N(dt) K(dt)]^(n-1) N(dt) K(dt/2)`: one FFT pair per step, not two.

### 4.2 Yoshida 4th order

**Yoshida (1990)** (see also *Suzuki 1990*) composes three Strang steps:

    S4(dt) = S2(th*dt) S2((1-2*th)*dt) S2(th*dt),   th = 1/(2 - 2^(1/3))

`th ~ 1.3512`, so the middle sub-step has `tau ~ -1.7024 * dt` — **negative**.
That is unavoidable for any symmetric composition above 2nd order (*Suzuki's
theorem*) and harmless here: `exp(-i tau H)` is unitary for either sign. It
would be fatal for a diffusive (imaginary-time) step, which is why imaginary
time stays at order 2.

Merging the adjacent kinetic factors gives what the code actually runs:

    S4 = K(th*dt/2) N(th*dt) K((1-th)*dt/2) N((1-2th)*dt) K((1-th)*dt/2) N(th*dt) K(th*dt/2)

and the outer `K(th*dt/2)` merges with its neighbour into `K(th*dt)` between
consecutive outer steps. Cost: 3x a Strang step. It pays only if it affords
more than 3x the `dt`.

`tests/test_existence.py::test_yoshida4_convergence_order` checks the
convergence *rate*, which is what a wrong coefficient or sign would break while
still looking numerically plausible.

### 4.3 Choosing dt

`choose_dt()` keeps the phase accumulated per step below `frac ~ 0.2` rad for
both the fastest Fourier mode and the interaction term at peak density:

    dt = frac / max( K_sq_max / 2 , |g| * n_peak )

This is an **accuracy** heuristic, not a stability condition. At order 4 it is
multiplied by a calibrated factor — see `README.md` "Integrator and dt" for the
measured values and the warning that short sweeps do not certify them.

### 4.4 Imaginary time

`t -> -i tau` turns (1) into a normalised gradient flow whose fixed point is
the ground state (**Chiofalo, Succi & Tosi (2000)**; **Bao & Du (2004)**):
each step is the same split-step composition with real exponentials, followed
by renormalisation to `n_particles` (imaginary time is dissipative, so norm is
not conserved). It is unconditionally stable; `dt` only sets convergence rate.
Convergence is declared when the energy stops changing by more than `tol`
relative.

---

## 5. Diagnostics (`gpe3d/physics.py`)

Two paths, because the two callers want different things.

**Energy only** — `_kinetic_energy()`. By Parseval, with the unnormalised
forward transform,

    E_kin = (dV / N^3) * sum_k (K^2/2) |psi_k|^2

and `K^2 = kx^2 + ky^2 + kz^2` separates, so `|psi_k|^2` is reduced onto three
length-`N` marginals and contracted with the `k` axes. **One forward FFT, no
inverse, no derivative array.** Agrees with the real-space form to 2e-7
relative (float32 round-off).

**The full report** — `_diagnostics_from_currents()`. Angular momentum weights
the current by *position*, so it genuinely needs `j(r)` in real space: one
shared forward FFT and one inverse per axis, four transforms instead of six.

Each `L` component is formed as **one elementwise difference reduced once**,
e.g. `sum(X*jy - Y*jx)`, never as a difference of two separate reductions. `L`
is strongly cancelling — on a generic field `|Lz| ~ 0.4` against per-term sums
of order `1e3`, i.e. 3.2 digits of cancellation out of float32's ~7 — so
splitting it costs about four significant digits. `tests/test_diagnostics_memory_opt.py`
pins this against a byte-for-byte reimplementation of the pre-optimisation
formula, and compares momentum and angular momentum against the *scale of their
own triple* rather than against themselves, for the same reason `gates.py`
does.

---

## 6. Truncated Wigner (`gpe3d/noise.py`, `gpe3d/tw_solver.py`)

### 6.1 Why the propagator is untouched

In the Wigner phase-space representation, the exact evolution of the Bose field
is a Fokker–Planck-like equation whose third-order derivative terms are
dropped in the **truncated Wigner approximation** (TWA). What survives is
*exactly* the classical GPE — with no noise term — evolving an *ensemble of
initial conditions* sampled from the Wigner function of the initial state.
So every trajectory here obeys the same deterministic equation (1), and all
of the quantum mechanics is in the initial draw.

Canonical sources: **Steel et al., PRA 58, 4824 (1998)** (the original
prescription for trapped BECs); **Sinatra, Lobo & Castin, J. Phys. B 35, 3599
(2002)** (limits of validity, the half-quantum-per-mode rule, and the cutoff
discussion this code follows); **Blakie et al., Adv. Phys. 57, 363 (2008)**
(the c-field review, and how TWA relates to PGPE/SGPE);
**Polkovnikov, Ann. Phys. 325, 1790 (2010)** (the phase-space derivation in
general).

### 6.2 The sample

Each plane-wave mode below an energy cutoff gets an independent complex
Gaussian amplitude `alpha_k` with

    <|alpha_k|^2> = n_k(T) + 1/2 = (1/2) coth( E_k / 2T ),     E_k = K_sq/2

`T = 0` gives the flat vacuum `1/2` exactly (half a quantum per mode).
`k = 0` is kept at the flat `1/2` at any `T`: it is the condensate's own mode,
already normalised into the mean field, and `n_k(T)` diverges as `E_k -> 0`.
`E_k` is the **bare** single-particle energy, not a Bogoliubov quasiparticle
energy — consistent only because the cutoff admits modes at or above `mu`,
where the two coincide (see §6.6 for what breaks when they do not).

Real space (`sample_noise`): on the orthonormal plane-wave basis
`delta_psi(r) = sum_k alpha_k V^(-1/2) exp(i k.r)`, which for FFT frequencies
is `(N^3 / sqrt(V)) * ifftn(alpha)` and yields
`<|delta_psi(r)|^2> = sum_k <|alpha_k|^2> / V` at every grid point. `V = N^3 dV`
is used, not `L^3` — see §3.1.

### 6.3 The ordering correction — the classic TWA bug

`<|psi_W|^2>` is **symmetrically (Wigner) ordered** and is therefore
contaminated by the noise's own particle content. The physical (normally
ordered) density is

    n_phys(r) = <|psi_W(r)|^2>_ensemble  -  offset_density
    offset_density = sum_k <|alpha_k|^2> / V                                 (3)

At `T=0` this is `N_cutoff_modes / (2V)`. It is **not** the flat `1/(2 dV)`
often quoted — that holds only when the cutoff spans every grid mode. For a
genuine partial cutoff the flat value over-subtracts by
`N^3 / N_cutoff_modes`, which makes a healthy run look like it is losing
particles it never had.

Implemented in `noise.offset_density()`; applied in
`tw_solver.physical_particle_number()`, in the ensemble's density slices, and
in `nucleation/detect.detect_frame()` (where an unsubtracted uniform noise
floor would drag `n_peak` around and move the mask threshold).

### 6.4 The UV cutoff is the physics, not a convenience

Without a cutoff the total noise energy grows with the number of grid modes:
the answer would depend on the grid, and refining it would change the physics.
So the cutoff is tied to the **interaction energy scale**,

    E_cut = cutoff_multiplier * g * n_peak       (i.e. k_cut ~ 1/xi)

and the grid must then be fine enough to resolve everything below it — not the
other way round. `cutoff_multiplier` is the one free literature parameter here
(papers use O(1–few)); it is meant to be **swept**, not trusted.

`scale_cutoff_multiplier_to_grid()` lets a finer grid admit more modes:

    E_cut = max( E_floor, min( E_floor * (N/N_ref)^p , E_nyquist / safety^2 ) )

The growth law is a small *power* of `N/N_ref` on purpose. Mode count scales as
`E_cut^1.5` in 3D, so parametrising the cutoff as any sizeable fraction of
`E_nyquist` explodes — measured at `g~3.5e-3, N=96, L=8`, `E_nyquist` is ~260x
`E_floor` and filling to it admitted 34% of all grid modes, for a 1503%
depletion fraction. `N == N_ref` reproduces the fixed cutoff exactly.

### 6.5 Validity: the depletion fraction

    depletion_fraction = n_vacuum_offset / n_particles_target

must be **percent-level**. Order-1 means the sampled noise carries as many
"particles" as the condensate and the run is outside TWA's regime whatever
norm and energy say. `run.py` prints it and fails the run above 10%. This is
not academic: at the dimensionless `g=1, N_PARTICLES=1` test convention every
conservation gate passed with 950% depletion — that convention was only ever
meant to validate the noiseless propagator.

Measured scaling with resolution (`N_ref=128`, `p=0.85`, `g~3.5e-3`, `N=1e4`):

| N | 128 | 160 | 192 | 224 | 256 |
|---|---|---|---|---|---|
| depletion | 5.10% | 6.83% | 8.96% | 11.03% ✗ | 13.00% ✗ |

More admitted modes means more vacuum "particles", so depletion climbs with
`N`. That is TWA's validity boundary, not a bug. Levers in order of
preference: raise `n_particles`, lower `cutoff_exponent`, raise `cutoff_n_ref`.

### 6.6 Finite temperature, and its known limitation

The sampler uses **bare plane-wave** energies, not the trap's Bogoliubov
spectrum. That is fine at or above the healing-length cutoff (`E_k >~ mu`),
but it over-populates the near-IR once `k_B T` approaches `mu`: the continuum
Rayleigh–Jeans tail `n_k ~ T/E_k` has no analogue of the trap's real level
spacing to regularise it.

Measured at `N=48, L=12`:

| T/Tc | 0 | 2% | 5% | 10% | 15% |
|---|---|---|---|---|---|
| depletion | 3.27% | 3.66% | 5.02% | 8.16% | 11.63% ✗ |

Smooth and monotonic while `T* < mu*`. `T/Tc` of 30%+ — where a typical BEC
experiment sits — blows the method up (22%+ depletion at `N=48` alone). That
regime needs a Bogoliubov-quasiparticle `mode_variance()`, not a bigger
cutoff. This is the single largest open physics limitation in the project.

It is also why `tw_trapped_thermal` keeps the weak 4.57 Hz trap: a stronger
`g` pushes `E_floor = 2 g n_peak` up until it swallows a swath of thermally
saturated low-`k` modes at any nonzero `T` (30–40% depletion even at modest
`T/Tc`, checked at 200 Hz). The weak trap keeps `mu*/Tc* ~ 0.1`.

### 6.7 Chunking is exact

`gpe3d/ensemble.py` evolves `batch_size`-sized chunks and folds each into a
running weighted mean. Every quantity combined is a per-trajectory mean within
its chunk, and the overall mean is the chunk-size-weighted average of those —
a sum over a partition equals the sum of partial sums. Trading memory for
wall-clock, with no approximation. Pinned by
`tests/test_ensemble_memory_opts.py`.

---

## 7. Vortices (`nucleation/`)

Nothing in `nucleation/` changes the physics — these modules only read `psi`.

### 7.1 What a vortex is, numerically

Superfluid circulation is quantised: around any loop enclosing a vortex core
the phase winds by `2*pi*q`, `q` integer (**Fetter, RMP 81, 647 (2009)**;
**Barenghi, Skrbek & Sreenivasan, PNAS 111, 4647 (2014)** for the turbulence
context). The core is a real density zero of size `~xi`.

For each grid plaquette, sum the four wrapped phase differences:

    w = [ d(a->b) + d(b->c) + d(c->d) + d(d->a) ] / 2*pi
    d(p->q) = arg( psi_q * conj(psi_p) )

**Always the product form**, never `arg(psi_q) - arg(psi_p)`: the product is
already wrapped to `(-pi, pi]` by construction; the difference is not, and
needs branch handling that silently goes wrong. The four differences are taken
counter-clockwise in the `(u,v)` plane with `(u, v, normal)` right-handed, so
the sign of `w` is the sign of the charge along `+normal`. This is the standard
plaquette method — see e.g. **Foster, Blakie & Davis, PRA 81, 023623 (2010)**
for its use in c-field simulations, and **Villois et al., J. Phys. A 49,
415502 (2016)** for the filament-tracking family it belongs to.

**Integrality is a free resolution check.** `w` is an integer for a resolved
core; `max |w - round(w)|` is exposed as `integrality_error()` and
`detect.pierce_points()` *raises* above `integrality_tol`. A non-integer
winding means `dx` is too coarse for `xi` and the "charges" being counted are
not integers at all.

### 7.2 All three plane families

A line is found by the planes it **pierces**, so a family of `z`-planes finds
lines running along `z` and misses a ring lying in the `y-z` plane entirely.
All three families are swept; `tests/test_detect.py` pins exactly that failure.
Because each family needs the two axes transverse to its normal, the three
per-axis phase differences are computed **once** and shared
(`winding.phase_differences()`) rather than six times.

### 7.3 Masking — where the false positives live

Outside the condensate the phase is numerical noise, and in a TW run the
*entire box* carries vacuum noise; an unmasked count returns thousands of
spurious vortices. So a plaquette counts only if all four corners have
`|psi|^2 > threshold * n_peak`.

The **opposite** error is easy to miss: a resolved core has `|psi|^2 -> 0` at
its centre, so it punches a hole in a raw mask and deletes exactly the
plaquettes worth counting. The mask is therefore morphologically **closed**
(dilate then erode, `mask_close_passes` cells) — which fills those holes
without moving the cloud boundary outward, as a plain dilation would.
`close_passes` must comfortably exceed the core radius in cells, `~xi/dx`.

Optionally (`require_density_dip`, on by default) a plaquette must also touch a
cell whose density is below `dip_factor` times its local 3x3x3 mean. Winding
**and** a density dip is much stronger than either alone, and is what separates
a real core from a phase feature riding on a sound pulse. The dip uses
`plaquette_any` (the core sits *inside* the plaquette, so only the nearest
corners are guaranteed minima), not `plaquette_mask`'s all-four rule.

`mask_threshold` is a **convergence parameter**, not a constant. Sweep it
(`runner.sweep_mask_threshold()`, free — it runs on one stored frame) and check
the count is stable, or the number says where the mask was drawn.

### 7.4 From pierce points to lines

Pierce points from all three families merge into one 3D cloud, are grouped into
connected components by a uniform spatial hash plus union-find (`link_cutoff_dx`
cells), and each component is measured:

* a minimum spanning tree (Prim, on a chunked dense distance matrix),
* the tree's **diameter path** by two greedy sweeps — on an open line this is
  the line; on a closed loop the spanning tree is the loop minus its longest
  edge, so the path is the loop minus one gap,
* which makes the endpoint-proximity test (`close_cutoff_dx`) a reliable
  ring detector; a closed line has that gap added back to its length.

`length_mst` is reported alongside `length`: it exceeds it when a component
branches, i.e. at a reconnection or when two separate lines pass within the
link cutoff.

**Independent cross-check.** A curve of length `Λ` crossing a family of planes
spaced `dx` apart makes `Λ<|t.e|>/dx` intersections; averaged over orientations
`<|t.e|> = 1/2`, so the three orthogonal families give
`n_x+n_y+n_z = 3Λ/(2 dx)`, i.e.

    Λ = 2 * dx * (n_x + n_y + n_z) / 3

(`detect.line_length_from_crossings`). It needs no linking at all, which makes
it the right check on whether `link_cutoff` is set sanely — the two should
agree to tens of percent.

### 7.5 Report `L(t)`, not `N(t)`

Integer counts are small and jumpy; across a handful of trajectories their
variance swamps the signal. Total vortex line length `L(t)` is continuous,
varies smoothly, and is the standard quantum-turbulence observable
(**Barenghi et al. 2014**). Both are written; do the statistics on `L(t)`.

### 7.6 Detection is per trajectory, before any averaging

Vortices nucleate in **different places in every trajectory**, so the
ensemble-mean density is a smooth blob with no cores in it at all. Averaging
first is the failure mode that silently produces a null result looking like
physics. The observer hooks into the live `psi` inside the evolution loop, and
both movies follow one trajectory.

### 7.7 Why a collision should nucleate anything

Two clouds closing at `+-v_half` interfere with fringe spacing `pi/v_half`. The
resulting grey solitons are **snake-unstable** when that spacing is a few
healing lengths, and decay into vortex rings — observed directly by
**Anderson et al., PRL 86, 2926 (2001)**; theory in **Muryshev et al., PRA 60,
R2665 (1999)** and **Brand & Reinhardt, PRA 65, 043612 (2002)**. The
condensate-collision route specifically is **Norrie, Ballagh & Gardiner,
PRL 94, 040401 (2005)** and **PRA 73, 043617 (2006)**.

The instability needs several `xi/c_s` **after** the clouds overlap. A run that
stops at overlap shows fringes and no vortices — which reads as a null result
but is just an early stop.

### 7.8 What a count needs before it means anything

Three knobs can manufacture vortices, and a count not stable under all three is
not a result:

1. **the noise cutoff** — raise it far enough and vortices always appear, out
   of unphysically large noise. Watch `depletion_fraction`.
2. **`mask_threshold`** — sweep it; it costs no GPU time.
3. **grid `N`** — cores need `dx <~ xi/3`; `run.py` prints `dx/xi`.

Plus the protocol: **20–50 independent seeds** (not the 8 the configs ship
with), bootstrap intervals rather than `+-SEM` on near-Poisson counts, an onset
time with a *sustain* requirement (with `sustain=1` the onset histogram is
dominated by the earliest false positive in each trajectory), and **the
mean-field control**, which must give zero. If the control is not clean, the
detector is firing on something that is not a vortex and every TW number is
suspect.

---

## 8. Bibliography

Annotated with what each source is actually used for here.

**Mean-field theory and BEC background**

1. F. Dalfovo, S. Giorgini, L. P. Pitaevskii, S. Stringari, *Theory of
   Bose-Einstein condensation in trapped gases*, **Rev. Mod. Phys. 71, 463
   (1999)**. — GPE derivation; the harmonic-trap `Tc` used in `units.py`;
   the `23-Na` scattering length.
2. C. J. Pethick, H. Smith, *Bose–Einstein Condensation in Dilute Gases*, 2nd
   ed., CUP (2008), ch. 6. — healing length `xi = 1/sqrt(2 g n)`, sound speed.
3. L. P. Pitaevskii, S. Stringari, *Bose-Einstein Condensation and
   Superfluidity*, 2nd ed., OUP (2016), ch. 5. — same, plus quantised
   circulation.
4. W. Kohn, **Phys. Rev. 123, 1242 (1961)**; J. F. Dobson, **PRL 73, 2244
   (1994)**. — the dipole (Kohn) mode oscillates at the bare trap frequency
   regardless of `g`: the analytic check in `tests/test_existence.py`.

**Numerical methods**

5. G. Strang, **SIAM J. Numer. Anal. 5, 506 (1968)**. — the 2nd-order splitting.
6. J. A. C. Weideman, B. M. Herbst, **SIAM J. Numer. Anal. 23, 485 (1986)**. —
   split-step Fourier for the nonlinear Schrödinger equation; convergence.
7. H. Yoshida, **Phys. Lett. A 150, 262 (1990)**; M. Suzuki, **Phys. Lett. A
   146, 319 (1990)**. — the 4th-order composition and the necessity of a
   negative sub-step.
8. W. Bao, D. Jaksch, P. A. Markowich, **J. Comput. Phys. 187, 318 (2003)**. —
   time-splitting spectral methods for the GPE specifically.
9. M. L. Chiofalo, S. Succi, M. P. Tosi, **Phys. Rev. E 62, 7438 (2000)**;
   W. Bao, Q. Du, **SIAM J. Sci. Comput. 25, 1674 (2004)**. — imaginary-time /
   normalised gradient flow for the ground state.

**Truncated Wigner and c-field methods**

10. M. J. Steel *et al.*, **Phys. Rev. A 58, 4824 (1998)**. — the original TWA
    prescription for trapped BECs; half a quantum per mode.
11. A. Sinatra, C. Lobo, Y. Castin, **J. Phys. B 35, 3599 (2002)**. — limits of
    validity, the mode-variance convention this code implements, the cutoff
    discussion, and the depletion criterion.
12. P. B. Blakie, A. S. Bradley, M. J. Davis, R. J. Ballagh, C. W. Gardiner,
    **Adv. Phys. 57, 363 (2008)**. — the c-field review: TWA vs PGPE vs SGPE,
    and how to choose a cutoff.
13. A. Polkovnikov, **Ann. Phys. 325, 1790 (2010)**. — phase-space
    representation and the systematics of truncation.
14. C. W. Gardiner, M. J. Davis, **J. Phys. B 36, 4731 (2003)**. — the
    stochastic GPE, i.e. what this project would need for a real thermal bath.

**Condensate collisions and vortex nucleation** *(the scenarios' source)*

15. A. A. Norrie, R. J. Ballagh, C. W. Gardiner, *Quantum turbulence in
    condensate collisions: an application of the classical field method*,
    **Phys. Rev. Lett. 94, 040401 (2005)**.
16. A. A. Norrie, R. J. Ballagh, C. W. Gardiner, *Quantum turbulence and
    correlations in Bose-Einstein condensate collisions*, **Phys. Rev. A 73,
    043617 (2006)**. — the calibration target: `23-Na`, `w_x = 2*pi*4.57 Hz`,
    `lambda_z/lambda_{x,y} = sqrt(8)` (z **tight**), 2e6 atoms total,
    wavepackets separating at 4.0 mm/s along x, `U0~ = 1e-2`. See §2.1 and
    `README.md` "Known flags".

**Vortices, detection and turbulence**

17. A. L. Fetter, **Rev. Mod. Phys. 81, 647 (2009)**. — vortices in trapped
    condensates.
18. C. F. Barenghi, L. Skrbek, K. R. Sreenivasan, **PNAS 111, 4647 (2014)**;
    A. C. White, B. P. Anderson, V. S. Bagnato, **PNAS 111, 4719 (2014)**. —
    quantum turbulence, and why line length is the observable.
19. N. G. Berloff, ... / S. Zuccher, M. Caliari, A. W. Baggaley, C. F.
    Barenghi, **Phys. Fluids 24, 125108 (2012)**. — vortex reconnections in
    the GPE; what `length_mst > length` is detecting.
20. A. Villois, G. Krstulovic, D. Proment, H. Salman, **J. Phys. A 49, 415502
    (2016)**. — vortex filament tracking in the GPE; the pierce-and-link
    family this detector belongs to.
21. S. P. Foster, P. B. Blakie, M. J. Davis, **Phys. Rev. A 81, 023623
    (2010)**. — plaquette phase-winding detection in c-field simulations.

**Solitons and the snake instability** *(why a collision nucleates)*

22. A. Muryshev, H. B. van Linden van den Heuvell, G. V. Shlyapnikov,
    **Phys. Rev. A 60, R2665 (1999)**.
23. B. P. Anderson *et al.*, *Watching dark solitons decay into vortex rings*,
    **Phys. Rev. Lett. 86, 2926 (2001)**.
24. J. Brand, W. P. Reinhardt, **Phys. Rev. A 65, 043612 (2002)**.

**Statistics**

25. B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman &
    Hall (1993). — the percentile bootstrap used in `nucleation/statistics.py`,
    and why `+-SEM` is wrong for small near-Poisson samples.
