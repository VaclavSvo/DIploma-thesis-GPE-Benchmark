"""Every physical and run parameter, for every scenario.

Pick SCENARIO, edit its row in SCENARIOS if you want different physics, then
run `python run.py`. No command-line flags.

The SI -> natural-units (hbar=m=1) derivation lives in gpe3d/units.py, so each
row states only its own species/trap/velocity/TW numbers. Everything below the
table is derivation and shared defaults -- read the table, not the machinery.

README.md has the physics behind the calibrated numbers here (cutoff scaling
and depletion, dt multipliers, what a nucleation count needs before it means
anything).
"""
from time import time

import numpy as np

from gpe3d.units import (derive_natural_units, velocity_to_natural,
                          condensate_critical_temperature_natural,
                          natural_temperature_to_kelvin,
                          NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M)

# ── Pick one ──────────────────────────────────────────────────────────────────
#   trapped_dipole       single condensate, offset/kicked, dimensionless g=1.
#                        Fast Kohn-mode sanity check of the propagator.
#   collision            two condensates released from a trap and collided in
#                        free space, scaled to Norrie/Ballagh/Gardiner,
#                        PRA 73, 043617 (2006).
#   tw_trapped_dipole    single trapped condensate + Truncated-Wigner vacuum
#                        noise, ensemble of N_TRAJECTORIES.
#   tw_collision         the collision geometry + TW vacuum noise.
#   tw_trapped_thermal   tw_trapped_dipole at finite (low) temperature.
#   nucleation_collision tw_collision tuned to actually nucleate vortices
#                        (faster collision, longer T_TOTAL) and instrumented to
#                        count them per trajectory, before any averaging.
#   nucleation_control   the same parameters with the noise off, through the
#                        same detector. Must give zero vortices -- if it does
#                        not, the detector is firing on something that is not a
#                        vortex and the TW numbers mean nothing. Run it
#                        alongside every nucleation_collision run.
#   halo_collision       the Na-23 collision of Norrie/Ballagh/Gardiner,
#                        PRA 73, 043617 (2006), calibrated number-for-number,
#                        so the s-wave scattering halo actually appears. See
#                        the _HALO block for every conversion.
#   halo_control         halo_collision with the noise off. The halo is a
#                        purely Truncated-Wigner effect -- the paper's opening
#                        claim is that the GPE does not predict it at all -- so
#                        this row must show two clean clouds and NO shell. It
#                        is the A/B that turns a picture into a result.
SCENARIO = "halo_collision"
# Everything below the table is derived from THIS row at import time, so a
# script that reassigns config.SCENARIO after importing config only moves the
# solver (run.py reads the row live) -- not OMEGA, N_PARTICLES, SOLVER_PARAMS
# or the NUCLEATION_* detector globals. tests/smoke_nucleation.py does exactly
# that, which is why it is only valid within one scenario family. Switching
# scenarios for real means editing this line and starting a fresh process.

# ── Scenario table ────────────────────────────────────────────────────────────
# solver     "trapped" | "collision"      -- which geometry
# tw         True adds Truncated-Wigner noise (a TW* solver class in run.py)
# physical   True derives g from Na-23 + OMEGA_REF_HZ; False uses a bare G
# v_rel_real  the clouds' RELATIVE closing speed in m/s. Each cloud is given
#            half of it in the COM frame, always -- there is no second knob.
#            (There used to be a `v_split` divisor, 4 on the collision rows
#            and 2 on the nucleation ones, so "v_rel_real" on the /4 rows was
#            twice the relative velocity those rows actually simulated. The
#            numbers below were halved to match, so every row's dynamics are
#            byte-for-byte what they were -- only the label is now true.)
# separation      (x,y,z) between the cloud centres. Half of it displaces each
#            cloud, in opposite directions.
# impact_offset   (x,y,z) impact parameter, split between the clouds the same
#            way and added to `separation`. A component transverse to the kick
#            makes the clouds shear past each other by that much rather than
#            meeting head-on. (0,0,0) = head-on.
#            ** A nonzero transverse offset breaks the symmetry by itself, so
#            the mean-field control nucleates vortices too and the clean
#            TW-vs-control A/B is lost. Useful as a "does anything happen"
#            probe; not as the headline result. **
OMEGA_REF_HZ = 4.57              # reference trap frequency, Hz (all Na-23 rows)
ANISOTROPY = np.sqrt(8.0)        # omega_y = omega_z = omega_x / ANISOTROPY.
                                  # FLAGGED: the source paper has z as the TIGHT
                                  # axis (omega_z = omega_x*sqrt(8)) and collides
                                  # along the wide x -- i.e. OMEGA should read
                                  # (1, 1, ANISOTROPY). Not silently flipped:
                                  # every collision row's L/N/velocity gate was
                                  # tuned against the current geometry, so this
                                  # needs a full re-tune and re-gate.
_WIDE = (1.0, 1.0 / ANISOTROPY, 1.0 / ANISOTROPY)
# The source paper's actual geometry: z tight (omega_z = sqrt(8)*omega_x),
# x and y wide, collision along the wide x. Used by the halo_* rows only --
# the collision/nucleation rows keep _WIDE because their L/N/velocity gates
# were tuned against that (flipped) geometry and would need a full re-tune.
_OBLATE = (1.0, 1.0, ANISOTROPY)

# Shared by both nucleation rows so the run and its control cannot drift apart.
_NUCLEATION = dict(
    solver="collision", physical=True, omega=_WIDE,
    n_per_cloud=5.0e3, n_clouds=2.0,
    v_rel_real=4.0e-3,
    separation=(8.0, 0.0, 0.0), impact_offset=(0.0, 3.0, 0.0),
    cutoff=3.0, cutoff_n_ref=96, cutoff_exponent=0.9, cutoff_nyquist_safety=3.0,
    seed=int(time()), order4_dt_multiplier=200.0,
    batch_size=2,          # detection needs each trajectory's own field anyway
    N=256, L=24.0,         # 250 = 2*5^3, all small prime factors. Was 248 =
                            # 2^3*31: FFT cost is set by N's LARGEST prime
                            # factor, and a factor of 31 pushes cuFFT onto its
                            # Bluestein path (~25% slower per point measured on
                            # CPU, worse on GPU) for a 0.8% change in dx.
                            # backend.check_fft_grid_size() warns about this and
                            # suggests exactly 250. N=384 (dx=0.063) resolves
                            # cores better; 250 is the quick look. run.py prints
                            # dx/xi, and the detector raises if the windings
                            # stop being integers.
    t_total=2.0,           # the snake instability needs several xi/c_s AFTER
                            # the clouds overlap; stopping at overlap shows
                            # fringes and no vortices, which reads as a null
                            # result but is just an early stop.
)

# ── Paper-calibrated collision halo ───────────────────────────────────────────
# A. A. Norrie, R. J. Ballagh & C. W. Gardiner, PRA 73, 043617 (2006)
# [cond-mat/0602061], Sec. IV -- the Na-23 parameter set behind its Figs. 1-5.
# Every number below is one of theirs converted into this code's units, with
# the conversion written out so it can be rechecked rather than trusted.
#
# UNITS. The paper measures length in x0 = sqrt(hbar/(2*m*omega_x)); this code
# uses l = sqrt(hbar/(m*omega_x)) = sqrt(2)*x0. So x_here = x_paper/sqrt(2) and
# k_here = sqrt(2)*k_paper. Time and energy units are the same in both
# (1/omega_x and hbar*omega_x), so their times and mu carry over unchanged.
#
#   paper                          here
#   U~0 = 1e-2                     G = 4*pi*a/l = 3.523e-3, and U~0 = 2*sqrt(2)*G
#                                  = 9.97e-3. OMEGA_REF_HZ = 4.57 with
#                                  a = 2.75 nm IS their system -- that is where
#                                  4.57 Hz comes from.
#   lambda = omega_z/omega_xy = sqrt(8)    OMEGA = _OBLATE
#   Dq~ = 10  (relative 4.0 mm/s)  v_rel_real = 4.0e-3 -> 14.20 natural, i.e.
#                                  +-7.10 per cloud (exact would be 7.071; the
#                                  0.4% is a=2.75nm vs the 2.756nm their
#                                  U~0 = 1e-2 implies)
#   k~cut = 18                     k_cut = 25.46           (see `cutoff`)
#   V~ = 48.9 x 33.5 x 33.5        34.6 x 23.7 x 23.7 -> cubic L = 35
#   t = 0 .. 37.7 ms               t_total = 1.08   (1 time unit = 34.83 ms;
#                                  their figure times 8.2/12.6/16.4/25.1/37.7 ms
#                                  are t = 0.235/0.362/0.471/0.721/1.082)
#
# ATOM NUMBER -- an inconsistency in the paper, flagged, not silently resolved.
# The text says the total population is two million, but its own quoted
# mu~_TF = 21.4, n~(0) = 2140 and x~_TF = 9.24 are the Thomas-Fermi values for
# ONE million (two million gives mu~ = 28.15, x~_TF = 10.61). We take the
# explicit "two million", which is also what its Fig. 1 caption independently
# supports: a central condensate mode population of 3.2e4, against 2.8e4
# computed for 2e6 and 0.93e4 for 1e6. To take the other reading instead, set
# n_per_cloud = 1.0e6 and cutoff = 5.42 (which keeps the same k_cut).
_HALO = dict(
    solver="collision", physical=True, omega=_OBLATE,

    # Their Eq. (56) is a Bragg split: ONE condensate carrying two momentum
    # components at the SAME place, not two clouds pushed apart in space. So
    # separation = 0, and bragg_split says the ground state to relax is the
    # WHOLE condensate rather than one cloud's worth -- see GROUND_STATE_N
    # below. The populations stay honest: 1e6 atoms travelling each way, 2e6 in
    # total. (An earlier version smuggled this in as n_clouds=1 with
    # n_per_cloud=2e6, which tests/test_regressions.py rightly rejected --
    # every collision row must have n_clouds=2.)
    bragg_split=True,
    n_per_cloud=1.0e6, n_clouds=2.0,
    v_rel_real=4.0e-3,
    separation=(0.0, 0.0, 0.0),      # Bragg pulse, not a spatial split
    impact_offset=(0.0, 0.0, 0.0),   # head-on by construction

    # GRID. L = 35 is the paper's box rounded up on its long (collision) axis;
    # this engine is cubic, so y and z get 35 too instead of 23.7 -- harmless
    # except that the bigger box holds proportionally more vacuum modes.
    # N = 320 -> dx = 0.1097, k_Nyquist = 28.6. That is the deliberate
    # compromise: it fits ~1.4 GiB (45.5 B/point, performance.md) and runs in
    # minutes, at the cost of the cutoff below. N = 448 (~3.8 GiB) affords the
    # paper's own k_cut; see `cutoff`.
    N=448, L=35.0,
    t_total=1.08,
    # Measured, not assumed: relaxing this row's ground state gives 2.000e6
    # atoms, an envelope peak of 7990 (mu = G*n0 = 28.151 against the
    # Thomas-Fermi 28.153) and x_TF = 7.6, z_TF = 2.9 -- i.e. the imaginary-time
    # state really is the paper's condensate. The Eq. (56) fringe peak comes out
    # at 1.91x the envelope rather than 2x on a coarse grid, because the fringe
    # period is 2*pi/(Dq/2) = 0.885 and needs ~8 cells; N=320 gives 8.1.

    # NOISE CUTOFF. energy_cutoff_mask sets E_cut = cutoff * G * n_peak, where
    # n_peak is the FRINGE peak of Eq. (56), 2*mu/G = 15980, so G*n_peak = 56.3
    # and k_cut = sqrt(2*E_cut). The paper's requirement is k_cut > 3*Dq/2 =
    # 21.21 -- every scattering channel involving the condensates and the halo
    # inside the low-energy subspace -- and it uses k_cut = 25.46 for margin.
    #     cutoff 4.5  -> E_cut = 253 -> k_cut = 22.5   (this row: clears the
    #                    requirement by ~6%, which survives the fringe peak
    #                    coming out a few percent under 2*mu/G on a real grid)
    #     cutoff 5.75 -> E_cut = 324 -> k_cut = 25.46  (the paper's own value)
    # At N=320 the Nyquist wavenumber is 28.6, so 25.46 would leave only 1.12x
    # of headroom and the noise aliases straight back through the nonlinearity;
    # 22.5 keeps 1.27x. Raise N and this together -- at N=448 use cutoff=5.75.
    cutoff=5.75,
    # cutoff_exponent = 0 pins k_cut to exactly that value at every N
    # (growth = (N/N_ref)**0 = 1), instead of letting it grow with the grid and
    # drift off the paper's number. Note that noise.py's Nyquist backstop is a
    # ceiling on the GROWTH only -- E_floor is never given up -- so it will not
    # rescue a grid too coarse for this cutoff; it only makes run.py print
    # "ceiling-limited", which on this row means "raise N", not "handled". At
    # N=320 it does not bind (E_cut = 253 against a ceiling of 307).
    cutoff_n_ref=320, cutoff_exponent=0.0, cutoff_nyquist_safety=2.0,

    seed=int(time()),
    # dt. choose_dt's kinetic accuracy limit here is 1.63e-4; the binding
    # constraint is instead that the fastest populated mode (k = k_cut = 22.5)
    # must not cross more than one cell per step, dt <= dx/k_cut = 4.9e-3, i.e.
    # a multiplier of ~30. That gives dt = 4.9e-3 and ~222 steps over t_total.
    # If the energy gate drifts more than a percent or two, halve this before
    # reaching for SPLITSTEP_ORDER = 2.
    order4_dt_multiplier=50.0,
    batch_size=1,   # 45.5 B/point x 320^3 = 1.39 GiB resident; keep it at 1

    # DETECTOR overrides for this row only (see the block at the bottom of this
    # file). Only the recording cadence and the momentum panels are changed --
    # the thresholds stay at the project's swept defaults, for a reason worth
    # writing down.
    #
    # The vacuum's uniform density here is delta_P = k_cut^3/(6*pi^2) = 192
    # against a fringe peak of 15980, so 1.2%, and the paper filters its Fig. 5
    # vortices at exactly that floor (its 3e11 cm^-3 IS delta_P in SI). It is
    # tempting to put mask_threshold there too. DON'T: |delta_psi|^2 is
    # EXPONENTIALLY distributed about delta_P, so a threshold of f*n_peak lets
    # a fraction exp(-f*n_peak/delta_P) of every empty grid point through --
    #     f = 0.03 -> exp(-2.5) = 8%    of the whole box speckles through
    #     f = 0.08 -> exp(-6.7) = 0.1%
    # and each speckle can carry a phase singularity. Measured at f = 0.03:
    # 15,684 "vortices" at t = 0, when the state is a clean condensate, rising
    # to 1.27 MILLION by the end -- 57s per frame to link on the host with the
    # GPU idle, which is what a hung run looks like. f = 0.08 is ~60x cleaner
    # and is what the rest of the project is swept against. Sweep it anyway,
    # with nucleation/runner.py's sweep_mask_threshold() (a stored frame, no GPU
    # time); the t = 0 count is the calibration -- it should be near zero.
    detector=dict(
        # Vacuum rejection. |delta_psi|^2 is exponentially distributed about
        # delta_P = k_cut^3/(6*pi^2), so a threshold of f*n_peak lets a fraction
        # exp(-f*n_peak/delta_P - 1) of every EMPTY grid point through, and each
        # such speckle can carry a phase singularity. At this row's numbers
        # n_peak/delta_P = 84 (n_peak = 15451, delta_P = 183), so over a 384^3
        # grid the expected number of vacuum speckles is
        #     f = 0.08 -> 24,000     (this is what flooded the detector)
        #     f = 0.15 ->     67
        #     f = 0.20 ->      1     <- drops below one speckle in the whole box
        #     f = 0.22 ->      0.2
        # 0.22 is that crossover plus margin. The calibration is the count at
        # t = 0: the initial state is a clean condensate, so it must be ~0 (it
        # was 375 at f = 0.08). Re-derive f if you change N, the cutoff or the
        # atom number -- f ~ (ln(N^3) - 1) * delta_P / n_peak -- or sweep it
        # with nucleation/runner.py's sweep_mask_threshold(), which works on a
        # stored frame for no GPU time.
        #
        # What this buys and what it costs: a clean count, by keeping only the
        # dense region. The halo's own density is a few delta_P, far below
        # 0.22*n_peak, so what survives is "vortices in the surviving
        # condensate", not halo turbulence. A density threshold cannot separate
        # halo vortices from vacuum ones -- the paper hits the same wall and
        # calls its own cut "somewhat arbitrary". The halo itself is unaffected:
        # it lives in the density and momentum panels, not in this number.
        mask_threshold=0.22,
        detect_stride=10,        # ~22 detections instead of 222: the halo is
                                  # the headline here, the vortex count rides along
        slice_stride=2,          # 111 recorded slices, cross-faded to the movie
        density_log=True,        # the halo is 2-3 decades under the condensate;
                                  # on the default linear scale it is invisible

        # The paper's own figures, and the view the halo is unmistakable in.
        # Ordered as they appear there: normal "z" is Fig. 1(a,c,e,g), the plane
        # holding the collision axis, where the halo is a ring between the two
        # departing packets; "x" is Fig. 1(b,d,f,h), transverse, where it is a
        # filled disc; "y" is Fig. 2, which is where the transient 3-wave
        # packets at k_x = +-3*Dq/2 show up.
        momentum_planes=("z", "x", "y"),
        momentum_k_max=26.0,     # holds the halo ring (|k| = Dq/2 = 7.10), the
                                  # 3-wave packets (3*Dq/2 = 21.2) and the noise
                                  # cutoff circle (k_cut = 22.5) -- that circle
                                  # is the aliasing check: nothing should appear
                                  # outside it. Nyquist is 28.6, so cropping
                                  # here throws away only empty corners.
        momentum_vmax=1.0e2,     # the paper's Fig. 1 saturation, chosen so the
                                  # halo (~1e0-1e2 per mode, over a vacuum floor
                                  # of exactly 1/2) uses the whole colour range
                                  # instead of the packets' ~3e4 doing so. Set
                                  # None to see the true peak.
        momentum_ring=7.10,      # |k| = Dq/2, where elastic scattering off two
                                  # packets at k_x = +-Dq/2 MUST put the shell.
                                  # If the bright ring is not on this circle,
                                  # the collision velocity or the units are
                                  # wrong -- it is the cheapest check there is.
    ),
)

SCENARIOS = {
    "trapped_dipole": dict(
        solver="trapped", tw=False, physical=False,
        G=1.0, n_particles=1.0, omega=(1.0, 1.0, 1.0),
        # A ground state is stationary; either of these makes it move.
        initial_offset=(0.0, 1.0, 0.0), initial_velocity=(4.0, 0.0, 0.0),
        N=384, L=16.0, t_total=3.0),

    "collision": dict(
        solver="collision", tw=False, physical=True, omega=_WIDE,
        n_per_cloud=1.0e4,     # scaled down from the paper's ~1e6/cloud
        n_clouds=2.0, v_rel_real=2.0e-3,   # was v_rel_real=4e-3 with v_split=4,
                                            # i.e. the same 2e-3 relative speed
        separation=(6.0, 0.0, 0.0),   # not from the paper: just large enough
                                       # that the clouds don't overlap at t=0
        impact_offset=(0.0, 0.0, 0.0),   # head-on
        N=384, L=16.0, t_total=3.0),

    "tw_trapped_dipole": dict(
        solver="trapped", tw=True, physical=True,
        n_particles=1.0e4, omega=(1.0, 1.0, 1.0),
        n_traj=4, batch_size=1,
        cutoff=2.0, cutoff_n_ref=128, cutoff_exponent=0.85, cutoff_nyquist_safety=2.0,
        seed=81646845484654, order4_dt_multiplier=100.0,
        # L=8, not 16: the noise mode count at fixed E_cut scales with box
        # VOLUME, so an oversized box inflates depletion for no physics.
        # N=384 is past this row's validated range -- depletion is 13% at
        # N=256 already, over run.py's <10% gate (README, "Depletion").
        N=384, L=8.0, t_total=3.0),

    "tw_collision": dict(
        solver="collision", tw=True, physical=True, omega=_WIDE,
        n_per_cloud=5.0e3,
        n_clouds=2.0,          # FIXED (was 4.0): there are two clouds, and
                                # N_PARTICLES = n_clouds*n_per_cloud is what
                                # two_cloud_collision_psi() renormalises the
                                # pair to, so 4.0 put 1e4 atoms in each 5e3
                                # cloud AND measured depletion_fraction
                                # against a target twice the real one.
        v_rel_real=1.0e-3,     # was v_rel_real=2e-3 with v_split=4: same speed
        separation=(6.0, 0.0, 0.0), impact_offset=(0.0, 0.0, 0.0),
        n_traj=4, batch_size=1,
        cutoff=3.0, cutoff_n_ref=128, cutoff_exponent=0.85, cutoff_nyquist_safety=3.0,
        seed=int(time()), order4_dt_multiplier=100.0,
        N=384, L=16.0, t_total=3.0),

    "tw_trapped_thermal": dict(
        solver="trapped", tw=True, physical=True,
        n_particles=3.0e4,     # 3x tw_trapped_dipole: headroom against the
                                # thermal addition to depletion
        omega=(1.0, 1.0, 1.0),
        temperature_fraction_tc=0.10,   # T/Tc. Do not raise much past this --
                                         # the sampler uses bare plane-wave
                                         # modes and over-populates the low-k
                                         # sector once k_B*T approaches mu
                                         # (README, "Finite temperature").
        n_traj=8, batch_size=1,
        cutoff=2.0, cutoff_n_ref=128,
        cutoff_exponent=0.5,   # 0.7 measured 10.23% depletion at N=256 (FAIL);
                                # 0.5 gives 8.83% with real margin
        cutoff_nyquist_safety=2.0, seed=int(time()),
        order4_dt_multiplier=50.0,   # 400 passed the short sweep and then blew
                                      # up (24000% energy drift) on a full
                                      # T_TOTAL run -- see README
        N=256, L=10.0, t_total=3.0),

    "nucleation_collision": dict(_NUCLEATION, tw=True, n_traj=10, batch_size=10),
    # n_traj was 1, which is why the run report said "trajectories: 1" and its
    # CIs collapsed onto the point estimate -- a single realisation has no
    # spread to measure. 8 is the smallest ensemble that gives a non-degenerate
    # bootstrap interval; 20-50 independent seeds for a number worth quoting.
    # batch_size: psi is (batch, 250^3) complex64, ~125 MB per trajectory, so a
    # chunk of 2 is ~0.25 GB before FFT workspace. 2 is
    # _NUCLEATION's own default; raise it only with GPU headroom to spare.

    "nucleation_control": dict(_NUCLEATION, tw=False, n_traj=1),

    # 4 trajectories, not 1: the halo itself shows up in every single one (the
    # paper's Figs. 1-5 are one trajectory), but the vortex count needs a
    # spread. Budget ~5-10 min per trajectory on a 6 GB card.
    "halo_collision": dict(_HALO, tw=True, n_traj=1),
    "halo_control": dict(_HALO, tw=False, n_traj=1),
}

if SCENARIO not in SCENARIOS:
    raise ValueError(f"config.py: unknown SCENARIO {SCENARIO!r} -- "
                      f"one of {sorted(SCENARIOS)}")
_S = SCENARIOS[SCENARIO]

# ── Shared run settings ───────────────────────────────────────────────────────
IMAG_TIME_STEPS = 500   # raise if "converged at step ..." looks premature
SPLITSTEP_ORDER = 4     # 2 = Strang, O(dt^2). 4 = Yoshida, O(dt^4) at 3x the
                         # work per outer step, so it affords a far larger dt.
                         # Set 2 for a direct A/B, or to rule the integrator out.
TOL = 0.01              # pass/fail tolerance on the conserved quantities
SAVE_GIF = True         # mid-plane density-slice animation (non-nucleation runs)

# Movie length is set here, independently of how much physical time the run
# covers: GIF_SECONDS of playback at GIF_FPS needs GIF_SECONDS*GIF_FPS frames,
# whether those depict T_TOTAL=1 or T_TOTAL=8. run.py derives the snapshot
# cadence from this and reports it before evolving.
#
# The ceiling is one snapshot per sub-step -- the solver only has T_TOTAL/dt
# distinct states to show. Ask for more and run.py records every state it has
# and the renderer cross-fades the rest up to GIF_FPS, so the movie still runs
# GIF_SECONDS long; it just carries fewer genuinely independent frames. run.py
# prints how many of the frames are real when that happens.
GIF_SECONDS = 4.0       # playback length of every movie, in real seconds
GIF_FPS = 60            # playback frame rate
SNAPSHOT_ROWS = 3       # report frames when SAVE_GIF is False (3 * this)
OUT_DIR = "outputs"
VERBOSE = True

# ── Derived, per selected scenario ────────────────────────────────────────────
# N and L set dx = L/N, which must resolve every velocity kick in the run: a
# kick is a plane wave exp(i*v*x), and |v| above the Nyquist wavenumber pi/dx
# aliases silently -- norm and energy conservation do not catch it, because
# aliasing is still unitary. run.py checks this before building anything.
N, L, T_TOTAL = _S["N"], _S["L"], _S["t_total"]
OMEGA = _S["omega"]
N_TRAJECTORIES = _S.get("n_traj", 1)
BATCH_SIZE = _S.get("batch_size", 1)
INITIAL_OFFSET = _S.get("initial_offset", (0.0, 0.0, 0.0))
INITIAL_VELOCITY = _S.get("initial_velocity", (0.0, 0.0, 0.0))

if _S["physical"]:
    units = derive_natural_units(NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M, OMEGA_REF_HZ)
    G = units.g            # ~3.523e-3
else:
    units = None
    G = _S["G"]

SOLVER_PARAMS = dict(imag_time_steps=IMAG_TIME_STEPS, verbose=VERBOSE)

if _S["solver"] == "collision":
    N_PER_CONDENSATE = _S["n_per_cloud"]
    N_PARTICLES = _S["n_clouds"] * N_PER_CONDENSATE
    COLLISION_SEPARATION = _S["separation"]
    COLLISION_IMPACT_OFFSET = _S.get("impact_offset", (0.0, 0.0, 0.0))
    # Each cloud carries half the relative velocity, by definition of the COM
    # frame -- see the v_rel_real note in the table header.
    COLLISION_HALF_VELOCITY = (
        0.5 * velocity_to_natural(_S["v_rel_real"], units), 0.0, 0.0)
    # Which state imaginary time relaxes. For a spatial split each cloud is its
    # own ground state, so that is n_per_cloud. For a Bragg split there is one
    # condensate holding every atom and the two momentum components are made
    # from it, so the envelope is the TOTAL -- two_cloud_collision_psi() then
    # kicks two copies of it and renormalises the sum (norm 2*N) back to
    # n_particles = N, giving psi = sqrt(2)*Psi(x)*cos(Dq*x/2), the paper's
    # Eq. (56) exactly. Both cases leave n_per_cloud/n_clouds meaning what they
    # say; only the envelope differs.
    GROUND_STATE_N = N_PARTICLES if _S.get("bragg_split") else N_PER_CONDENSATE
    SOLVER_PARAMS.update(n_particles_per_cloud=GROUND_STATE_N,
                          separation=COLLISION_SEPARATION,
                          impact_offset=COLLISION_IMPACT_OFFSET,
                          half_velocity=COLLISION_HALF_VELOCITY)
else:
    N_PARTICLES = _S["n_particles"]
    COLLISION_HALF_VELOCITY = COLLISION_IMPACT_OFFSET = (0.0, 0.0, 0.0)
    SOLVER_PARAMS.update(initial_offset=INITIAL_OFFSET,
                          initial_velocity=INITIAL_VELOCITY)
SOLVER_PARAMS["n_particles"] = N_PARTICLES

# Left in place even for nucleation_control, whose classical solver ignores
# them -- so the run and its control stay one table row.
if "cutoff" in _S:
    SOLVER_PARAMS.update(
        cutoff_multiplier=_S["cutoff"],
        scale_cutoff_to_grid=True,   # let a finer grid resolve more small-scale
                                      # vacuum structure; see gpe3d/noise.py's
                                      # scale_cutoff_multiplier_to_grid for the
                                      # growth law and its depletion caveat.
                                      # cutoff_exponent=0 pins it instead.
        cutoff_scale_n_ref=_S["cutoff_n_ref"],
        cutoff_scale_exponent=_S["cutoff_exponent"],
        cutoff_nyquist_safety=_S["cutoff_nyquist_safety"],
        seed=_S["seed"],
        order4_dt_multiplier=_S["order4_dt_multiplier"],
    )

TEMPERATURE_FRACTION_TC = _S.get("temperature_fraction_tc", 0.0)
if TEMPERATURE_FRACTION_TC:
    TEMPERATURE_NATURAL = TEMPERATURE_FRACTION_TC * condensate_critical_temperature_natural(
        OMEGA, N_PARTICLES)
    TEMPERATURE_KELVIN = natural_temperature_to_kelvin(TEMPERATURE_NATURAL, units)  # ~nK
    SOLVER_PARAMS["temperature_natural"] = TEMPERATURE_NATURAL

# ── Vortex detector (nucleation_* scenarios only) ─────────────────────────────
# Read by nucleation/settings.from_config(); anything unset falls back to that
# module's DetectorConfig defaults.
#
# The values written here are the defaults for every nucleation_* row. A single
# scenario can override any of them with a `detector=dict(...)` entry in its own
# row (halo_collision does) without disturbing the others.
#
# Three knobs can MANUFACTURE vortices, and all three must be swept before a
# count is quotable: the noise cutoff (`cutoff` in the table above -- watch the
# depletion_fraction run.py prints), NUCLEATION_MASK_THRESHOLD below
# (nucleation/runner.py's sweep_mask_threshold() does this on one stored frame,
# so it costs no GPU time), and N (run.py prints dx/xi).
_DET = _S.get("detector", {})

NUCLEATION_MASK_THRESHOLD = _DET.get("mask_threshold", 0.08)
                                    # |psi|^2 > f * n_peak is "inside the cloud"
NUCLEATION_MASK_CLOSE_PASSES = _DET.get("mask_close_passes", 4)   # closing of that mask, in cells. Must exceed
                                    # the core radius (~xi/dx): a resolved core
                                    # has |psi|^2 -> 0 at its centre, so it
                                    # punches a hole in a raw mask and deletes
                                    # exactly the plaquettes worth counting.
NUCLEATION_REQUIRE_DENSITY_DIP = _DET.get("require_density_dip", True)   # winding AND a local density minimum
NUCLEATION_DIP_FACTOR = _DET.get("dip_factor", 0.75)
NUCLEATION_NORMALS = _DET.get("normals", ("x", "y", "z"))    # all three plane families: a line is
                                         # found by the planes it PIERCES, so
                                         # z-planes alone miss a ring lying in
                                         # the y-z plane entirely
NUCLEATION_INTEGRALITY_TOL = _DET.get("integrality_tol", 0.25)       # abort if windings aren't this integer
NUCLEATION_MAX_PIERCE_POINTS = _DET.get("max_pierce_points", 400_000)
                                # skip line tracing on a frame with more pierce
                                # points than this. Linking is host-side, so a
                                # frame with a million of them stalls the run for
                                # minutes with the GPU idle -- and a million
                                # singularities is the TW vacuum passing the
                                # mask, not vortices. Raise MASK_THRESHOLD.
NUCLEATION_LINK_CUTOFF_DX = _DET.get("link_cutoff_dx", 1.8)    # pierce points this close are one line
NUCLEATION_CLOSE_CUTOFF_DX = _DET.get("close_cutoff_dx", 2.5)   # traced ends this close -> a closed ring

# Recording cadence. Detection is three full-grid winding sweeps per trajectory
# per frame; a slice is two 2D planes pulled off the GPU. So record slices
# densely and detect sparsely, rather than tying both to one number.
NUCLEATION_DETECT_STRIDE = _DET.get("detect_stride", 1)   # detect every N-th recorded frame. This, not the
                                # frame count, is the cost knob for a nucleation
                                # run: a slice is two 2D planes off the GPU, a
                                # detection is three full-grid winding sweeps per
                                # trajectory. run.py prints both counts.
NUCLEATION_SLICE_STRIDE = _DET.get("slice_stride", 1)    # record a movie frame every N-th frame
NUCLEATION_SLICE_PLANES = _DET.get("slice_planes", ("x", "y", "z"))   # one figure column per plane
NUCLEATION_SLICE_TRAJECTORY = _DET.get("slice_trajectory", 0)   # which trajectory the movie follows. Never an
                                   # ensemble average: vortices nucleate in
                                   # different places in every trajectory, so
                                   # averaging is exactly what erases them.
NUCLEATION_DENSITY_LOG = _DET.get("density_log", False)
                                # log colour scale on the movie's density panels.
                                # Off by default (a single condensate spans one
                                # decade and linear reads better); ON for any run
                                # whose point is a low-density feature sitting
                                # under a bright cloud -- a scattering halo is
                                # 2-3 decades down, which is why the source
                                # paper's Fig. 4 is logarithmic.
NUCLEATION_DENSITY_LOG_FLOOR = _DET.get("density_log_floor", 1.0e-4)
                                # bottom of that log scale, as a fraction of the
                                # frame's 99.5th-percentile density

# Momentum-space movie (momentum.gif), one column per plane normal. Empty = off.
# A scattering halo is a SHELL in k-space, so a k-plane cuts it as a clean ring
# -- in coordinate space the same atoms are a diffuse glow smeared under two
# bright clouds. Costs one (N,N) transform per plane per recorded frame; no 3D
# FFT (nucleation/observer.py's momentum_plane explains why).
NUCLEATION_MOMENTUM_PLANES = _DET.get("momentum_planes", ())
NUCLEATION_MOMENTUM_K_MAX = _DET.get("momentum_k_max", None)
                                # crop to |k| <= this; None = out to Nyquist,
                                # which is mostly empty space around the ring
NUCLEATION_MOMENTUM_FLOOR = _DET.get("momentum_floor", 0.2)
                                # bottom of the log scale in ATOMS PER MODE, not
                                # a fraction: the TW vacuum is exactly 1/2 per
                                # mode, so 0.2 puts the floor just under it and
                                # everything brighter is signal
NUCLEATION_MOMENTUM_VMAX = _DET.get("momentum_vmax", None)
                                # None = data max; ~1e2 saturates the condensate
                                # packets the way the source paper's figures do
NUCLEATION_MOMENTUM_RING = _DET.get("momentum_ring", None)
                                # dashed circle at this |k| -- the expected halo
                                # radius, so the panel is a check, not a picture

NUCLEATION_VORTEX_MAP = True   # also write vortex_map.gif from pierce_points.npz
                                # -- no GPU time, re-renderable afterwards, and
                                # the view that stays readable once a tangle has
                                # hundreds of lines in it
