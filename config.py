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
SCENARIO = "nucleation_collision"

# ── Scenario table ────────────────────────────────────────────────────────────
# solver     "trapped" | "collision"      -- which geometry
# tw         True adds Truncated-Wigner noise (a TW* solver class in run.py)
# physical   True derives g from Na-23 + OMEGA_REF_HZ; False uses a bare G
# v_split    each cloud's COM-frame speed is v_rel / v_split. The collision
#            scenarios use /4, the nucleation ones /2; /2 is the correct
#            "each cloud carries half the relative velocity" convention, and
#            the /4 rows are kept as-is because their grid/velocity gate was
#            tuned against them.
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

# Shared by both nucleation rows so the run and its control cannot drift apart.
_NUCLEATION = dict(
    solver="collision", physical=True, omega=_WIDE,
    n_per_cloud=5.0e3, n_clouds=2.0,
    v_rel_real=2.0e-3, v_split=2.0, separation=(8.0, 0.0, 0.0),
    cutoff=3.0, cutoff_n_ref=128, cutoff_exponent=0.85, cutoff_nyquist_safety=3.0,
    seed=54579554444758, order4_dt_multiplier=100.0,
    batch_size=1,          # detection needs each trajectory's own field anyway
    N=384, L=16.0,         # N=384 (dx=0.042) resolves cores better; 256 is the
                            # quick look. run.py prints dx/xi, and the detector
                            # raises if the windings stop being integers.
    t_total=5.0,           # the snake instability needs several xi/c_s AFTER
                            # the clouds overlap; stopping at overlap shows
                            # fringes and no vortices, which reads as a null
                            # result but is just an early stop.
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
        n_clouds=2.0, v_rel_real=4.0e-3, v_split=4.0,
        separation=(8.0, 0.0, 0.0),   # not from the paper: just large enough
                                       # that the clouds don't overlap at t=0
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
        n_clouds=4.0,          # FLAGGED: inconsistent with n_per_cloud (two
                                # clouds), and depletion_fraction is measured
                                # against it. The nucleation rows use 2.0.
        v_rel_real=2.0e-3, v_split=4.0, separation=(8.0, 0.0, 0.0),
        n_traj=4, batch_size=1,
        cutoff=3.0, cutoff_n_ref=128, cutoff_exponent=0.85, cutoff_nyquist_safety=3.0,
        seed=54579597972457297527, order4_dt_multiplier=100.0,
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
        cutoff_nyquist_safety=2.0, seed=0,
        order4_dt_multiplier=50.0,   # 400 passed the short sweep and then blew
                                      # up (24000% energy drift) on a full
                                      # T_TOTAL run -- see README
        N=256, L=10.0, t_total=3.0),

    "nucleation_collision": dict(_NUCLEATION, tw=True, n_traj=8),
    # 4 trajectories is fine for the conservation gates and useless for
    # nucleation statistics; 20-50 independent seeds for a real number.

    "nucleation_control": dict(_NUCLEATION, tw=False, n_traj=1),
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
GIF_FPS = 60            # frame count is chosen so playback length == T_TOTAL
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
INITIAL_OFFSET = _S.get("initial_offset", (10.0, 0.0, 0.0))
INITIAL_VELOCITY = _S.get("initial_velocity", (4.0, 0.0, 0.0))

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
    COLLISION_HALF_VELOCITY = (
        velocity_to_natural(_S["v_rel_real"], units) / _S["v_split"], 0.0, 0.0)
    SOLVER_PARAMS.update(n_particles_per_cloud=N_PER_CONDENSATE,
                          separation=COLLISION_SEPARATION,
                          half_velocity=COLLISION_HALF_VELOCITY)
else:
    N_PARTICLES = _S["n_particles"]
    COLLISION_HALF_VELOCITY = (0.0, 0.0, 0.0)
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
                                      # growth law and its depletion caveat
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
# Three knobs can MANUFACTURE vortices, and all three must be swept before a
# count is quotable: the noise cutoff (`cutoff` in the table above -- watch the
# depletion_fraction run.py prints), NUCLEATION_MASK_THRESHOLD below
# (nucleation/runner.py's sweep_mask_threshold() does this on one stored frame,
# so it costs no GPU time), and N (run.py prints dx/xi).
NUCLEATION_MASK_THRESHOLD = 0.08   # |psi|^2 > f * n_peak is "inside the cloud"
NUCLEATION_MASK_CLOSE_PASSES = 4   # closing of that mask, in cells. Must exceed
                                    # the core radius (~xi/dx): a resolved core
                                    # has |psi|^2 -> 0 at its centre, so it
                                    # punches a hole in a raw mask and deletes
                                    # exactly the plaquettes worth counting.
NUCLEATION_REQUIRE_DENSITY_DIP = True   # winding AND a local density minimum
NUCLEATION_DIP_FACTOR = 0.75
NUCLEATION_NORMALS = ("x", "y", "z")    # all three plane families: a line is
                                         # found by the planes it PIERCES, so
                                         # z-planes alone miss a ring lying in
                                         # the y-z plane entirely
NUCLEATION_INTEGRALITY_TOL = 0.25       # abort if windings aren't this integer
NUCLEATION_LINK_CUTOFF_DX = 1.8    # pierce points this close are one line
NUCLEATION_CLOSE_CUTOFF_DX = 2.5   # traced ends this close -> a closed ring

# Recording cadence. Detection is three full-grid winding sweeps per trajectory
# per frame; a slice is two 2D planes pulled off the GPU. So record slices
# densely and detect sparsely, rather than tying both to one number.
NUCLEATION_FRAMES = 120        # recorded frames per run, and the movie's time
                                # resolution -- rendering cross-fades these up
                                # to NUCLEATION_GIF_FPS. Past ~200 this mostly
                                # buys memory pressure, not smoothness.
NUCLEATION_DETECT_STRIDE = 2   # detect every N-th frame
NUCLEATION_SLICE_STRIDE = 1    # record a movie frame every N-th frame
NUCLEATION_SLICE_PLANES = ("x", "y", "z")   # one figure column per plane
NUCLEATION_SLICE_TRAJECTORY = 0   # which trajectory the movie follows. Never an
                                   # ensemble average: vortices nucleate in
                                   # different places in every trajectory, so
                                   # averaging is exactly what erases them.
NUCLEATION_GIF_FPS = 60        # playback rate of both movies (length = T_TOTAL)
NUCLEATION_VORTEX_MAP = True   # also write vortex_map.gif from pierce_points.npz
                                # -- no GPU time, re-renderable afterwards, and
                                # the view that stays readable once a tangle has
                                # hundreds of lines in it
