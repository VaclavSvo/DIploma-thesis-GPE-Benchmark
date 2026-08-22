"""Explore how far _ORDER4_DT_MULTIPLIER can safely go, on your actual
collision config (N=128 or whatever config.py currently has, real physics),
given README.md's mult=4 run showed only 0.009% energy drift against a
5% budget -- ~500x headroom.

Runs order=2 once as reference, then order=4 at each candidate multiplier in
SWEEP_MULTIPLIERS, evolving T_SWEEP simulated time (default 1.2, not the full
T_TOTAL) at each -- short on purpose: with COLLISION_SEPARATION=6 and the
scenario's ~7 (natural units) closing velocity, the clouds finish colliding
around t~0.86, so T_SWEEP=1.2 already captures the overlap/interaction peak
(the numerically hardest part -- the rest of a full T_TOTAL=5 run is mostly
gentler post-collision free dispersion), while running much faster than a
full sweep would.

Sweep order is BIGGEST multiplier first, descending -- the risky end. Once
3 CONSECUTIVE multipliers (in that descending order) all pass gates, the
sweep stops early: once you're 3 steps deep into passing territory, smaller
(safer) values in the list are essentially guaranteed to also pass (drift
shrinks monotonically-ish as dt shrinks), so testing them just burns wall
time without changing the answer. This also means the sweep can spend most
of its time near the actual crossover instead of re-confirming values
already known safe (mult=4).

This is an exploration tool, not the final verdict -- CONFIRM the multiplier
you land on with one full T_TOTAL run (e.g. via bench_orders_collision.py,
after editing _ORDER4_DT_MULTIPLIER in gpe3d/physics.py) before trusting it
for production runs.

Run with: python tests/sweep_order4_dt.py
"""
import sys
import os
import re
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run as run_module
import gpe3d.physics as physics
from gpe3d.backend import HAS_GPU, check_fft_grid_size
from gpe3d import evolve, gates

SWEEP_MULTIPLIERS = sorted(( 24,  48,  128, 186,  248, 300, 364, 450, 600, 700, 1000), reverse=True)  # biggest (riskiest) first
STOP_AFTER_CONSECUTIVE_PASSES = 4
T_SWEEP = 1.0          # simulated time per sweep point (see docstring)
SAFETY_HEADROOM = 10.0  # recommend the largest mult whose worst drift still
                        # leaves at least this much margin below tol (i.e.
                        # worst drift < tol/SAFETY_HEADROOM), not just "passed"


def _worst_drift_fraction(results):
    """Largest (drift / tol_equivalent) across all gates, from each
    GateResult's own printed detail string -- reuses gates.py's own
    computed numbers rather than recomputing drift independently."""
    worst = 0.0
    for r in results:
        m_drift = re.search(r"drift ([\d.eE+-]+)%?", r.detail)
        m_tol = re.search(r"tol ([\d.eE+-]+)%?", r.detail)
        if not (m_drift and m_tol):
            continue
        drift, tol = float(m_drift.group(1)), float(m_tol.group(1))
        if tol > 0:
            worst = max(worst, drift / tol)
    return worst


def run_case(order: int, T_total: float):
    solver, state, check_momentum = run_module.build_solver_and_state()
    if state is None:
        # TW scenario: build_solver_and_state() only calls prepare_mean_field()
        # (ground state + dt + noise cutoff mask), it doesn't draw a trajectory
        # -- see run.py's own comment on this. Draw ONE noise trajectory here,
        # same convention tests/test_tw_milestone_d.py's
        # test_order4_tw_multiplier_passes_at_physical_parameters() uses (single
        # trajectory, check_momentum already False for every TW scenario via
        # run.py's SCENARIOS table, so this is directly comparable to that test).
        state = solver.sample_chunk(1, seed=config.SOLVER_PARAMS.get("seed", 0))
    dt = solver.engine.dt
    n_blocks, steps_per_block = 6, max(1, round(T_total / dt / 6))
    total_steps = steps_per_block * n_blocks

    t0 = time.time()
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block, track_density=False)
    wall = time.time() - t0

    r_char = 1.0 / (2.0 * min(config.OMEGA)) ** 0.5
    results = gates.run_gates(history, tol=config.TOL, r_char=r_char, check_momentum=check_momentum)
    solver.engine.free()
    return dict(dt=dt, total_steps=total_steps, wall=wall, results=results,
                worst_frac=_worst_drift_fraction(results))


if __name__ == "__main__":
    print(f"backend: {'CuPy (GPU)' if HAS_GPU else 'NumPy (CPU)'}  scenario: {config.SCENARIO}  "
          f"N={config.N}  L={config.L}  T_SWEEP={T_SWEEP} (config.T_TOTAL={config.T_TOTAL})")
    print(f"sweep order (descending): {SWEEP_MULTIPLIERS}, "
          f"stopping early after {STOP_AFTER_CONSECUTIVE_PASSES} consecutive passes")
    check_fft_grid_size(config.N)

    # No order=2 reference run -- skipped on purpose (too slow at N=256^3 to
    # spend on a number this script doesn't actually need: the recommendation
    # below is driven entirely by each order=4 candidate's own gate margin,
    # not by a comparison to order=2). Get an order=2 timing number instead
    # from bench_orders_collision.py if/when you want one.
    rows = []
    consecutive_passes = 0
    stopped_early = False
    _is_tw = run_module.SCENARIOS[config.SCENARIO]["is_tw"]
    for mult in SWEEP_MULTIPLIERS:
        physics._ORDER4_DT_MULTIPLIER = float(mult)
        # TW scenarios' config.py block always sets an EXPLICIT
        # order4_dt_multiplier in SOLVER_PARAMS (gpe3d/base.py's
        # _finalize_state prefers that explicit value over the module-level
        # physics._ORDER4_DT_MULTIPLIER whenever it's not None) -- so for a TW
        # scenario, the line above alone is a silent no-op: every mult in this
        # sweep would use the SAME config-supplied multiplier, making the
        # "sweep" meaningless (every row identical) without this override too.
        if _is_tw:
            config.SOLVER_PARAMS["order4_dt_multiplier"] = float(mult)
        config.SPLITSTEP_ORDER = 4
        r = run_case(4, T_SWEEP)
        all_pass = all(g.passed for g in r["results"])
        rows.append((mult, r, all_pass))
        print(f"\norder=4 mult={mult:3d}: dt={r['dt']:.4e} steps={r['total_steps']} "
              f"wall={r['wall']:.2f}s  gates={'PASS' if all_pass else 'FAIL'}  "
              f"worst_drift/tol={r['worst_frac']:.4f}")
        for g in r["results"]:
            print(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.detail}")

        consecutive_passes = consecutive_passes + 1 if all_pass else 0
        if consecutive_passes >= STOP_AFTER_CONSECUTIVE_PASSES:
            stopped_early = mult != SWEEP_MULTIPLIERS[-1]
            print(f"\n{consecutive_passes} consecutive passes reached at mult={mult} -- "
                  f"stopping early, smaller multipliers are safely inside passing territory.")
            break

    print("\n=== Sweep summary ===")
    print(f"{'mult':>5} {'dt':>10} {'steps':>7} {'wall(s)':>8} {'gates':>6} {'worst_drift/tol':>16}")
    for mult, r, all_pass in rows:
        print(f"{mult:>5} {r['dt']:>10.3e} {r['total_steps']:>7d} {r['wall']:>8.2f} "
              f"{'PASS' if all_pass else 'FAIL':>6} {r['worst_frac']:>16.4f}")
    if stopped_early:
        skipped = [m for m in SWEEP_MULTIPLIERS if m < rows[-1][0]]
        print(f"(stopped early -- {skipped} not tested, assumed safe since they're smaller "
              f"than {STOP_AFTER_CONSECUTIVE_PASSES} consecutive passing values)")

    passing_with_headroom = [mult for mult, r, all_pass in rows
                              if all_pass and r["worst_frac"] < 1.0 / SAFETY_HEADROOM]
    print()
    if passing_with_headroom:
        best_safe = max(passing_with_headroom)
        print(f"Recommended: _ORDER4_DT_MULTIPLIER = {best_safe} "
              f"(largest tested value keeping worst drift under tol/{SAFETY_HEADROOM:.0f}).")
        print("Next step: set that in gpe3d/physics.py, then confirm with a full "
              "T_TOTAL run (bench_orders_collision.py) before trusting it in production.")
    else:
        print("No swept multiplier kept enough safety margin -- stick with the current "
              "default (4) or try smaller candidates (edit SWEEP_MULTIPLIERS).")
