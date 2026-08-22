"""Apples-to-apples order=2 vs order=4 wall-clock comparison, on whatever
scenario/grid config.py currently specifies (reuses run.py's own solver-
building code so there's no chance of the two runs silently using different
N/L/g -- the thing that made the first README.md comparison ambiguous:
the order=2 and order=4 numbers in it came from two different config.py
edits, most likely different L, so the timing difference conflated "which
order" with "which grid" and couldn't be trusted for a clean verdict).

Not a gate, not part of the test suite -- a one-off you run manually:
    python tests/bench_orders_collision.py

Skips the gif (SAVE_GIF is honoured only for actual run.py runs) to keep
this fast and focused on timing + gates. Both orders run back-to-back in
this one process against the *same* config.py, so N/L/g/omega/T_TOTAL are
guaranteed identical between the two -- only splitstep_order differs.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run as run_module
from gpe3d.backend import HAS_GPU, check_fft_grid_size
from gpe3d import evolve, gates


def bench_one_order(order: int):
    config.SPLITSTEP_ORDER = order  # only touches the in-memory module, not config.py on disk
    solver, state, check_momentum = run_module.build_solver_and_state()

    dt = solver.engine.dt
    total_frames = max(2, round(config.T_TOTAL * config.GIF_FPS))
    n_blocks = total_frames - 1
    steps_per_block = max(1, round(config.T_TOTAL / dt / n_blocks))
    total_steps = steps_per_block * n_blocks

    t0 = time.time()
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block, track_density=False)
    wall = time.time() - t0

    r_char = 1.0 / (2.0 * min(config.OMEGA)) ** 0.5
    results = gates.run_gates(history, tol=config.TOL, r_char=r_char, check_momentum=check_momentum)
    all_pass = all(r.passed for r in results)

    solver.engine.free()
    return dict(order=order, dt=dt, total_steps=total_steps, wall=wall,
                steps_per_s=total_steps / wall, all_pass=all_pass, results=results)


if __name__ == "__main__":
    print(f"backend: {'CuPy (GPU)' if HAS_GPU else 'NumPy (CPU)'}  scenario: {config.SCENARIO}  "
          f"N={config.N}  L={config.L}  T_TOTAL={config.T_TOTAL}")
    check_fft_grid_size(config.N)

    rows = []
    for order in (2, 4):
        print(f"\n--- order={order} ---")
        r = bench_one_order(order)
        rows.append(r)
        print(gates.report(r["results"]))
        print(f"dt={r['dt']:.4e}  total_steps={r['total_steps']}  "
              f"wall={r['wall']:.2f}s  ({r['steps_per_s']:.1f} steps/s)")

    print("\n=== Summary (same N/L/g/omega/T_TOTAL for both) ===")
    r2, r4 = rows[0], rows[1]
    print(f"order=2: dt={r2['dt']:.4e}  steps={r2['total_steps']:6d}  wall={r2['wall']:8.2f}s  "
          f"gates={'PASS' if r2['all_pass'] else 'FAIL'}")
    print(f"order=4: dt={r4['dt']:.4e}  steps={r4['total_steps']:6d}  wall={r4['wall']:8.2f}s  "
          f"gates={'PASS' if r4['all_pass'] else 'FAIL'}")
    dt_ratio = r4["dt"] / r2["dt"]
    wall_ratio = r4["wall"] / r2["wall"]
    print(f"dt(order=4)/dt(order=2) = {dt_ratio:.2f}x   "
          f"wall(order=4)/wall(order=2) = {wall_ratio:.2f}x "
          f"({'order=4 net faster' if wall_ratio < 1 else 'order=2 net faster'} at matched simulated time)")
