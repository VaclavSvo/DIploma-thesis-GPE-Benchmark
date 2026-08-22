"""Standalone wall-clock benchmark, not a gate -- run manually to compare
before/after an optimization change. Times ground-state prep + evolving a
fixed *simulated time* T_TOTAL (not a fixed step count, so order=2 and
order=4 -- which pick different dt via choose_dt() -- are compared on an
apples-to-apples "time to reach the same physical point" basis) on this
sandbox's backend, for both splitstep orders at a couple of N.
Run with `python tests/bench_perf.py`.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.solver import ClassicalTrappedGPESolver


def bench(N, L, order, T_total=0.3, g=1.0, omega=(1.0, 1.0, 1.0)):
    solver = ClassicalTrappedGPESolver(N=N, L=L, g=g, omega=omega, splitstep_order=order)
    t0 = time.perf_counter()
    state = solver.init_state(dict(n_particles=1.0, imag_time_steps=600, verbose=False))
    t_ground = time.perf_counter() - t0

    dt = solver.engine.dt
    n_steps = max(1, round(T_total / dt))

    t1 = time.perf_counter()
    solver.call(state, n_steps)
    t_evolve = time.perf_counter() - t1

    solver.engine.free()
    return t_ground, dt, n_steps, t_evolve


if __name__ == "__main__":
    print(f"{'N':>4} {'order':>5} {'dt':>10} {'n_steps':>8} {'ground(s)':>10} {'evolve(s)':>10}")
    rows = {}
    for N in (32, 48, 64):
        for order in (2, 4):
            tg, dt, n_steps, te = bench(N, L=12.0, order=order)
            rows[(N, order)] = te
            print(f"{N:>4} {order:>5} {dt:>10.3e} {n_steps:>8d} {tg:>10.3f} {te:>10.3f}")
    print()
    print(f"{'N':>4} {'order=4 evolve time / order=2 evolve time':>45}")
    for N in (32, 48, 64):
        r = rows[(N, 4)] / rows[(N, 2)]
        print(f"{N:>4} {r:>45.3f}  ({'order=4 faster' if r < 1 else 'order=2 faster'})")
