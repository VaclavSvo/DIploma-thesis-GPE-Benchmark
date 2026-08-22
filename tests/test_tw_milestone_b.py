"""Milestone B (README.md): one noisy trajectory through the
*unmodified* propagator. Gate: norm/energy conserved to the same tight
tolerance as a noiseless run -- a noisy initial condition doesn't break
unitarity. If this fails, the bug is in the noise sampler producing a
badly-normalized/inconsistent state, not in the propagator (Milestone A
already validated the sampler in isolation).

Momentum/angular-momentum are deliberately NOT gated here -- see the note
in the test below, it's a real physics point, not a shortcut.

Run: python tests/test_tw_milestone_b.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.tw_solver import TWTrappedGPESolver
from gpe3d import evolve, gates


def test_single_noisy_trajectory_conserves_norm_and_energy():
    solver = TWTrappedGPESolver(N=32, L=12.0, g=1.0, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0, imag_time_steps=1200,
                                    seed=0, cutoff_multiplier=2.0))
    print(f"noise cutoff included {solver.n_cutoff_modes}/{solver.engine.N**3} modes")

    dt = solver.engine.dt
    n_blocks, steps_per_block = 8, max(1, round(2.0 / dt / 8))
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block)

    # check_momentum=False: vacuum noise is a random, generically NOT
    # centrosymmetric perturbation of the trapped ground state -- so
    # momentum/angular momentum are NOT expected to stay ~0 for a SINGLE
    # noisy trajectory (Ehrenfest's theorem: an off-center noisy blob in a
    # harmonic trap genuinely sloshes, same mechanism as Kohn-mode dipole
    # oscillation from a deliberate offset). <P>=0 only holds in the
    # ENSEMBLE average over many independent noise draws (Milestone D),
    # not per-trajectory. This mirrors run.py's own check_momentum=False
    # for a displaced/kicked ClassicalTrappedGPESolver run -- same
    # underlying reason (broken centrosymmetry), different source.
    results = gates.run_gates(history, tol=0.05, check_momentum=False)
    print(gates.report(results))
    assert all(r.passed for r in results), gates.report(results)


def test_noisy_trajectory_reproducible_with_fixed_seed():
    """Same seed -> bit-identical trajectory (RANDOM_SEED reproducibility,
    per README.md's risk list: "a failing gate can be
    reproduced exactly while debugging, rather than chasing a
    non-reproducible statistical fluctuation")."""
    from gpe3d.backend import to_numpy

    def run_once():
        solver = TWTrappedGPESolver(N=24, L=10.0, g=1.0, omega=(1.0, 1.0, 1.0))
        state = solver.init_state(dict(n_particles=1.0, imag_time_steps=600, seed=42))
        state = solver.call(state, 20)
        return to_numpy(solver.engine._density(state.psi))

    d1, d2 = run_once(), run_once()
    max_diff = float(abs(d1 - d2).max())
    assert max_diff < 1e-9, f"same seed produced different trajectories, max_diff={max_diff:.3e}"
    print(f"PASS: identical seed reproduces identical trajectory (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    test_single_noisy_trajectory_conserves_norm_and_energy()
    test_noisy_trajectory_reproducible_with_fixed_seed()
    print("Milestone B: single noisy trajectory validated.")
