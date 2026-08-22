"""Milestone C (README.md): give GPEPhysics3D a leading
batch axis, evolve BATCH_SIZE trajectories at once. This is a pure
performance change (same math, same run_block, just vectorized over
trajectories) -- the gate is that batched evolution must reproduce B
independent sequential single-trajectory runs (same seeds) to
floating-point tolerance. Not a physics check (Milestones A/B already
covered that); a regression test that batching didn't silently change
the math (e.g. an axes=None default doing a full 4D FFT across the batch
axis by mistake, or a scratch buffer being shared/aliased across
trajectories when it shouldn't be).

Run: python tests/test_tw_milestone_c.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp, to_numpy
from gpe3d.tw_solver import TWTrappedGPESolver
from gpe3d.noise import sample_noise


def _sequential_reference(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                           cutoff_multiplier, n_steps, splitstep_order, dt_fixed):
    """n_traj independent TWTrappedGPESolver runs, one at a time (today's
    only code path -- Milestone B), each with a different seed. dt_fixed
    is passed explicitly (not each run's own choose_dt()) so this is a
    pure propagator/batching comparison, not also a "did two runs pick
    slightly different dt from slightly different peak densities"
    confound. Returns a list of (N,N,N) final densities."""
    out = []
    for i in range(n_traj):
        solver = TWTrappedGPESolver(N=N, L=L, g=g, omega=omega, splitstep_order=splitstep_order)
        state = solver.init_state(dict(n_particles=n_particles, imag_time_steps=imag_time_steps,
                                        seed=i, cutoff_multiplier=cutoff_multiplier, dt=dt_fixed))
        state = solver.call(state, n_steps)
        out.append(to_numpy(solver.engine._density(state.psi)))
    return out


def _batched_engine_run(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                         cutoff_multiplier, n_steps, splitstep_order, dt_fixed):
    """Same physics, but psi carries an explicit leading batch axis through
    a single GPEPhysics3D engine and one run_block() call -- exercises the
    exact code path this milestone patched (axes=_FFT_AXES on every
    fftn/ifftn call, _ensure_cpu_scratch_buffers's shape handling). Same
    dt_fixed as the sequential reference -- see its docstring."""
    from gpe3d.physics import GPEPhysics3D
    from gpe3d.potentials import harmonic_trap
    from gpe3d.noise import energy_cutoff_mask

    engine = GPEPhysics3D(N=N, L=L, g=g)
    V = harmonic_trap(engine.X, engine.Y, engine.Z, omega)

    # Same mean-field ground state every trajectory shares before noise is
    # added (matches TWTrappedGPESolver.init_state -- one imaginary-time
    # relaxation, not one per trajectory).
    psi_mean = engine.ground_state_imaginary_time(V=V, n_particles=n_particles,
                                                    n_steps=imag_time_steps)
    n_peak = float(engine._density(psi_mean).max())
    mask = energy_cutoff_mask(engine.K_sq, g, n_peak, cutoff_multiplier)

    psis = []
    for i in range(n_traj):
        rng = xp.random.default_rng(i)
        delta_psi = sample_noise(engine.K_sq, engine.dV, mask, n_traj=1, rng=rng)[0]
        psis.append((psi_mean + delta_psi).astype(xp.complex64))
    psi0_batch = xp.stack(psis, axis=0)  # (n_traj, N, N, N)

    engine.set_dt(dt_fixed, order=splitstep_order)
    engine.V = V
    engine.psi = psi0_batch
    engine.run_block(n_steps)

    dens_batch = to_numpy(engine._density(engine.psi))  # (n_traj, N, N, N)
    return [dens_batch[i] for i in range(n_traj)]


def test_batched_matches_sequential():
    n_traj, N, L, g = 4, 20, 10.0, 1.0
    omega = (1.0, 1.0, 1.0)
    n_particles, imag_time_steps, cutoff_multiplier = 1.0, 500, 2.0
    n_steps = 10
    dt_fixed = 2e-3  # small, conservative, fixed for both runs -- see docstrings above

    seq = _sequential_reference(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                                 cutoff_multiplier, n_steps, splitstep_order=2, dt_fixed=dt_fixed)
    batched = _batched_engine_run(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                                   cutoff_multiplier, n_steps, splitstep_order=2, dt_fixed=dt_fixed)

    max_rel_err = 0.0
    for i in range(n_traj):
        num = float(abs(seq[i] - batched[i]).max())
        den = float(seq[i].max())
        rel_err = num / den
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  trajectory {i}: max abs density diff={num:.3e} (rel to peak={rel_err:.3e})")

    assert max_rel_err < 1e-4, (
        f"batched run diverges from sequential single-trajectory reference: "
        f"max_rel_err={max_rel_err:.3e} (expected float32-rounding-level agreement)")
    print(f"PASS: batched matches sequential to max_rel_err={max_rel_err:.3e}")


def test_batched_matches_sequential_order4():
    """Same check, splitstep_order=4 (Yoshida) -- 3x the fftn/ifftn calls
    per outer step and the kin4_edge_full local (see physics.py's
    _run_block_4th), so worth its own regression check rather than
    assuming order=2 passing implies order=4 does too."""
    n_traj, N, L, g = 3, 20, 10.0, 1.0
    omega = (1.0, 1.0, 1.0)
    n_particles, imag_time_steps, cutoff_multiplier = 1.0, 500, 2.0
    n_steps = 6
    dt_fixed = 5e-3

    seq = _sequential_reference(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                                 cutoff_multiplier, n_steps, splitstep_order=4, dt_fixed=dt_fixed)
    batched = _batched_engine_run(n_traj, N, L, g, omega, n_particles, imag_time_steps,
                                   cutoff_multiplier, n_steps, splitstep_order=4, dt_fixed=dt_fixed)

    max_rel_err = 0.0
    for i in range(n_traj):
        num = float(abs(seq[i] - batched[i]).max())
        den = float(seq[i].max())
        rel_err = num / den
        max_rel_err = max(max_rel_err, rel_err)
        print(f"  trajectory {i}: max abs density diff={num:.3e} (rel to peak={rel_err:.3e})")

    assert max_rel_err < 1e-4, f"order=4 batched diverges from sequential: max_rel_err={max_rel_err:.3e}"
    print(f"PASS: order=4 batched matches sequential to max_rel_err={max_rel_err:.3e}")


if __name__ == "__main__":
    test_batched_matches_sequential()
    test_batched_matches_sequential_order4()
    print("Milestone C: batched evolution matches sequential single-trajectory runs.")
