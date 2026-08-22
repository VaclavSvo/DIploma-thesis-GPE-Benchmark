"""Milestone D, first slice (README.md): the ordering-
correction gate ("the single most important new gate in phase 2") plus a
depletion sanity check, using the CORRECTED offset formula (see
gpe3d/noise.py's vacuum_offset_density docstring -- the plan doc's flat
1/(2*dV) is only exact when the cutoff spans every grid mode).

This also nails down a real parameter-choice bug found while building
this: the phase-1 dimensionless test convention (g=1, n_particles=1,
borrowed from tests/test_existence.py, which only needed to validate the
propagator, not real physics) puts TW noise's own "particle count" at
5-1000x the condensate's -- wildly outside TWA's validity regime, at
ANY n_particles tried up to 1e6 (see README.md).
The fix isn't the noise sampler or the correction formula (both already
validated in Milestone A to <0.1%) -- it's that g=1 is not a physical
dilute-gas coupling. Using the SAME real-physics derivation config.py's
"collision" scenario already uses (Na-23 scattering length -> G~3.5e-3,
not 1) with n_particles~1e4 (matching config.py's laptop-scaled
N_PER_CONDENSATE) gives a small, physical depletion fraction instead.

Run: python tests/test_tw_milestone_d.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.tw_solver import TWTrappedGPESolver
from gpe3d import evolve, gates

# Same real-physics G derivation as config.py's "collision" scenario
# (Na-23, arXiv:cond-mat/0602061's reference trap frequency) -- see
# README.md for the numeric
# derivation. NOT g=1 -- see module docstring for why that matters here.
_G_REAL = 0.003523257353164062


def test_unphysical_g1_convention_flags_large_depletion():
    """Negative control: the OLD g=1/n_particles=1 test convention should
    correctly report itself as unphysical (large depletion_fraction), not
    silently look fine -- confirms the diagnostic actually discriminates,
    same pattern as Milestone A's resolution_gate discrimination test."""
    solver = TWTrappedGPESolver(N=32, L=12.0, g=1.0, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0, imag_time_steps=800,
                                    seed=0, cutoff_multiplier=2.0))
    diag = solver.physical_particle_number(state)
    print(f"g=1, n_particles=1: depletion_fraction={diag['depletion_fraction']:.1%} "
          f"(n_vacuum_offset={diag['n_vacuum_offset']:.2f} vs n_particles_target=1)")
    assert diag["depletion_fraction"] > 1.0, (
        "expected the old g=1 convention to clearly fail the depletion sanity check")


def test_physical_parameters_give_small_depletion():
    """Positive case: real dilute-gas G + realistic n_particles -> small
    (percent-level) depletion, inside TWA's validity regime."""
    solver = TWTrappedGPESolver(N=32, L=12.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0e4, imag_time_steps=800,
                                    seed=0, cutoff_multiplier=2.0))
    diag = solver.physical_particle_number(state)
    print(f"G={_G_REAL:.4e}, n_particles=1e4: depletion_fraction={diag['depletion_fraction']:.2%}, "
          f"n_cutoff_modes={solver.n_cutoff_modes}/{solver.engine.N**3}, "
          f"n_phys={diag['n_phys']:.1f} (target {solver.n_particles_target:.0f})")
    assert diag["depletion_fraction"] < 0.10, (
        f"expected small depletion at physical parameters, got {diag['depletion_fraction']:.2%}")
    # n_phys should recover close to the mean-field target -- loose bound,
    # this is a single trajectory (n_traj=1), not yet an ensemble average,
    # so real trajectory-to-trajectory scatter is expected on top of the
    # exact-in-ensemble-mean identity derived in noise.py's docstring.
    rel_err = abs(diag["n_phys"] - solver.n_particles_target) / solver.n_particles_target
    assert rel_err < 0.05, f"n_phys={diag['n_phys']:.1f} vs target={solver.n_particles_target:.0f}"


def test_ordering_correction_matches_across_ensemble():
    """n_traj>1 (Milestone C's batch axis): ensemble-mean n_phys should
    track n_particles_target more tightly than a single trajectory does
    (statistical averaging), not just individually pass a loose bound."""
    solver = TWTrappedGPESolver(N=24, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0e4, imag_time_steps=500,
                                    seed=0, cutoff_multiplier=2.0, n_traj=8))
    diag = solver.physical_particle_number(state)
    rel_err = abs(diag["n_phys"] - solver.n_particles_target) / solver.n_particles_target
    print(f"n_traj=8 ensemble: n_phys={diag['n_phys']:.1f} (target {solver.n_particles_target:.0f}, "
          f"rel_err={rel_err:.3%}), depletion_fraction={diag['depletion_fraction']:.2%}")
    assert rel_err < 0.03, f"ensemble n_phys off target by {rel_err:.3%}"


def test_order4_tw_multiplier_passes_at_physical_parameters():
    """The TW-specific order=4 dt multiplier (gpe3d/tw_solver.py's
    _TW_ORDER4_DT_MULTIPLIER_DEFAULT=25, picked at g=1 to survive the
    worst-case-density noise regime) should also pass comfortably at the
    real, much-weaker G used above -- checked explicitly rather than
    assumed, since a weaker g changes the nonlinear-kick dt heuristic
    too."""
    solver = TWTrappedGPESolver(N=32, L=12.0, g=_G_REAL, omega=(1.0, 1.0, 1.0),
                                 splitstep_order=4)
    state = solver.init_state(dict(n_particles=1.0e4, imag_time_steps=800,
                                    seed=0, cutoff_multiplier=2.0))
    n_blocks, steps_per_block = 8, max(1, round(2.0 / solver.engine.dt / 8))
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block)
    results = gates.run_gates(history, tol=0.05, check_momentum=False)
    print(gates.report(results))
    assert all(r.passed for r in results), gates.report(results)


if __name__ == "__main__":
    test_unphysical_g1_convention_flags_large_depletion()
    test_physical_parameters_give_small_depletion()
    test_ordering_correction_matches_across_ensemble()
    test_order4_tw_multiplier_passes_at_physical_parameters()
    print("Milestone D (ordering correction + depletion sanity check): validated.")
