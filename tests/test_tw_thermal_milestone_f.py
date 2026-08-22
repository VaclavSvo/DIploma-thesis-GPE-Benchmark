"""Finite, low-temperature Truncated-Wigner noise (thermal + vacuum) -- see
gpe3d/noise.py for the physics.

Same pattern as the other TW tests: isolate the sampler math first (no
dynamics), then check it survives contact with the real propagator, gated the
same way.

Run: python tests/test_tw_thermal_milestone_f.py
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp, to_numpy
from gpe3d.physics import GPEPhysics3D
from gpe3d.noise import (energy_cutoff_mask, mode_variance, sample_alpha,
                          sample_noise, offset_density)
from gpe3d.units import (derive_natural_units, kelvin_to_natural_temperature,
                          natural_temperature_to_kelvin,
                          condensate_critical_temperature_natural,
                          NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M)
from gpe3d.tw_solver import TWTrappedGPESolver
from gpe3d import gates


def test_thermal_variance_matches_flat_half_at_T0():
    N, L, g, n_peak = 24, 10.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    mask = energy_cutoff_mask(eng.K_sq, g, n_peak, cutoff_multiplier=2.0)

    var_T0 = to_numpy(mode_variance(eng.K_sq, mask, T=0.0))
    expected = to_numpy(mask).astype(float) * 0.5
    max_diff = float(abs(var_T0 - expected).max())
    assert max_diff == 0.0, f"T=0 mode_variance != flat 0.5*mask, max_diff={max_diff:.3e}"
    print("PASS: mode_variance(T=0) == flat 0.5*mask exactly")


def test_zero_mode_never_thermally_enhanced():
    """k=0 must stay exactly 0.5 at any T>0 -- it's the condensate's own
    mode (see mode_variance's docstring), and n_k(T) formally
    diverges as E_k->0, so this is both a physics necessity and a
    numerical-finiteness requirement."""
    N, L, g, n_peak = 16, 8.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    mask = xp.ones_like(eng.K_sq, dtype=bool)  # admit every mode, including k=0

    for T in (0.01, 1.0, 100.0):
        var = to_numpy(mode_variance(eng.K_sq, mask, T=T))
        k_sq_np = to_numpy(eng.K_sq)
        zero_idx = k_sq_np == 0
        assert zero_idx.sum() == 1, "expected exactly one k=0 mode on this grid"
        assert var[zero_idx][0] == 0.5, f"k=0 mode variance={var[zero_idx][0]} != 0.5 at T={T}"
        assert xp.isfinite(xp.asarray(var)).all(), f"non-finite variance at T={T}"
    print("PASS: k=0 mode stays exactly 0.5 (vacuum-only) at every T tested, all other modes finite")


def test_thermal_variance_limits():
    """Analytic sanity check on the coth formula itself, independent of any
    grid/sampler code: high-E (E_k >> T) -> vacuum-dominated (~0.5); low-E,
    nonzero (E_k << T) -> classical/Rayleigh-Jeans equipartition,
    n_k+0.5 ~ T/E_k (coth(x) ~ 1/x as x->0)."""
    T = 5.0
    E_high = 50.0 * T   # deep vacuum regime
    E_low = 1.0e-3 * T  # deep classical/thermal regime

    var_high = 0.5 / math.tanh(E_high / (2.0 * T))
    var_low = 0.5 / math.tanh(E_low / (2.0 * T))

    assert abs(var_high - 0.5) < 1e-3, f"high-E limit should -> 0.5, got {var_high:.6f}"
    rj_predicted = T / E_low
    rel_err = abs(var_low - rj_predicted) / rj_predicted
    assert rel_err < 1e-3, (f"low-E limit should match Rayleigh-Jeans T/E_k={rj_predicted:.3f}, "
                            f"got {var_low:.3f} (rel_err={rel_err:.3%})")
    print(f"PASS: coth formula limits -- high-E: {var_high:.6f} (-> 0.5), "
          f"low-E: {var_low:.3f} vs Rayleigh-Jeans {rj_predicted:.3f} (rel_err={rel_err:.3%})")


def test_kspace_variance_matches_thermal_target():
    """Ensemble check (Milestone A's style, generalized to T>0): per-mode
    <|alpha_k|^2> should match mode_variance's prediction for a
    representative low-E and high-E mode."""
    N, L, g, n_peak = 24, 10.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    mask = xp.ones_like(eng.K_sq, dtype=bool)
    T = 3.0

    n_traj = 6000
    rng = xp.random.default_rng(11)
    variance = mode_variance(eng.K_sq, mask, T=T)
    alpha = sample_alpha(eng.K_sq.shape, variance, n_traj, rng)
    mean_abs2 = to_numpy(xp.mean(xp.abs(alpha) ** 2, axis=0))
    variance_np = to_numpy(variance)

    # Pick a handful of modes spanning low-to-high E_k (excluding k=0, which
    # is deliberately NOT thermally enhanced -- see test above) and check
    # each individually against its own predicted variance.
    k_sq_np = to_numpy(eng.K_sq)
    flat_idx = k_sq_np.reshape(-1).argsort()
    candidates = [flat_idx[i] for i in (1, N**3 // 8, N**3 // 2, -1)]  # skip idx 0 (k=0)
    stat_err = 1.0 / math.sqrt(n_traj)
    for idx in candidates:
        pos = tuple(int(c) for c in xp.unravel_index(idx, k_sq_np.shape))
        predicted = float(variance_np[pos])
        measured = float(mean_abs2[pos])
        rel_dev = abs(measured - predicted) / predicted
        assert rel_dev < max(10 * stat_err, 0.15), (
            f"mode {pos}: E_k={0.5*k_sq_np[pos]:.4g}, predicted variance={predicted:.4f}, "
            f"measured={measured:.4f} (rel_dev={rel_dev:.2%})")
    print(f"PASS: thermal k-space variance matches prediction at T={T} across low-to-high-E modes")


def test_units_temperature_roundtrip_and_Tc():
    units = derive_natural_units(NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M, omega_ref_hz=200.0)
    T_kelvin = 100e-9  # 100 nK
    T_nat = kelvin_to_natural_temperature(T_kelvin, units)
    T_back = natural_temperature_to_kelvin(T_nat, units)
    rel_err = abs(T_back - T_kelvin) / T_kelvin
    assert rel_err < 1e-12, f"temperature round-trip failed: {T_kelvin} -> {T_nat} -> {T_back}"

    Tc_nat = condensate_critical_temperature_natural((1.0, 1.0, 1.0), n_particles=1.0e4)
    Tc_kelvin = natural_temperature_to_kelvin(Tc_nat, units)
    # Sanity bounds only (order-of-magnitude, not a precision claim): for a
    # ~200 Hz isotropic trap and N=1e4, Tc should land in the tens-to-few-
    # hundred nK range typical of alkali BEC experiments, not nK/1000 or uK*1000.
    assert 1e-9 < Tc_kelvin < 1e-6, f"Tc={Tc_kelvin:.3e} K outside sane BEC range"
    print(f"PASS: T round-trip exact; Tc(N=1e4, 200Hz isotropic)={Tc_kelvin*1e9:.1f} nK "
          f"(T*={Tc_nat:.3f})")


def test_solver_thermal_depletion_increases_with_temperature():
    """Contact with the real solver/propagator (Milestone B/D's style):
    finite T should (a) still pass norm/energy gates (T only changes the
    INITIAL noise, not the propagator -- same guarantee as Milestone B),
    and (b) increase depletion_fraction monotonically with T, since thermal
    occupation only adds population on top of the same vacuum floor."""
    G_REAL = 0.003523257353164062  # same real Na-23 G as test_tw_milestone_d.py

    depletions = []
    for T_nat in (0.0, 0.5, 2.0):
        solver = TWTrappedGPESolver(N=28, L=10.0, g=G_REAL, omega=(1.0, 1.0, 1.0))
        state = solver.init_state(dict(n_particles=1.0e4, imag_time_steps=600,
                                        seed=0, cutoff_multiplier=2.0,
                                        temperature_natural=T_nat))
        diag = solver.physical_particle_number(state)
        depletions.append(diag["depletion_fraction"])
        print(f"  T*={T_nat}: depletion_fraction={diag['depletion_fraction']:.2%}, "
              f"n_phys={diag['n_phys']:.1f} (target {solver.n_particles_target:.0f})")

    assert depletions[0] < depletions[1] < depletions[2], (
        f"depletion should strictly increase with temperature, got {depletions}")
    print(f"PASS: depletion_fraction increases monotonically with T*: {depletions}")


def test_solver_thermal_gates_pass_at_modest_T():
    """T only perturbs the INITIAL state (same guarantee Milestone B
    established for vacuum noise) -- norm/energy conservation must still
    hold under the unmodified propagator at a modest, TWA-safe temperature."""
    from gpe3d import evolve
    G_REAL = 0.003523257353164062
    solver = TWTrappedGPESolver(N=28, L=10.0, g=G_REAL, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0e4, imag_time_steps=600, seed=0,
                                    cutoff_multiplier=2.0, temperature_natural=0.5,
                                    order4_dt_multiplier=25.0))
    n_blocks, steps_per_block = 6, max(1, round(1.0 / solver.engine.dt / 6))
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block)
    results = gates.run_gates(history, tol=0.05, check_momentum=False)
    print(gates.report(results))
    assert all(r.passed for r in results), gates.report(results)


if __name__ == "__main__":
    test_thermal_variance_matches_flat_half_at_T0()
    test_zero_mode_never_thermally_enhanced()
    test_thermal_variance_limits()
    test_kspace_variance_matches_thermal_target()
    test_units_temperature_roundtrip_and_Tc()
    test_solver_thermal_depletion_increases_with_temperature()
    test_solver_thermal_gates_pass_at_modest_T()
    print("Milestone F: finite low-temperature TW noise validated.")
