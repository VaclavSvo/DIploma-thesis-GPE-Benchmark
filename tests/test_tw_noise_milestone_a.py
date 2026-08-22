"""Milestone A (README.md): validate the TW vacuum-noise
sampler in isolation, no dynamics at all -- isolates sampler bugs from
propagator/batching bugs before they can compound (Milestone B/C build on
top of this). Two checks:

  1. k-space: per-mode ensemble variance matches 0.5 inside the energy
     cutoff, exactly 0 outside.
  2. real-space: <|delta_psi(r)|^2>, averaged over an ensemble, matches
     the analytic prediction N_cutoff_modes/(2*V) -- the same quantity
     Milestone D's density ordering-correction gate will later subtract.

Run: python tests/test_tw_noise_milestone_a.py
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp, to_numpy
from gpe3d.physics import GPEPhysics3D
from gpe3d.noise import energy_cutoff_mask, mode_variance, sample_alpha, sample_noise
from gpe3d import gates


def test_kspace_variance_matches_target():
    N, L, g, n_peak = 24, 10.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    mask = energy_cutoff_mask(eng.K_sq, g, n_peak, cutoff_multiplier=2.0)
    n_incl = int(to_numpy(mask).sum())
    # Want a comfortable number of modes for the ensemble-variance statistic
    # below, not just "nonzero" -- a mask that admits only the k=0 mode
    # would technically satisfy 0 < n_incl < N**3 but give a useless,
    # single-sample "ensemble" check.
    assert n_incl >= 20, (
        f"cutoff mask too tight for a meaningful variance check: only {n_incl}/{N**3} "
        f"modes included -- raise n_peak or cutoff_multiplier")

    n_traj = 4000
    rng = xp.random.default_rng(0)
    alpha = sample_alpha(eng.K_sq.shape, mode_variance(eng.K_sq, mask), n_traj, rng)
    mean_abs2 = to_numpy(xp.mean(xp.abs(alpha) ** 2, axis=0))
    mask_np = to_numpy(mask)

    incl = mean_abs2[mask_np]
    excl = mean_abs2[~mask_np]

    stat_err = 1.0 / math.sqrt(n_traj)  # order-1/sqrt(n) relative std error on a variance-0.5 estimate
    rel_dev = abs(incl.mean() - 0.5) / 0.5
    assert rel_dev < 5 * stat_err, (
        f"in-cutoff <|alpha_k|^2>={incl.mean():.4f}, expected 0.5, rel_dev={rel_dev:.3%} "
        f"(5x stat_err={5*stat_err:.3%})")
    assert excl.max() == 0.0, f"modes outside cutoff should be exactly zero, got max={excl.max()}"
    print(f"PASS: k-space variance -- in-cutoff mean={incl.mean():.4f} (target 0.500, "
          f"{n_incl}/{N**3} modes included), out-of-cutoff max={excl.max():.2e}")


def test_realspace_density_offset_matches_prediction():
    N, L, g, n_peak = 24, 10.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    mask = energy_cutoff_mask(eng.K_sq, g, n_peak, cutoff_multiplier=2.0)
    n_incl = int(to_numpy(mask).sum())
    V = (N ** 3) * eng.dV

    n_traj = 300
    rng = xp.random.default_rng(1)
    delta_psi = sample_noise(eng.K_sq, eng.dV, mask, n_traj, rng)
    dens = to_numpy(xp.mean(xp.abs(delta_psi) ** 2, axis=0))

    predicted = n_incl / (2.0 * V)
    numeric = float(dens.mean())
    rel_err = abs(numeric - predicted) / predicted
    # Looser tolerance than the k-space check: this is a full real-space
    # ensemble mean over a modest n_traj, not a tight per-mode statistic.
    assert rel_err < 0.15, (
        f"<|delta_psi(r)|^2>={numeric:.4e} vs predicted {predicted:.4e} (rel_err={rel_err:.2%})")
    print(f"PASS: real-space density offset -- numeric={numeric:.4e} predicted={predicted:.4e} "
          f"(N_cutoff={n_incl}/{N**3} modes, rel_err={rel_err:.2%})")


def test_full_cutoff_matches_1_over_2dV():
    """Sanity check on the normalization itself: with the cutoff opened to
    include every grid mode, <|delta_psi(r)|^2> must equal 1/(2*dV) exactly
    (N_cutoff=N^3 => N^3/(2*V) = 1/(2*dx^3) = 1/(2*dV)) -- this is the
    exact quantity README.md's ordering-correction
    (n_phys = <|psi_W|^2> - 1/(2*dV)) is built around.
    """
    N, L = 16, 8.0
    eng = GPEPhysics3D(N=N, L=L, g=1.0)
    mask_all = xp.ones_like(eng.K_sq, dtype=bool)

    n_traj = 600
    rng = xp.random.default_rng(2)
    delta_psi = sample_noise(eng.K_sq, eng.dV, mask_all, n_traj, rng)
    dens = to_numpy(xp.mean(xp.abs(delta_psi) ** 2, axis=0))

    target = 1.0 / (2.0 * eng.dV)
    numeric = float(dens.mean())
    rel_err = abs(numeric - target) / target
    assert rel_err < 0.15, f"full-cutoff <|delta_psi|^2>={numeric:.4e} vs 1/(2dV)={target:.4e}"
    print(f"PASS: full-cutoff offset -- numeric={numeric:.4e} target(1/2dV)={target:.4e} "
          f"(rel_err={rel_err:.2%})")


def test_cutoff_wired_into_resolution_gate():
    """README.md flagged gates.resolution_gate
    (built for the velocity-kick-aliasing bug) as reusable for the TW noise
    cutoff too -- this demonstrates that wiring and checks it actually
    discriminates: a sane cutoff passes with margin, a cutoff pushed close
    to the grid's Nyquist limit fails. E_cut -> k_char via E=0.5*k^2 =>
    k_char=sqrt(2*E_cut), the same "velocity/wavenumber" identification
    resolution_gate already uses (hbar=m=1).
    """
    import math
    N, L, g, n_peak = 24, 10.0, 1.0, 0.5
    eng = GPEPhysics3D(N=N, L=L, g=g)
    k_nyquist = math.pi / eng.dx

    E_cut_sane = 2.0 * g * n_peak
    res_sane = gates.resolution_gate("tw_noise_cutoff", math.sqrt(2 * E_cut_sane), k_nyquist)
    assert res_sane.passed, f"expected sane cutoff to pass: {res_sane.detail}"

    E_cut_aggressive = 0.9 * (0.5 * k_nyquist ** 2)  # cutoff energy right at ~Nyquist
    res_aggressive = gates.resolution_gate("tw_noise_cutoff", math.sqrt(2 * E_cut_aggressive), k_nyquist)
    assert not res_aggressive.passed, f"expected near-Nyquist cutoff to fail: {res_aggressive.detail}"

    print(f"PASS: resolution_gate discriminates -- sane cutoff: {res_sane.detail}")
    print(f"                                     aggressive cutoff: {res_aggressive.detail}")


if __name__ == "__main__":
    test_kspace_variance_matches_target()
    test_realspace_density_offset_matches_prediction()
    test_full_cutoff_matches_1_over_2dV()
    test_cutoff_wired_into_resolution_gate()
    print("Milestone A: noise sampler validated in isolation.")
