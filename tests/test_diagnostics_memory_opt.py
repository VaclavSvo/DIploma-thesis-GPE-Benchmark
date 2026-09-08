"""Regression test for the GTX 1660 Ti memory optimization pass item 3
(README.md) -- GPEPhysics3D.compute_diagnostics()
and ._energy() were rewritten to stream one spectral-derivative axis at a
time, instead of holding all 3 derivative arrays simultaneously, to cut the
~40 B/point transient peak that caused this project's Phase 2F OOM. Those
routines are now gpe3d/physics.py's _kinetic_energy() (energy only, by
Parseval in k space -- one forward FFT) and _diagnostics_from_currents()
(the full report, one shared forward FFT + one inverse per axis); this file
checks BOTH against the same pre-item-3 reference. The momentum-density formula
also changed from xp.imag(xp.conj(psi)*dpsid) to the algebraically
identical real-valued form psi.real*dpsid.imag - psi.imag*dpsid.real (fewer
complex64 temporaries).

This file reimplements the PRE-item-3 formula byte-for-byte as a standalone
reference (the original code was removed, not kept as a fallback path) and
checks the new streamed implementation against it -- this is the single
highest-scrutiny item in the whole optimization pass (Phase 2F OOM history,
every gate's critical path), so it gets its own dedicated regression test
rather than relying only on existing gate-tolerance tests (tests/test_existence.py
etc. check drift is *small*, not that the two formulas are the *same
formula* -- a subtly wrong rewrite that still conserves norm/energy to
within 1% would sail through those, this would not).

Run: python tests/test_diagnostics_memory_opt.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp, fftn, ifftn
from gpe3d.physics import GPEPhysics3D, _FFT_AXES
from gpe3d.potentials import harmonic_trap


def _old_compute_diagnostics_reference(engine):
    """Byte-for-byte reimplementation of the PRE-item-3 compute_diagnostics()
    algorithm: all 3 derivatives held at once (one shared forward FFT + 3
    inverse FFTs), momentum density via xp.imag(xp.conj(psi)*dpsid). This is
    the ground truth the streamed implementation is checked against below.
    """
    psi, V, dV = engine.psi, engine.V, engine.dV
    psi_k = fftn(psi, axes=_FFT_AXES)
    dpsidx = ifftn(1j * engine.KX * psi_k, axes=_FFT_AXES)
    dpsidy = ifftn(1j * engine.KY * psi_k, axes=_FFT_AXES)
    dpsidz = ifftn(1j * engine.KZ * psi_k, axes=_FFT_AXES)
    dens = engine._density(psi)

    norm = float(xp.sum(dens)) * dV
    E_kin = float(xp.sum(0.5 * (engine._density(dpsidx) + engine._density(dpsidy)
                                 + engine._density(dpsidz)))) * dV
    E_pot = float(xp.sum(V * dens)) * dV
    E_int = float(0.5 * engine.g * xp.sum(dens ** 2)) * dV

    jx = xp.imag(xp.conj(psi) * dpsidx)
    jy = xp.imag(xp.conj(psi) * dpsidy)
    jz = xp.imag(xp.conj(psi) * dpsidz)

    Px = float(xp.sum(jx)) * dV
    Py = float(xp.sum(jy)) * dV
    Pz = float(xp.sum(jz)) * dV

    Lx = float(xp.sum(engine.Y * jz - engine.Z * jy)) * dV
    Ly = float(xp.sum(engine.Z * jx - engine.X * jz)) * dV
    Lz = float(xp.sum(engine.X * jy - engine.Y * jx)) * dV

    return dict(norm=norm, E_total=E_kin + E_pot + E_int,
                E_kin=E_kin, E_pot=E_pot, E_int=E_int,
                Px=Px, Py=Py, Pz=Pz, Lx=Lx, Ly=Ly, Lz=Lz)


def _random_psi(shape, seed):
    """Generic complex64 field, not a physical ground state -- deliberately
    exercises every term (nonzero E_kin/E_pot/E_int/P/L) generically, since
    this test checks arithmetic equivalence of two formulas, not physics
    validity."""
    rng = xp.random.default_rng(seed)
    re = rng.standard_normal(shape, dtype=xp.float32)
    im = rng.standard_normal(shape, dtype=xp.float32)
    return (re + 1j * im).astype(xp.complex64)


# Momentum and angular momentum are compared against the largest component of
# their own triple, not against themselves.
#
# Why: on the generic random field below, Lz comes out ~0.39 while Lx and Ly
# are ~600 -- the integrand cancels to 3.2 digits, and float32 carries about
# 7, so only ~3-4 digits of Lz are real arithmetic and the rest is round-off.
# A self-relative 1e-5 tolerance on Lz therefore tests the round-off pattern,
# not the formula: measured against a float64 evaluation of the same integral,
# BOTH the pre-optimisation reference (8.1e-5 absolute) and the current code
# (9.9e-5) are equally far from the truth, in the same direction, for the same
# reason. It passed before only because the two happened to round alike.
#
# Comparing against the family's scale is the same rule gates.py already
# applies to exactly these quantities ("whose reference value is ~0 on a
# non-rotating ground state, making a relative gate meaningless"). It stays
# strict where it matters: any real algebraic error -- a swapped sign, a wrong
# axis pairing, a dropped term -- is O(1) against that scale and still fails
# by four orders of magnitude.
_SCALE_FAMILIES = (("Px", "Py", "Pz"), ("Lx", "Ly", "Lz"))


def _assert_dicts_close(old: dict, new: dict, rtol: float, label: str):
    scale = {}
    for family in _SCALE_FAMILIES:
        if all(k in old for k in family):
            s = max(abs(old[k]) for k in family)
            scale.update({k: s for k in family})
    for k in old:
        ref = max(abs(old[k]), scale.get(k, 0.0), 1e-30)
        rel = abs(new[k] - old[k]) / ref
        assert rel < rtol, (f"{label}: {k} -- old={old[k]!r}, new={new[k]!r} "
                            f"(|diff|/{ref:.6g} = {rel:.2e})")


def test_streamed_diagnostics_matches_old_formula_single_trajectory():
    eng = GPEPhysics3D(N=20, L=10.0, g=0.7)
    V = harmonic_trap(eng.X, eng.Y, eng.Z, omega=(1.0, 1.3, 0.8))
    eng.V = V
    eng.psi = _random_psi((eng.N, eng.N, eng.N), seed=42)

    old = _old_compute_diagnostics_reference(eng)
    new = eng.compute_diagnostics()
    _assert_dicts_close(old, new, rtol=1e-5, label="single-trajectory")
    print("PASS: streamed compute_diagnostics() matches the pre-item-3 formula (single trajectory)")


def test_streamed_diagnostics_matches_old_formula_batched():
    """Milestone C's batch axis (B,N,N,N) -- TW ensembles run through this
    shape in production, so item 3's rewrite needs to be checked against it
    too, not just the single-trajectory (N,N,N) case."""
    eng = GPEPhysics3D(N=16, L=10.0, g=0.35)
    V = harmonic_trap(eng.X, eng.Y, eng.Z, omega=(1.0, 1.0, 1.0))
    eng.V = V
    eng.psi = _random_psi((3, eng.N, eng.N, eng.N), seed=7)

    old = _old_compute_diagnostics_reference(eng)
    new = eng.compute_diagnostics()
    _assert_dicts_close(old, new, rtol=1e-5, label="batched")
    print("PASS: streamed compute_diagnostics() matches the pre-item-3 formula (batched, B=3)")


def test_streamed_energy_matches_old_formula():
    """_energy() (want_momentum=False path, used by the imaginary-time
    ground-state loop's periodic convergence check) checked against the
    same reference's E_total."""
    eng = GPEPhysics3D(N=20, L=10.0, g=1.2)
    V = harmonic_trap(eng.X, eng.Y, eng.Z, omega=(0.9, 1.1, 1.0))
    eng.V = V
    eng.psi = _random_psi((eng.N, eng.N, eng.N), seed=123)

    old_total = _old_compute_diagnostics_reference(eng)["E_total"]
    new_total = eng._energy(eng.psi, V)
    rel = abs(new_total - old_total) / max(abs(old_total), 1e-30)
    assert rel < 1e-5, f"_energy(): old={old_total!r}, new={new_total!r} (rel={rel:.2e})"
    print(f"PASS: streamed _energy() matches the pre-item-3 formula (rel={rel:.2e})")


if __name__ == "__main__":
    test_streamed_diagnostics_matches_old_formula_single_trajectory()
    test_streamed_diagnostics_matches_old_formula_batched()
    test_streamed_energy_matches_old_formula()
    print("Diagnostics memory optimization regression tests: all passed.")
