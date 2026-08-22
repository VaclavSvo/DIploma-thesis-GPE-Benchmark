"""Automated pass/fail gates on conserved quantities."""
from dataclasses import dataclass


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


def _rel_drift_gate(name: str, values: list[float], tol: float) -> GateResult:
    """Drift relative to the initial value -- for norm and energy, which have
    a nonzero, physically meaningful reference."""
    ref = values[0]
    drift = max(abs(v - ref) / abs(ref) for v in values) if abs(ref) > 1e-30 else float("inf")
    passed = drift < tol
    return GateResult(name, passed, f"max rel drift {drift:.3%} (tol {tol:.0%}, ref={ref:.6g})")


def _abs_drift_gate(name: str, values: list[float], scale: float, tol: float) -> GateResult:
    """Drift against a characteristic scale -- for momentum and angular
    momentum, whose reference value is ~0 on a non-rotating ground state,
    making a relative gate meaningless."""
    ref = values[0]
    drift = max(abs(v - ref) for v in values)
    passed = drift < tol * max(scale, 1e-30)
    return GateResult(name, passed,
                       f"abs drift {drift:.3e} (tol {tol * scale:.3e}, scale={scale:.3e})")


def resolution_gate(name: str, k_char: float, k_nyquist: float,
                     safety_factor: float = 2.0) -> GateResult:
    """Check the grid resolves a characteristic wavenumber with margin
    (hbar=m=1, so velocity and wavenumber are the same number).

    A velocity kick is a plane wave exp(i*v*x); if |v| exceeds the Nyquist
    wavenumber pi/dx the grid aliases it to something else -- and no
    conservation gate catches that, because aliasing is still exactly
    unitary. A collision velocity aliased to near-zero looks like two clouds
    sitting still with a perfect conservation report. Run this as a
    pre-flight check, before the expensive part.
    """
    passed = k_nyquist > safety_factor * abs(k_char)
    return GateResult(name, passed,
                       f"k_char={k_char:.3g}, k_nyquist={k_nyquist:.3g} "
                       f"(need > {safety_factor}x k_char = {safety_factor * abs(k_char):.3g}); "
                       f"{'OK' if passed else 'increase N (or decrease L) to raise k_nyquist=pi/dx'}")


def run_gates(history: dict, tol: float = 0.05, r_char: float = 1.0,
              check_momentum: bool = True) -> list[GateResult]:
    """history: dict of equal-length lists, one entry per recorded snapshot,
    with keys norm, E_total, Px, Py, Pz, Lx, Ly, Lz, E_kin (used to build the
    momentum/angular-momentum characteristic scale p_char = sqrt(2*E_kin)).
    r_char: characteristic length scale of the cloud (e.g. trap width), used
    to turn p_char into an angular-momentum scale p_char * r_char.

    check_momentum: momentum and angular momentum are conserved only for a
    resting, centred cloud, where the trap's net force and torque vanish by
    symmetry. Displace or kick it and the trap pulls on it, so momentum
    genuinely oscillates (Ehrenfest, and Kohn for the dipole mode) -- correct
    physics, not drift. Set False for any offset/kicked run. Norm and energy
    are always checked; a time-independent trap conserves both regardless.
    """
    results = [
        _rel_drift_gate("norm", history["norm"], tol),
        _rel_drift_gate("energy", history["E_total"], tol),
    ]

    if not check_momentum:
        return results

    p_char = (2.0 * max(history["E_kin"])) ** 0.5  # ~ typical momentum magnitude, E_kin ~ p^2/2
    for axis in "xyz":
        results.append(_abs_drift_gate(f"momentum_{axis}", history[f"P{axis}"], p_char, tol))
    l_char = p_char * r_char  # angular momentum ~ p_char * length scale
    for axis in "xyz":
        results.append(_abs_drift_gate(f"angular_momentum_{axis}", history[f"L{axis}"], l_char, tol))

    return results


def report(results: list[GateResult]) -> str:
    lines = [f"[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}" for r in results]
    lines.append("ALL GATES PASSED" if all(r.passed for r in results) else "GATE FAILURE")
    return "\n".join(lines)
