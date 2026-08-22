"""External potentials. Harmonic trap only, so far."""
from .backend import xp


def harmonic_trap(X, Y, Z, omega=(1.0, 1.0, 1.0)):
    """V(r) = 0.5*(wx^2 x^2 + wy^2 y^2 + wz^2 z^2), natural units (hbar=m=1).

    omega=(1,1,1) makes the noninteracting ground state an isotropic Gaussian
    of per-axis width sigma = 1/sqrt(2*omega) with E0 = 1.5*omega -- the
    analytic cross-check in tests/test_existence.py.
    """
    wx, wy, wz = omega
    V = 0.5 * (wx**2 * X**2 + wy**2 * Y**2 + wz**2 * Z**2)
    return V.astype(xp.float32)
