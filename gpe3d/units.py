"""Real SI -> natural units (hbar=m=1), shared by every physically-scaled
config.py scenario, so each scenario block states only its own species, trap
and velocity numbers. Pure math -- no backend dependency, since config.py runs
before any engine exists.
"""
from dataclasses import dataclass
import math

HBAR = 1.054571817e-34             # J*s
U_ATOMIC_MASS = 1.66053906660e-27  # kg per atomic mass unit
K_BOLTZMANN = 1.380649e-23         # J/K
ZETA_3 = 1.2020569031595943        # Riemann zeta(3), ideal-Bose-gas Tc formula

# Reference species (Na-23) for every physically-scaled scenario.
# The scattering length came from an automated extraction of the source paper,
# not a hand check of its equations -- best effort, worth verifying.
NA23_ATOM_MASS_U = 22.9897693
NA23_SCATTERING_LENGTH_M = 2.75e-9  # m, s-wave -- VERIFY vs. source paper


@dataclass(frozen=True)
class NaturalUnits:
    """A real dilute-gas experiment expressed in natural units (hbar=m=1)."""
    atom_mass_kg: float
    omega_ref_rad_s: float
    l_phys_m: float   # length unit (meters) -- harmonic osc. length at omega_ref
    t_phys_s: float   # time unit (seconds)
    g: float          # dimensionless coupling, g = 4*pi*a/L_phys


def derive_natural_units(atomic_mass_u: float, scattering_length_m: float,
                          omega_ref_hz: float) -> NaturalUnits:
    """L_phys is the harmonic oscillator length at omega_ref, and the
    dimensionless coupling is g = 4*pi*a/L_phys."""
    atom_mass_kg = atomic_mass_u * U_ATOMIC_MASS
    omega_ref_rad_s = 2.0 * math.pi * omega_ref_hz
    l_phys_m = math.sqrt(HBAR / (atom_mass_kg * omega_ref_rad_s))
    t_phys_s = 1.0 / omega_ref_rad_s
    g = 4.0 * math.pi * (scattering_length_m / l_phys_m)
    return NaturalUnits(atom_mass_kg=atom_mass_kg, omega_ref_rad_s=omega_ref_rad_s,
                         l_phys_m=l_phys_m, t_phys_s=t_phys_s, g=g)


def velocity_to_natural(v_m_s: float, units: NaturalUnits) -> float:
    """v_natural = v_real / (L_phys * omega_ref)."""
    return v_m_s / (units.l_phys_m * units.omega_ref_rad_s)


def kelvin_to_natural_temperature(T_kelvin: float, units: NaturalUnits) -> float:
    """T* = k_B*T / (hbar*omega_ref) -- the same energy unit every other
    natural-units energy here uses, so T* plugs straight into
    noise.mode_variance() alongside its E_k = 0.5*K_sq with no further
    conversion. noise.py never imports this module or knows omega_ref exists.
    """
    return (K_BOLTZMANN * T_kelvin) / (HBAR * units.omega_ref_rad_s)


def natural_temperature_to_kelvin(T_natural: float, units: NaturalUnits) -> float:
    """Inverse of kelvin_to_natural_temperature, for logging (typically nK)."""
    return T_natural * HBAR * units.omega_ref_rad_s / K_BOLTZMANN


def condensate_critical_temperature_natural(omega_natural: tuple, n_particles: float) -> float:
    """Ideal-Bose-gas condensation temperature for N atoms in a 3D harmonic
    trap, in the SAME natural-units temperature scale as
    kelvin_to_natural_temperature (k_B*T/(hbar*omega_ref)):

        k_B*T_c = hbar*omega_bar * (N/zeta(3))**(1/3),  omega_bar = (wx*wy*wz)**(1/3)

    (Dalfovo, Giorgini, Pitaevskii & Stringari, RMP 71, 463 (1999), Eq. 2.)
    The interacting shift is a further few-percent correction, deliberately
    not applied: this is a reference scale for picking a "low but nonzero"
    T/T_c, not a precision prediction. omega_natural is config.py's OMEGA
    tuple; T_c comes out in the same units as kelvin_to_natural_temperature,
    so a chosen T* compares against it directly.
    """
    wx, wy, wz = omega_natural
    omega_bar_natural = (wx * wy * wz) ** (1.0 / 3.0)
    return omega_bar_natural * (n_particles / ZETA_3) ** (1.0 / 3.0)
