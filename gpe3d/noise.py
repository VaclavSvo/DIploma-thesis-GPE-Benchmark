"""Truncated-Wigner initial-condition noise sampler.

Pure preprocessing: the propagator in gpe3d/physics.py is untouched. Every TW
trajectory obeys the same deterministic GPE; all the stochasticity is here.

Convention (Sinatra, Lobo & Castin, J. Phys. B 35, 3599 (2002); the same
grid/plane-wave method Norrie/Ballagh/Gardiner use): each k-mode below an
energy cutoff carries an independent complex Gaussian amplitude alpha_k,

    <|alpha_k|^2> = n_k(T) + 1/2 = (1/2)*coth(E_k / 2T),   E_k = 0.5*K_sq

i.e. half a quantum of vacuum fluctuation plus the Bose occupation n_k(T).
T = 0 gives the flat vacuum 1/2 and is handled exactly, not as a limit.

E_k is the bare single-particle energy, not a Bogoliubov spectrum -- consistent
with a cutoff that only ever admits modes well above the mean-field energy
scale, where the two coincide.

T is in natural units, k_B*T/(hbar*omega_ref) -- see gpe3d/units.py.
"""
import math
from .backend import xp, ifftn


def energy_cutoff_mask(K_sq, g: float, n_peak: float, cutoff_multiplier: float = 2.0):
    """Boolean mask: True where E_k = 0.5*k^2 < E_cut = cutoff_multiplier *
    g * n_peak. Ties the UV cutoff to the interaction energy scale (the
    healing-length criterion, xi = 1/sqrt(2*g*n_peak)) rather than to grid
    resolution. cutoff_multiplier is the one free literature parameter here
    (papers use O(1-few)) and is meant to be swept, not trusted blindly.
    """
    E_cut = cutoff_multiplier * abs(g) * n_peak
    if E_cut <= 0.0:
        raise ValueError(f"energy_cutoff_mask: E_cut={E_cut} <= 0 -- need g>0 and n_peak>0")
    return 0.5 * K_sq < E_cut


def scale_cutoff_multiplier_to_grid(K_sq, g: float, n_peak: float,
                                     floor_multiplier: float = 2.0,
                                     N_ref: int | None = None,
                                     growth_exponent: float = 2.0 / 3.0,
                                     safety_margin: float = 2.0):
    """Grid-resolution-scaled cutoff: as N grows past N_ref (finer grid, same
    L), admit progressively more modes instead of leaving the extra grid
    points unused above a fixed cutoff.

        E_cut = max(E_floor, min(E_target, E_ceiling))
        E_floor   = floor_multiplier * g * n_peak      (never given up)
        E_target  = E_floor * max(1, (N/N_ref)**growth_exponent)
        E_ceiling = 0.5*max(K_sq) / safety_margin**2   (hard backstop)

    N == N_ref reproduces the fixed cutoff exactly. Returns
    (effective_multiplier, info); the multiplier plugs straight into
    energy_cutoff_mask.

    The growth law is a small POWER of N/N_ref on purpose. Mode count scales
    as E_cut**1.5 in 3D, so parametrising the cutoff as any sizeable fraction
    of E_nyquist explodes: measured at this project's real parameters
    (g~3.5e-3, N=96, L=8), E_nyquist is ~260x E_floor, and filling to it
    admitted 34% of all grid modes for a 1503% depletion fraction.

    More admitted modes still means more vacuum "particles"
    (n_offset = sum_k <|alpha_k|^2> / V), so depletion_fraction still climbs
    with N, just gently. Measured at N_ref=128, exponent=0.85, g~3.5e-3,
    N_PARTICLES=1e4: 5.10% at N=128, 6.83% at 160, 8.96% at 192, 11.03% at
    224, 13.00% at 256. run.py's <10% depletion gate therefore starts failing
    around N=224 at those parameters -- TWA's validity regime breaking down,
    not a bug. Lower growth_exponent, raise N_ref, or raise N_PARTICLES.
    """
    if safety_margin <= 0.0:
        raise ValueError(f"scale_cutoff_multiplier_to_grid: safety_margin must be > 0, got {safety_margin}")
    N = K_sq.shape[-1]
    if N_ref is None:
        N_ref = N
    growth = max(1.0, (N / N_ref) ** growth_exponent)

    E_floor = floor_multiplier * abs(g) * n_peak
    E_target = E_floor * growth
    E_nyquist = 0.5 * float(xp.max(K_sq))
    E_ceiling = E_nyquist / (safety_margin ** 2)
    E_cut = max(E_floor, min(E_target, E_ceiling))

    denom = abs(g) * n_peak
    effective_multiplier = E_cut / denom if denom > 0.0 else floor_multiplier
    info = dict(E_floor=E_floor, E_target=E_target, E_nyquist=E_nyquist, E_ceiling=E_ceiling,
                E_cut=E_cut, growth=growth, ceiling_limited=(E_target > E_ceiling),
                effective_multiplier=effective_multiplier)
    return effective_multiplier, info


def mode_variance(K_sq, mask, T: float = 0.0):
    """Per-mode <|alpha_k|^2>, same shape as K_sq: 0 outside `mask`, and
    inside it a flat 0.5 at T=0 or (1/2)*coth(E_k/2T) = n_k(T)+1/2 at T>0.

    k=0 is always the flat 0.5 at any T: it is the condensate's own mode,
    already normalised into the mean field, and n_k(T) diverges as E_k -> 0.
    Every other mode gets the full thermal enhancement, however small its E_k
    -- long-wavelength thermal excitations are the point of a finite-T run.
    """
    if T < 0.0:
        raise ValueError(f"mode_variance: T={T} < 0")
    # Built in place throughout: at production N every full-grid temporary
    # here is hundreds of MB of pool pressure right before the (much larger)
    # trajectory batch is drawn.
    mask_f = (mask.astype(xp.float32) if hasattr(mask, "astype")
              else xp.asarray(mask, dtype=xp.float32))
    if T == 0.0:
        mask_f *= 0.5
        return mask_f

    zero_mode = (K_sq == 0)
    E_safe = xp.where(zero_mode, 1.0, 0.5 * K_sq)   # placeholder keeps tanh finite at k=0
    variance = xp.where(zero_mode, 0.5, 0.5 / xp.tanh(E_safe / (2.0 * T)))
    del zero_mode, E_safe
    mask_f *= variance.astype(xp.float32)
    return mask_f


# Typed, not the python literal 1j: a python complex scalar leaves the
# promotion of `float32_array * 1j` up to the backend's casting rules, and a
# complex128 result here is a silent doubling of the largest array in the run.
_I = xp.complex64(1j)


def sample_alpha(k_shape, variance, n_traj: int, rng):
    """k-space amplitudes, shape (n_traj,)+k_shape, complex64, drawn against a
    real per-mode `variance` array (mode_variance's output).

    Written as in-place scaling of float32 normals rather than the arithmetic
    it reads as -- `re*std + 1j*(im*std)` allocates four more full
    (n_traj,N,N,N) temporaries, which at production N is gigabytes of pool
    pressure and was enough to OOM a run that previously fitted. Peak here is
    the returned complex64 array plus the two real draws.
    """
    shape = (n_traj,) + tuple(k_shape)
    std = xp.asarray(variance, dtype=xp.float32) * 0.5
    xp.sqrt(std, out=std)

    re = rng.standard_normal(shape, dtype=xp.float32)
    re *= std
    im = rng.standard_normal(shape, dtype=xp.float32)
    im *= std
    del std

    alpha = im * _I          # complex64, = i*im
    del im
    alpha += re              # ... + re, without a second complex temporary
    return alpha


def offset_density(K_sq, dV: float, mask, T: float = 0.0) -> float:
    """The constant per-grid-point density the noise itself contributes,
    sum_k <|alpha_k|^2> / V. Subtract it from the ensemble-mean raw
    (Wigner-ordered) density to recover the physical (normally-ordered) one:

        n_phys(r) = <|psi_W(r)|^2>_ensemble - offset_density(...)

    At T=0 this is exactly N_cutoff_modes/(2*V) -- NOT the flat 1/(2*dV)
    that only holds when the cutoff spans every grid mode. For a genuine
    partial cutoff the flat value over-subtracts by N^3/N_cutoff_modes, which
    makes a healthy run look like it is losing particles it never had.
    """
    V = (K_sq.shape[-1] ** 3) * dV
    return float(xp.sum(mode_variance(K_sq, mask, T))) / V


def sample_noise(K_sq, dV: float, mask, n_traj: int, rng, T: float = 0.0):
    """Batch of `n_traj` independent real-space noise fields to add to a
    mean-field ground state, shape (n_traj, N, N, N) complex64.

    Takes dV (= dx**3) rather than the box parameter L: the grid is
    linspace(-L/2, L/2, N) with both endpoints included, so the FFT's own
    periodic box is N*dx, a few percent off L at small N. V = N^3*dV is exact
    by construction.

    Normalisation: psi(r) = sum_k alpha_k * V^-0.5 * exp(i k.r) on the
    orthonormal plane-wave basis, which for FFT frequencies is
    (N^3/sqrt(V)) * ifftn(alpha) up to a per-mode phase from the grid origin
    -- dropped, since alpha_k's phase is already uniform. This yields
    <|delta_psi(r)|^2> = sum_k <|alpha_k|^2> / V at every grid point.
    """
    N = K_sq.shape[-1]
    V = (N ** 3) * dV
    alpha = sample_alpha(K_sq.shape, mode_variance(K_sq, mask, T), n_traj, rng)
    delta_psi = ifftn(alpha, axes=(-3, -2, -1))
    del alpha                       # freed before the scaling, not after
    delta_psi *= (N ** 3) / math.sqrt(V)
    return (delta_psi if delta_psi.dtype == xp.complex64
            else delta_psi.astype(xp.complex64))
