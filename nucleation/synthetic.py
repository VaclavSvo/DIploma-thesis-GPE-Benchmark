"""Analytically imprinted fields with a known vortex content.

These are the ground truth the detector is tested against: a field where the
number, position, orientation and length of every vortex is known in closed
form, so a wrong answer is unambiguous. They are also useful outside the
tests -- imprinting a known ring and re-detecting it is the cheapest way to
check that a given (N, L, mask threshold) combination can actually resolve
the cores you intend to count, before spending GPU hours on a run.

Everything here is CPU-cheap and works on whatever backend gpe3d/backend.py
selected.
"""
from __future__ import annotations

from gpe3d.backend import xp

_AXIS_FIELDS = {"x": ("Y", "Z"), "y": ("Z", "X"), "z": ("X", "Y")}


def _core_profile(r_sq, xi: float):
    """|psi| = sqrt(r^2 / (r^2 + xi^2)) -- the standard Pade approximation to
    a vortex core of healing length xi. Goes to 0 at the core (so the density
    dip criterion sees a real minimum) and to 1 far away."""
    return xp.sqrt(r_sq / (r_sq + xi * xi))


def gaussian_envelope(engine, sigma: float):
    """Smooth cloud so the density mask has something to mask to."""
    r_sq = engine.X ** 2 + engine.Y ** 2 + engine.Z ** 2
    return xp.exp(-r_sq / (2.0 * sigma * sigma)).astype(xp.float32)


def straight_vortices(engine, axis: str = "z", centres=((0.0, 0.0),), charges=(1,),
                      xi: float = 0.5, sigma: float = 3.0):
    """Any number of straight vortex lines running along `axis`.

    `centres` are the transverse coordinates of each line, in (u, v) order for
    that axis (see _AXIS_FIELDS). Phases add and core profiles multiply, so
    every line keeps its own winding and its own density hole.

    The lines span the whole box, so each is topologically open -- it ends on
    the boundary, not on itself.
    """
    fu, fv = (getattr(engine, name) for name in _AXIS_FIELDS[axis])
    amp = gaussian_envelope(engine, sigma)
    phase = None
    for (cu, cv), q in zip(centres, charges):
        u, v = fu - cu, fv - cv
        r_sq = u * u + v * v
        amp = amp * _core_profile(r_sq, xi)
        term = q * xp.arctan2(v, u)
        phase = term if phase is None else phase + term
    return (amp * xp.exp(1j * phase)).astype(xp.complex64)


def straight_vortex(engine, axis: str = "z", charge: int = 1, centre=(0.0, 0.0),
                    xi: float = 0.5, sigma: float = 3.0):
    """A single straight vortex line -- straight_vortices() for one line."""
    return straight_vortices(engine, axis, (centre,), (charge,), xi, sigma)


def vortex_ring(engine, radius: float = 2.0, xi: float = 0.5, sigma: float = 4.0,
                axis: str = "z"):
    """A vortex ring of the given radius, centred at the origin, with its
    axis along `axis` -- so the ring itself is a circle of circumference
    2*pi*radius lying in the plane normal to `axis`.

    Phase ansatz S = atan2(w, rho - R) - atan2(w, rho + R): a +1 vortex at
    (rho=R, w=0) with its image at rho=-R, which is what makes the field
    single-valued on the symmetry axis. `w` is the coordinate along `axis`
    and `rho` the distance from it.

    Detected almost entirely by the two plane families *transverse* to
    `axis`; the family normal to `axis` is tangent to the ring and sees
    little or nothing. That is the point -- a detector running only z-planes
    misses a z-axis ring, which is why all three families are swept.
    """
    fu, fv = (getattr(engine, name) for name in _AXIS_FIELDS[axis])
    w = getattr(engine, {"x": "X", "y": "Y", "z": "Z"}[axis])
    rho = xp.sqrt(fu * fu + fv * fv)
    phase = xp.arctan2(w, rho - radius) - xp.arctan2(w, rho + radius)
    core_sq = (rho - radius) ** 2 + w * w
    amp = gaussian_envelope(engine, sigma) * _core_profile(core_sq, xi)
    return (amp * xp.exp(1j * phase)).astype(xp.complex64)


def grey_soliton(engine, axis: str = "x", depth: float = 0.9, width: float = 1.0,
                 sigma: float = 3.0):
    """A grey soliton -- a density dip WITH a phase step but no winding.

    The detector's most important null: a soliton looks exactly like a vortex
    in a density slice, and a count that cannot tell them apart is worthless.
    """
    w = getattr(engine, {"x": "X", "y": "Y", "z": "Z"}[axis])
    v = xp.sqrt(max(0.0, 1.0 - depth * depth))
    tanh = xp.tanh(w / width)
    psi = (1j * v + xp.sqrt(xp.asarray(depth * depth, dtype=xp.float32)) * tanh)
    return (gaussian_envelope(engine, sigma) * psi).astype(xp.complex64)


def sound_pulse(engine, amplitude: float = 0.3, k: float = 2.0, axis: str = "x",
                sigma: float = 3.0):
    """A propagating density modulation with a smooth phase -- another null.
    No winding anywhere, however deep the density ripple gets."""
    w = getattr(engine, {"x": "X", "y": "Y", "z": "Z"}[axis])
    amp = gaussian_envelope(engine, sigma) * (1.0 + amplitude * xp.cos(k * w))
    phase = amplitude * xp.sin(k * w)
    return (amp * xp.exp(1j * phase)).astype(xp.complex64)


def uniform_with_noise(engine, noise_amplitude: float = 0.05, seed: int = 0,
                       sigma: float = 3.0):
    """A smooth cloud plus small random complex noise -- what a TW trajectory
    looks like before anything has nucleated. Must give zero detections."""
    rng = xp.random.default_rng(seed)
    shape = (engine.N, engine.N, engine.N)
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * noise_amplitude
    return (gaussian_envelope(engine, sigma) * (1.0 + noise)).astype(xp.complex64)
