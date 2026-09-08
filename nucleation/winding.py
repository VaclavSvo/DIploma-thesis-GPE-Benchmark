"""Plaquette phase winding -- the primitive every vortex diagnostic here is
built on.

A quantised vortex is a line about which the condensate phase winds by an
integer multiple of 2*pi. Detect it by walking the four corners of each grid
plaquette and summing the wrapped phase differences:

    w = [ d(a->b) + d(b->c) + d(c->d) + d(d->a) ] / 2*pi
    d(p->q) = arg( psi_q * conj(psi_p) )

Always the product form, never arg(psi_q) - arg(psi_p): the product is
already wrapped to (-pi, pi] by construction, the difference is not and
needs branch handling that silently goes wrong.

w is an integer for a resolved core (-1, 0, +1, occasionally +-2 for two
cores in one plaquette). Non-integer w means dx is too coarse to resolve the
core -- a free, built-in resolution check, exposed as integrality_error().

Array convention, inherited from gpe3d/physics.py's grid: the last three
axes of psi are (z, y, x) == (-3, -2, -1). Everything here uses negative
axis indices, so a plain (N,N,N) field and a batched (B,N,N,N) TW ensemble
go through the same code path.

Periodic boundaries: the solver is spectral, so the box genuinely wraps.
Plaquettes are therefore built with xp.roll and every winding array has the
same shape as its input -- no edge trimming, no index bookkeeping.
"""
from __future__ import annotations

import math

from gpe3d.backend import xp

AX_Z, AX_Y, AX_X = -3, -2, -1

# normal axis -> (u, v) transverse axes, ordered so that (u, v, normal) is
# right-handed. Circulation is then measured about the +normal direction, and
# the sign of w is the sign of the vortex charge along that axis.
_TRANSVERSE = {
    "x": (AX_Y, AX_Z),
    "y": (AX_Z, AX_X),
    "z": (AX_X, AX_Y),
}
_NORMAL_AXIS = {"x": AX_X, "y": AX_Y, "z": AX_Z}
NORMALS = ("x", "y", "z")

_TWO_PI = 2.0 * math.pi


def _phase_diff(psi, axis):
    """arg(psi[i+1] * conj(psi[i])) along `axis`, wrapped to (-pi, pi].

    Written on the real/imaginary parts rather than as a complex product so
    no full complex temporary is allocated -- at N=384 that is a 450MB array
    per call, and this runs three times per frame.
    """
    pr, pi_ = psi.real, psi.imag
    qr = xp.roll(pr, -1, axis=axis)
    qi = xp.roll(pi_, -1, axis=axis)
    im = qi * pr
    im -= qr * pi_
    re = qr * pr
    re += qi * pi_
    return xp.arctan2(im, re)


def phase_differences(psi, normals=NORMALS) -> dict:
    """{axis: arg(psi[..+1] * conj(psi))} for every axis the requested plane
    families need -- computed ONCE and shared between them.

    Each plane family uses the two axes transverse to its normal, so across
    all three families every axis is needed exactly twice. Computing them per
    family runs six arctan2 sweeps of the full grid per trajectory per frame
    where three will do; arctan2 is the single most expensive operation in
    the detector.

    Costs one extra resident float32 (N,N,N) (three cached differences rather
    than the two a single family holds) and saves half the phase work.
    """
    axes = {a for n in normals for a in _TRANSVERSE[n]}
    return {a: _phase_diff(psi, a) for a in axes}


def winding_from_differences(diffs: dict, normal: str):
    """Plaquette winding for one plane family from phase_differences()'s
    cache. Same result as winding_volume(psi, normal), one array allocated.
    """
    u, v = _TRANSVERSE[normal]
    du, dv = diffs[u], diffs[v]
    w = xp.roll(dv, -1, axis=u)
    w += du
    w -= xp.roll(du, -1, axis=v)
    w -= dv
    w *= 1.0 / _TWO_PI      # in place: `w / _TWO_PI` is another full grid
    return w


def plaquette_winding(psi, axis_u, axis_v):
    """Winding number per plaquette spanned by (axis_u, axis_v).

    Result[..., i, j] belongs to the plaquette whose lower corner is index
    (i, j) along (axis_u, axis_v) -- i.e. its centre sits half a cell up in
    both directions (see pierce_coordinates() in detect.py).

    Returns a float array; round it and check integrality_error() before
    treating the values as charges.
    """
    du = _phase_diff(psi, axis_u)
    dv = _phase_diff(psi, axis_v)
    w = xp.roll(dv, -1, axis=axis_u)
    w += du
    w -= xp.roll(du, -1, axis=axis_v)
    w -= dv
    w *= 1.0 / _TWO_PI
    return w


def winding_volume(psi, normal: str):
    """Full (…, N, N, N) winding field for the plane family normal to
    `normal` ("x", "y" or "z").

    A vortex line is found by the planes it *pierces*: running only z-planes
    finds lines along z and misses a ring lying in the y-z plane entirely.
    Compute all three families (or at least the ones you can afford) so lines
    of any orientation are caught.

    Sweeping more than one family: use phase_differences() +
    winding_from_differences() instead, which shares the phase work.
    """
    u, v = _TRANSVERSE[normal]
    return plaquette_winding(psi, u, v)


def slice_axes(normal: str):
    """(axis_u, axis_v) of an extracted 2D plane, in the same right-handed
    convention winding_volume() uses for the full field.

    slice_plane() drops the normal axis, so the two surviving axes are the
    transverse pair in their original relative order; this maps (u, v) onto
    the remaining (-2, -1) so the 2D and 3D paths agree on the SIGN of the
    winding, not just its magnitude.
    """
    u, _ = _TRANSVERSE[normal]
    axes_left = sorted(a for a in (AX_Z, AX_Y, AX_X) if a != _NORMAL_AXIS[normal])
    axis_u = -2 if axes_left[0] == u else -1
    return axis_u, (-1 if axis_u == -2 else -2)


def winding_slice(psi_2d, normal: str):
    """Winding of a single extracted 2D plane, for plotting.

    `psi_2d` must come from slice_plane(), which fixes the axis order.
    """
    return plaquette_winding(psi_2d, *slice_axes(normal))


def slice_plane(field, normal: str, index: int):
    """Extract the 2D plane at `index` along `normal`, keeping any leading
    batch axes. Used for both psi (-> winding_slice) and density (-> plot).
    """
    axis = _NORMAL_AXIS[normal]
    return xp.take(field, index, axis=axis)


def integrality_error(w, rounded=None) -> float:
    """max |w - round(w)| -- 0 for perfectly resolved cores. A large value
    means dx is too coarse for the healing length and the "charges" being
    counted are not integers at all. Worth asserting every frame.

    `rounded` lets a caller that already needs rint(w) (the pierce finder
    needs it as the charge) hand it in, so the rounding is done once.

    max|d| is taken as max(max(d), -min(d)) rather than max(abs(d)): two
    reductions over an array already in memory instead of one more full
    (N,N,N) temporary.
    """
    d = xp.rint(w) if rounded is None else rounded
    d = d - w
    return float(max(float(xp.max(d)), -float(xp.min(d))))


_SPATIAL_AXES = (AX_Z, AX_Y, AX_X)


def dilate(mask, passes: int = 1, axes=_SPATIAL_AXES):
    """Grow a boolean mask by `passes` cells along each of `axes`.

    One pass is a single 6-neighbour (von Neumann) dilation: every roll is of
    the pass's INPUT mask, never of the partly-grown result, so `passes` cells
    means exactly `passes` cells. Accumulated in place -- with close_passes=4
    the out-of-place form allocated 24 full boolean grids per call.
    """
    for _ in range(passes):
        grown = mask.copy()
        for axis in axes:
            grown |= xp.roll(mask, 1, axis=axis)
            grown |= xp.roll(mask, -1, axis=axis)
        mask = grown
    return mask


def erode(mask, passes: int = 1, axes=_SPATIAL_AXES):
    """Shrink a boolean mask by `passes` cells along each of `axes` -- the
    exact dual of dilate(), same in-place accumulation."""
    for _ in range(passes):
        shrunk = mask.copy()
        for axis in axes:
            shrunk &= xp.roll(mask, 1, axis=axis)
            shrunk &= xp.roll(mask, -1, axis=axis)
        mask = shrunk
    return mask


def density_mask(dens, n_peak: float, threshold: float, close_passes: int = 3,
                 axes=_SPATIAL_AXES):
    """"Inside the condensate": |psi|^2 > threshold * n_peak, morphologically
    closed (dilate then erode) so vortex cores don't punch holes in it.

    The masking itself is the single biggest source of false positives:
    outside the condensate the phase is pure noise, and in a TW run the
    *entire box* carries vacuum noise, so an unmasked count returns thousands
    of spurious vortices.

    The closing matters just as much in the other direction, and is easy to
    miss: a resolved core has |psi|^2 -> 0 at its centre, so the cells around
    a genuine vortex fall BELOW the threshold and a raw mask deletes exactly
    the plaquettes worth counting. Closing fills those holes without moving
    the cloud boundary outward (which a plain dilation would). close_passes
    should comfortably exceed the core radius in cells, i.e. >~ xi/dx.

    `threshold` (0.05-0.1) is a convergence parameter, not a constant: the
    reported count must be stable across a sweep of it, or the detection is
    not real.
    """
    mask = dens > (threshold * n_peak)
    if close_passes:
        mask = erode(dilate(mask, close_passes, axes), close_passes, axes)
    return mask


def plaquette_mask(mask, axis_u, axis_v):
    """A plaquette is valid only if all four of its corners are inside the
    condensate -- otherwise a plaquette straddling the cloud edge picks up
    the vacuum's random phase on one corner and fires."""
    m = mask & xp.roll(mask, -1, axis=axis_u)
    m &= xp.roll(mask, -1, axis=axis_v)
    m &= xp.roll(xp.roll(mask, -1, axis=axis_u), -1, axis=axis_v)
    return m


def plaquette_any(flag, axis_u, axis_v):
    """True where ANY of a plaquette's four corners satisfies `flag`.

    Used for the density-dip criterion, which plaquette_mask's all-four-corners
    rule would be wrong for: the core sits *inside* the plaquette, so only the
    corner(s) nearest it are guaranteed to be local minima.
    """
    m = flag | xp.roll(flag, -1, axis=axis_u)
    m |= xp.roll(flag, -1, axis=axis_v)
    m |= xp.roll(xp.roll(flag, -1, axis=axis_u), -1, axis=axis_v)
    return m


def local_density_dip(dens, factor: float = 0.6, axes=_SPATIAL_AXES):
    """True where the density is below `factor` times its local 3x3x3 box mean
    -- a genuine core has |psi|^2 -> 0 at its centre.

    Winding AND a density dip is much stronger than either alone, and is what
    separates a real core from a phase feature riding on a sound pulse. Built
    from rolls (a separable box filter), so it needs no scipy and runs on both
    backends.
    """
    smooth = dens
    for axis in axes:
        nxt = xp.roll(smooth, 1, axis=axis)
        nxt += smooth
        nxt += xp.roll(smooth, -1, axis=axis)
        nxt *= 1.0 / 3.0
        smooth = nxt
    if smooth is dens:      # axes=() -- never mutate the caller's array
        return dens < (factor * dens)
    smooth *= factor        # `factor * smooth` was one more full grid
    return dens < smooth
