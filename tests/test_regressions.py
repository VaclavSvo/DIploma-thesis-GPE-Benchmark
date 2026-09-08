#!/usr/bin/env python3
"""One test per bug fixed in the September 2026 audit pass.

Each of these pinned a defect that the existing suite passed over: they check
*behaviour that was wrong*, not new features. Kept together so the list of
what was broken, and why it mattered, stays readable in one place.

Run: python tests/test_regressions.py   (or through pytest with the rest)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp
from gpe3d.physics import GPEPhysics3D, grid_spacing
from nucleation import statistics as st, winding as W
from nucleation.detect import _pairwise_distances
from nucleation.runner import healing_length


# ---------------------------------------------------------------------------
# gpe3d/physics.py -- run_block(0) was not a no-op
# ---------------------------------------------------------------------------

def test_run_block_zero_steps_is_a_no_op():
    """Both compositions used to apply their leading half/edge kinetic factor
    before discovering there was nothing to do. At order 4 that left an
    unpaired exp(-i*theta*dt*K^2/4) on psi -- a corrupted state returned from
    a call that should not have touched it."""
    rng = np.random.default_rng(0)
    for order in (2, 4):
        eng = GPEPhysics3D(N=8, L=4.0, g=1.0)
        eng.V = xp.zeros_like(eng.K_sq)
        eng.set_dt(0.01, order=order)
        psi0 = (rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8))
                ).astype(xp.complex64)
        eng.psi = psi0.copy()
        eng.run_block(0)
        moved = float(xp.max(xp.abs(eng.psi - psi0)))
        assert moved == 0.0, f"order={order}: run_block(0) moved psi by {moved:.3g}"
    print("PASS: run_block(0) leaves psi untouched at both orders")


def test_run_block_rejects_negative_steps():
    eng = GPEPhysics3D(N=8, L=4.0, g=1.0)
    eng.V = xp.zeros_like(eng.K_sq)
    eng.set_dt(0.01, order=2)
    eng.psi = xp.ones((8, 8, 8), dtype=xp.complex64)
    try:
        eng.run_block(-1)
    except ValueError:
        print("PASS: run_block(-1) raises")
        return
    raise AssertionError("run_block(-1) should raise, not silently do nothing")


# ---------------------------------------------------------------------------
# gpe3d/physics.py -- one definition of dx
# ---------------------------------------------------------------------------

def test_grid_spacing_matches_the_engine():
    """run.py's pre-flight aliasing gate needs dx before an engine exists and
    used L/N, while the engine's grid keeps both endpoints and so spaces its
    points L/(N-1). The gate's Nyquist pi/dx was therefore N/(N-1) too high --
    a gate that is systematically too generous, on the one failure mode
    (aliasing) that no conservation gate can catch."""
    for N, L in ((8, 4.0), (64, 16.0), (250, 24.0)):
        eng = GPEPhysics3D(N=N, L=L, g=1.0)
        assert abs(grid_spacing(N, L) - eng.dx) < 1e-6 * eng.dx, (N, L)
    print("PASS: grid_spacing() agrees with GPEPhysics3D.dx")


# ---------------------------------------------------------------------------
# gpe3d/physics.py -- the density-slice path
# ---------------------------------------------------------------------------

def test_density_slice_matches_the_full_density():
    """get_density_slice_numpy() replaces get_density_numpy()[n], which built
    the whole N^3 density and copied all of it off the GPU to keep one plane."""
    eng = GPEPhysics3D(N=12, L=6.0, g=1.0)
    rng = np.random.default_rng(5)
    eng.psi = (rng.standard_normal((12, 12, 12))
               + 1j * rng.standard_normal((12, 12, 12))).astype(xp.complex64)
    full = eng.get_density_numpy()
    for n in (0, 6, 11):
        assert np.array_equal(full[n], eng.get_density_slice_numpy(n)), n
    print("PASS: get_density_slice_numpy() == get_density_numpy()[n]")


# ---------------------------------------------------------------------------
# nucleation/winding.py -- shared phase differences, in-place morphology
# ---------------------------------------------------------------------------

def _imprinted(eng):
    from nucleation import synthetic
    return synthetic.straight_vortices(eng, "z", ((0.5, 0.0), (-1.2, 0.8)), (1, -1),
                                       xi=0.5, sigma=3.0)


def test_shared_phase_differences_match_the_per_family_form():
    """phase_differences() + winding_from_differences() halve the arctan2 work
    of a 3-plane sweep; they must reproduce winding_volume() exactly."""
    eng = GPEPhysics3D(N=24, L=10.0, g=1.0)
    psi = _imprinted(eng)
    diffs = W.phase_differences(psi, ("x", "y", "z"))
    for normal in ("x", "y", "z"):
        a = W.winding_volume(psi, normal)
        b = W.winding_from_differences(diffs, normal)
        assert float(xp.max(xp.abs(a - b))) < 1e-6, normal
    print("PASS: shared phase differences reproduce winding_volume() on all 3 families")


def test_morphology_is_unchanged_by_the_in_place_rewrite():
    """dilate/erode now accumulate in place. One pass must still be exactly one
    6-neighbour cell -- rolling the partly-grown result instead of the pass's
    input would silently grow the mask faster than `passes` says."""
    rng = np.random.default_rng(11)
    mask = xp.asarray(rng.random((16, 16, 16)) > 0.85)

    def reference(m, passes, op):
        for _ in range(passes):
            out = m
            for axis in (-3, -2, -1):
                out = op(op(out, xp.roll(m, 1, axis=axis)), xp.roll(m, -1, axis=axis))
            m = out
        return m

    for passes in (1, 2, 4):
        assert bool(xp.all(W.dilate(mask, passes)
                           == reference(mask, passes, lambda a, b: a | b)))
        assert bool(xp.all(W.erode(mask, passes)
                           == reference(mask, passes, lambda a, b: a & b)))
    print("PASS: in-place dilate/erode match the out-of-place reference")


def test_local_density_dip_does_not_mutate_its_input():
    eng = GPEPhysics3D(N=16, L=8.0, g=1.0)
    dens = eng._density(_imprinted(eng))
    before = dens.copy()
    W.local_density_dip(dens, 0.75)
    assert bool(xp.all(dens == before)), "local_density_dip mutated its argument"
    print("PASS: local_density_dip() leaves its input alone")


def test_integrality_error_accepts_a_precomputed_rounding():
    rng = np.random.default_rng(3)
    w = xp.asarray((rng.integers(-1, 2, (8, 8, 8)) + 0.03 * rng.standard_normal((8, 8, 8)))
                   .astype(np.float32))
    assert abs(W.integrality_error(w) - W.integrality_error(w, rounded=xp.rint(w))) < 1e-7
    assert abs(W.integrality_error(w) - float(xp.max(xp.abs(w - xp.rint(w))))) < 1e-6
    print("PASS: integrality_error() unchanged, with and without a shared rint")


# ---------------------------------------------------------------------------
# nucleation/detect.py -- chunked distance matrix
# ---------------------------------------------------------------------------

def test_chunked_distances_match_the_dense_form():
    """The (P,P,3) difference the old one-liner materialised is three times the
    matrix it reduces to; at the old max_component=20000 guard that was 4.8GB
    of temporary, i.e. the guard could not fire before the allocation it
    guarded against had failed."""
    rng = np.random.default_rng(0)
    pts = (rng.standard_normal((500, 3)) * 4.0).astype(np.float32)
    ref = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    assert np.array_equal(ref, _pairwise_distances(pts, row_chunk=64))
    print("PASS: chunked pairwise distances are bit-identical to the dense form")


# ---------------------------------------------------------------------------
# nucleation/runner.py -- one healing-length convention
# ---------------------------------------------------------------------------

def test_healing_length_uses_the_same_convention_as_the_noise_cutoff():
    """gpe3d/noise.py ties its UV cutoff to xi = 1/sqrt(2*g*n); runner.py
    reported dx/xi against 1/sqrt(g*n), larger by sqrt(2). The printed
    resolution verdict was optimistic by that factor."""
    g, n = 3.5e-3, 120.0
    assert abs(healing_length(g, n) - 1.0 / math.sqrt(2.0 * g * n)) < 1e-12
    # E_k at the healing wavenumber must equal the interaction energy g*n,
    # which is the scale energy_cutoff_mask() measures its cutoff against.
    k_xi = 1.0 / healing_length(g, n)
    assert abs(0.5 * k_xi ** 2 - g * n) < 1e-12 * g * n
    assert healing_length(0.0, n) == float("inf")
    print("PASS: healing_length() == 1/sqrt(2 g n), consistent with the noise cutoff")


# ---------------------------------------------------------------------------
# nucleation/statistics.py -- aggregate() on a non-default observable
# ---------------------------------------------------------------------------

def _rows(n_traj=3, n_t=5):
    return [dict(trajectory=tr, t=float(i), n_lines=i + tr, L_total=1.5 * i, n_pierce=2 * i)
            for tr in range(n_traj) for i in range(n_t)]


def test_aggregate_works_without_n_lines():
    """aggregate() read out["n_lines"] unconditionally, so every observable
    except that one raised KeyError -- and compare_to_control(key=...) with
    it, which is the documented way to compare runs on L_total."""
    agg = st.aggregate(_rows(), observables=("L_total",), n_resamples=50)
    assert agg["count_key"] == "L_total"
    assert agg["final_counts"].shape == (3,)
    cmp = st.compare_to_control(_rows(), _rows(), key="L_total", n_resamples=50)
    assert set(cmp) >= {"tw_mean", "control_mean", "control_clean", "separated"}
    # and n_lines still drives onset when it is among the observables
    agg2 = st.aggregate(_rows(), observables=("L_total", "n_lines"), n_resamples=50)
    assert agg2["count_key"] == "n_lines"
    print("PASS: aggregate()/compare_to_control() work on any observable")


# ---------------------------------------------------------------------------
# nucleation/vortex_map.py -- one trajectory, not all of them interleaved
# ---------------------------------------------------------------------------

def test_vortex_map_loads_a_single_trajectory(tmp_path=None):
    """The observer stores one frame per (trajectory, time). load_run() used to
    return all of them, so an 8-trajectory run animated eight realisations
    cutting between each other, with every time repeated eight times on the
    count-vs-time axis."""
    import tempfile
    from nucleation.io import write_pierce_points
    from nucleation.vortex_map import load_run

    n_traj, n_t = 3, 4
    counts = np.arange(n_traj * n_t) + 1
    arrays = dict(
        frame_t=np.array([float(i) for i in range(n_t) for _ in range(n_traj)], np.float32),
        frame_trajectory=np.array([tr for _ in range(n_t) for tr in range(n_traj)], np.int32),
        frame_offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        points=np.zeros((int(counts.sum()), 3), np.float32),
        charges=np.ones(int(counts.sum()), np.int8),
        normals=np.zeros(int(counts.sum()), np.uint8),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = write_pierce_points(arrays, os.path.join(tmp, "p.npz"))
        for want in range(n_traj):
            times, pts = load_run(path, trajectory=want)
            assert len(times) == n_t, (want, len(times))
            assert np.all(np.diff(times) > 0), "times must be strictly increasing"
            expected = [counts[i * n_traj + want] for i in range(n_t)]
            assert [p.shape[0] for p in pts] == expected, (want, expected)
        assert len(load_run(path, trajectory=None)[0]) == n_traj * n_t
        # an absent trajectory falls back rather than producing an empty movie
        assert len(load_run(path, trajectory=99)[0]) == n_t
    print("PASS: vortex_map.load_run() selects one trajectory, in time order")


# ---------------------------------------------------------------------------
# config.py -- the two mislabelled collision numbers
# ---------------------------------------------------------------------------

def test_collision_rows_are_self_consistent():
    """n_clouds must match the two clouds two_cloud_collision_psi() builds --
    tw_collision said 4.0, which renormalised each 5e3 cloud to 1e4 atoms and
    measured depletion_fraction against a target twice the real one. And
    v_rel_real must be the relative velocity, not twice it."""
    import config
    for name, row in config.SCENARIOS.items():
        if row["solver"] != "collision":
            continue
        assert row["n_clouds"] == 2.0, f"{name}: n_clouds={row['n_clouds']}"
        assert "v_split" not in row, f"{name}: v_split is gone; v_rel_real is the real thing"
    print("PASS: every collision row has n_clouds=2 and a single velocity knob")


def test_configured_grid_sizes_are_fft_friendly():
    """FFT cost is set by N's largest prime factor, not its size."""
    from gpe3d.backend import _largest_prime_factor
    for name, row in config.SCENARIOS.items():
        f = _largest_prime_factor(row["N"])
        assert f <= 13, f"{name}: N={row['N']} has prime factor {f}"
    print("PASS: every scenario's N factors into small primes")


import config  # noqa: E402  (used by the two tests above)

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nRegression tests: all passed.")
