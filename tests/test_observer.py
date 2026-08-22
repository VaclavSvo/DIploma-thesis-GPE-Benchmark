#!/usr/bin/env python3
"""The observer against live dynamics, not a frozen field.

tests/test_winding.py and tests/test_detect.py pin the detector on static
imprinted fields. This one checks the piece those cannot: that the hook into
gpe3d/evolve.py fires at every frame, sees each trajectory's own psi, and
keeps finding a real vortex while the GPE actually propagates it -- and that
the same solver with no vortex in it returns nothing.

Small grid, short time, CPU seconds. Standalone or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d import evolve
from gpe3d.backend import xp, to_numpy
from gpe3d.base import State
from gpe3d.solver import ClassicalTrappedGPESolver
from nucleation.observer import NucleationObserver, plane_index
from nucleation.settings import DetectorConfig
from nucleation.synthetic import _core_profile

CFG = DetectorConfig(mask_threshold=0.08, mask_close_passes=4, slice_planes=("z", "x"))
PARAMS = dict(n_particles=1.0, imag_time_steps=200, verbose=False, dt=2e-3)


def _solver_and_state(with_vortex: bool, xi: float = 0.6):
    solver = ClassicalTrappedGPESolver(N=32, L=8.0, g=1.0, omega=(1.0, 1.0, 1.0),
                                        splitstep_order=2)
    state = solver.init_state(PARAMS)
    if with_vortex:
        eng = solver.engine
        r_sq = eng.X ** 2 + eng.Y ** 2
        psi = state.psi * _core_profile(r_sq, xi) * xp.exp(1j * xp.arctan2(eng.Y, eng.X))
        state = State(psi=psi.astype(xp.complex64), t=0.0, step=0)
    return solver, state


def _run(with_vortex: bool, n_blocks: int = 4):
    solver, state = _solver_and_state(with_vortex)
    obs = NucleationObserver(solver.engine, CFG)
    evolve.run_and_record(solver, state, n_blocks=n_blocks, steps_per_block=20,
                          observer=obs)
    return solver, obs


def test_observer_records_one_frame_per_block():
    _, obs = _run(with_vortex=True, n_blocks=4)
    assert len(obs.frames) == 5, len(obs.frames)     # t=0 plus one per block
    assert len(obs.times) == 5
    assert [f.t for f in obs.frames] == sorted(f.t for f in obs.frames)


def test_observer_tracks_an_imprinted_vortex_through_evolution():
    _, obs = _run(with_vortex=True, n_blocks=4)
    rows = obs.summary_table()
    assert all(r["n_lines"] >= 1 for r in rows), [r["n_lines"] for r in rows]
    assert all(r["n_pierce_z"] > 5 for r in rows), [r["n_pierce_z"] for r in rows]
    for f in obs.frames:
        assert set(to_numpy(f.charges).tolist()) == {1}, "charge must not flip during evolution"
        assert f.integrality_error < 1e-3


def test_plain_ground_state_stays_empty():
    """The null with real dynamics in it: a vortex-free condensate must give
    zero at every frame, or the detector is firing on the cloud itself."""
    _, obs = _run(with_vortex=False, n_blocks=4)
    counts = [r["n_pierce"] for r in obs.summary_table()]
    assert counts == [0] * len(counts), counts


def test_movie_slices_are_recorded_for_every_requested_plane():
    solver, obs = _run(with_vortex=True, n_blocks=3)
    for normal in CFG.slice_planes:
        dens = obs.slices[(normal, "density")]
        wind = obs.slices[(normal, "winding")]
        assert len(dens) == len(wind) == len(obs.times)
        assert dens[0].shape == (solver.engine.N, solver.engine.N)
        assert wind[0].dtype.kind == "i", "winding slices are stored as integers"
    # The z-plane through the vortex line must actually show the core.
    z_wind = obs.slices[("z", "winding")]
    assert abs(z_wind[0]).sum() >= 1, "the movie's winding panel shows no vortex"


def test_plane_index_picks_the_plane_nearest_zero():
    """For even N the grid straddles 0 with two equidistant planes, so this
    checks the distance, not a particular index -- N//2 alone would be half a
    cell off, which is the bug this helper exists to avoid."""
    solver, _ = _solver_and_state(with_vortex=False)
    eng = solver.engine
    coords = to_numpy(eng.Z).ravel()
    for normal in ("x", "y", "z"):
        idx = plane_index(eng, normal, 0.0)
        assert 0 <= idx < eng.N
        assert abs(abs(coords[idx]) - abs(coords).min()) < 1e-6, (idx, coords[idx])


def test_pierce_arrays_round_trip_per_frame():
    import numpy as np
    _, obs = _run(with_vortex=True, n_blocks=2)
    arrays = obs.pierce_arrays()
    offsets = arrays["frame_offset"]
    assert offsets.size == len(obs.frames) + 1
    for i, frame in enumerate(obs.frames):
        lo, hi = offsets[i], offsets[i + 1]
        assert hi - lo == frame.points.shape[0]
        assert np.allclose(arrays["points"][lo:hi], frame.points)


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} observer tests passed")


if __name__ == "__main__":
    _main()
