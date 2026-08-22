#!/usr/bin/env python3
"""End-to-end smoke test of a nucleation run, shrunk to seconds on CPU.

Not a physics test -- N=32 cannot resolve a vortex core, and the point here
is only that the whole pipeline connects: solver -> ensemble loop -> observer
-> detection -> CSV/npz/gif/metadata, for BOTH the TW scenario and its
mean-field control. The physics is pinned by tests/test_winding.py and
tests/test_detect.py, which use imprinted fields with known answers.

Run: python tests/smoke_nucleation.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def _shrink(scenario: str, out_dir: str) -> None:
    """Override config in place: a real nucleation run is N=384 over T=8."""
    config.SCENARIO = scenario
    config.N, config.L = 32, 16.0
    config.T_TOTAL = 0.4
    config.NUCLEATION_FRAMES = 4
    config.N_TRAJECTORIES = 2 if scenario == "nucleation_collision" else 1
    config.BATCH_SIZE = 2
    config.OUT_DIR = out_dir
    config.VERBOSE = False
    config.TOL = 1.0                      # not a conservation test
    config.COLLISION_HALF_VELOCITY = (2.0, 0.0, 0.0)
    config.SOLVER_PARAMS = dict(config.SOLVER_PARAMS)
    config.SOLVER_PARAMS.update(half_velocity=config.COLLISION_HALF_VELOCITY,
                                imag_time_steps=40, verbose=False)


def _check_outputs(out_dir: str, scenario: str, expect_gif: bool = True) -> None:
    from nucleation.io import read_observables_csv

    d = os.path.join(out_dir, scenario)
    for name in ("observables.csv", "pierce_points.npz", "run_metadata.json"):
        path = os.path.join(d, name)
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"missing/empty {path}"
    if expect_gif:
        assert os.path.exists(os.path.join(d, "density_winding.gif"))

    rows = read_observables_csv(os.path.join(d, "observables.csv"))
    n_traj = len({r["trajectory"] for r in rows})
    times = sorted({round(r["t"], 9) for r in rows})
    # Detection runs every NUCLEATION_DETECT_STRIDE blocks, plus always on the
    # last one -- so the frame count follows from both, and the final block
    # must be present whatever the stride.
    blocks = config.NUCLEATION_FRAMES - 1
    stride = config.NUCLEATION_DETECT_STRIDE
    expected = len({*range(0, blocks + 1, stride), blocks})
    assert len(times) == expected, (len(times), expected)
    assert len(rows) == n_traj * len(times), (len(rows), n_traj, len(times))
    n_times = len(times)
    for key in ("n_lines", "L_total", "n_pierce_x", "n_pierce_y", "n_pierce_z"):
        assert key in rows[0], f"{key} missing from the observables table"
    print(f"  {scenario}: {n_traj} trajectories x {n_times} frames, "
          f"final n_lines={[r['n_lines'] for r in rows if r['t'] == rows[-1]['t']]}")


def main() -> int:
    import importlib
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ("nucleation_collision", "nucleation_control"):
            _shrink(scenario, tmp)
            run = importlib.import_module("run")
            importlib.reload(run)
            print(f"\n=== {scenario} ===")
            run.main()
            _check_outputs(tmp, scenario)
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
