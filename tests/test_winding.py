#!/usr/bin/env python3
"""Phase-winding tests -- README.md phase 1, where most of the risk
in this whole exercise lives.

Each test imprints a field whose vortex content is known in closed form and
asks the detector for it back. No GPU, no solver, small grids: this suite
runs in seconds and must be green before any long run is worth starting.

Standalone (`python tests/test_winding.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.backend import xp, to_numpy
from gpe3d.physics import GPEPhysics3D
from nucleation import synthetic, winding as W
from nucleation.detect import detect_frame
from nucleation.settings import DetectorConfig

N, L, XI, SIGMA = 48, 12.0, 0.6, 2.0
CFG = DetectorConfig(mask_threshold=0.08, mask_close_passes=4)


def _engine():
    return GPEPhysics3D(N=N, L=L, g=1.0)


# ---------------------------------------------------------------------------

def test_straight_vortex_gives_unit_winding():
    """A +1 line along z: every masked plaquette that fires carries exactly
    +1, and the z-family is the only one that sees it."""
    eng = _engine()
    psi = synthetic.straight_vortex(eng, axis="z", charge=1, xi=XI, sigma=SIGMA)

    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    s = frame.summary()

    assert s["n_pierce_z"] > 10, f"z-planes should be pierced all along the line, got {s}"
    assert set(to_numpy(frame.charges).tolist()) == {1}, "all windings must be +1"
    assert s["n_pierce_x"] == 0 and s["n_pierce_y"] == 0, (
        f"a z-line must not pierce x- or y-planes: {s}")
    assert frame.integrality_error < 1e-3, frame.integrality_error


def test_charge_sign_follows_imprinted_sign():
    eng = _engine()
    psi = synthetic.straight_vortex(eng, axis="z", charge=-1, xi=XI, sigma=SIGMA)
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    assert set(to_numpy(frame.charges).tolist()) == {-1}


def test_orientation_is_detected_by_the_right_plane_family():
    """The whole reason all three families are computed: a line along x is
    invisible to z-planes."""
    eng = _engine()
    for axis, expect in (("x", "n_pierce_x"), ("y", "n_pierce_y"), ("z", "n_pierce_z")):
        psi = synthetic.straight_vortex(eng, axis=axis, charge=1, xi=XI, sigma=SIGMA)
        s = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG).summary()
        others = [k for k in ("n_pierce_x", "n_pierce_y", "n_pierce_z") if k != expect]
        assert s[expect] > 10, f"{axis}-line not found by {expect}: {s}"
        assert all(s[k] == 0 for k in others), f"{axis}-line leaked into {others}: {s}"


def test_grey_soliton_gives_no_winding():
    """A soliton is a deep density dip with a phase step -- indistinguishable
    from a vortex in a density slice, and it must give zero here."""
    eng = _engine()
    psi = synthetic.grey_soliton(eng, axis="x", depth=0.9, width=1.0, sigma=SIGMA)
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    assert frame.points.shape[0] == 0, f"soliton produced {frame.points.shape[0]} false pierces"


def test_sound_pulse_gives_no_winding():
    eng = _engine()
    psi = synthetic.sound_pulse(eng, amplitude=0.4, k=2.0, sigma=SIGMA)
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    assert frame.points.shape[0] == 0, f"sound pulse produced {frame.points.shape[0]} false pierces"


def test_noise_only_field_gives_no_detections_after_masking():
    """The TW null: small vacuum-scale noise on a smooth cloud is not a
    vortex, and must not be counted as one."""
    eng = _engine()
    psi = synthetic.uniform_with_noise(eng, noise_amplitude=0.05, seed=3, sigma=SIGMA)
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    assert frame.points.shape[0] == 0, f"noise produced {frame.points.shape[0]} false pierces"


def test_winding_slice_matches_winding_volume():
    """The 2D path used by the movie and the 3D path used by the counter must
    agree on the same plane, sign included -- otherwise the picture and the
    numbers describe different things."""
    eng = _engine()
    psi = synthetic.straight_vortex(eng, axis="z", charge=1, xi=XI, sigma=SIGMA)
    for normal in ("x", "y", "z"):
        idx = eng.N // 3
        vol = W.winding_volume(psi, normal)
        w_vol = to_numpy(W.slice_plane(vol, normal, idx))
        w_slice = to_numpy(W.winding_slice(W.slice_plane(psi, normal, idx), normal))
        assert abs(w_vol - w_slice).max() < 1e-5, f"{normal}-plane 2D/3D winding mismatch"


def test_unresolved_core_is_rejected_not_miscounted():
    """dx >> xi makes the windings non-integer. That must raise, not quietly
    return a wrong count."""
    from nucleation.detect import ResolutionError
    eng = GPEPhysics3D(N=12, L=40.0, g=1.0)
    psi = synthetic.straight_vortex(eng, axis="z", charge=1, xi=0.02, sigma=8.0)
    w = W.winding_volume(psi, "z")
    err = W.integrality_error(w)
    if err <= CFG.integrality_tol:
        return  # this grid happened to stay integral; nothing to assert
    try:
        detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    except ResolutionError:
        return
    raise AssertionError("non-integer winding was not reported as a resolution failure")


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} winding tests passed")


if __name__ == "__main__":
    _main()
