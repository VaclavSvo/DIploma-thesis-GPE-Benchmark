#!/usr/bin/env python3
"""Linking tests: pierce points -> vortex lines, lengths, ring vs line.

These are what make L(t) trustworthy. A count can be checked by eye on a
slice; a length cannot, so the geometry is pinned against imprinted fields
whose length is known analytically (a line spanning the cloud, a ring of
known radius).

Standalone (`python tests/test_detect.py`) or under pytest.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.physics import GPEPhysics3D
from nucleation import synthetic
from nucleation.detect import detect_frame, line_length_from_crossings, trace_lines
from nucleation.settings import DetectorConfig

CFG = DetectorConfig(mask_threshold=0.08, mask_close_passes=4)


def _engine(N=48, L=12.0):
    return GPEPhysics3D(N=N, L=L, g=1.0)


# ---------------------------------------------------------------------------
# Synthetic point clouds -- linking logic on its own, no physics involved
# ---------------------------------------------------------------------------

def test_linker_finds_one_open_line():
    pts = np.stack([np.zeros(40), np.zeros(40), np.linspace(-4, 4, 40)], axis=1).astype(np.float32)
    lines = trace_lines(pts, np.ones(40, np.int8), link_cutoff=0.5, close_cutoff=0.5)
    assert len(lines) == 1
    assert not lines[0].closed
    assert abs(lines[0].length - 8.0) < 0.05, lines[0].length


def test_linker_finds_two_separate_lines():
    a = np.stack([np.zeros(30), np.zeros(30), np.linspace(-3, 3, 30)], axis=1)
    b = a + np.array([5.0, 0.0, 0.0])
    pts = np.concatenate([a, b]).astype(np.float32)
    lines = trace_lines(pts, np.ones(60, np.int8), link_cutoff=0.5, close_cutoff=0.5)
    assert len(lines) == 2, [ln.length for ln in lines]


def test_linker_closes_a_ring():
    theta = np.linspace(0, 2 * math.pi, 60, endpoint=False)
    r = 2.0
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)], axis=1)
    pts = pts.astype(np.float32)
    step = 2 * math.pi * r / 60
    lines = trace_lines(pts, np.ones(60, np.int8),
                        link_cutoff=2.0 * step, close_cutoff=2.0 * step)
    assert len(lines) == 1
    assert lines[0].closed, "a closed loop of points must be classified as a ring"
    assert abs(lines[0].length - 2 * math.pi * r) < 0.1 * 2 * math.pi * r, lines[0].length


# ---------------------------------------------------------------------------
# End to end, on imprinted fields
# ---------------------------------------------------------------------------

def test_straight_vortex_is_one_open_line_of_the_right_length():
    eng = _engine()
    psi = synthetic.straight_vortex(eng, axis="z", charge=1, xi=0.6, sigma=2.0)
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)

    assert frame.summary()["n_lines"] == 1, frame.summary()
    line = frame.lines[0]
    assert not line.closed
    # The line is only detected where the cloud is dense enough to mask in;
    # that span is the extent of the pierce points along z, so the traced
    # length must reproduce it.
    span = float(line.points[:, 2].max() - line.points[:, 2].min())
    assert abs(line.length - span) < 2.0 * eng.dx, (line.length, span)


def test_vortex_ring_is_one_closed_line_of_the_right_circumference():
    eng = _engine(N=64, L=12.0)
    radius = 2.5
    psi = synthetic.vortex_ring(eng, radius=radius, xi=0.6, sigma=3.0, axis="z")
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    s = frame.summary()

    assert s["n_lines"] == 1, s
    line = frame.lines[0]
    assert line.closed, "an imprinted ring must come back closed"
    expected = 2 * math.pi * radius
    assert abs(line.length - expected) < 0.25 * expected, (line.length, expected)
    # Radius, independently of the length: every pierce point sits on the ring.
    rho = np.linalg.norm(line.points[:, :2], axis=1)
    assert abs(rho.mean() - radius) < 3.0 * eng.dx, rho.mean()


def test_ring_needs_the_transverse_plane_families():
    """A z-axis ring is tangent to z-planes: running z-planes alone finds
    almost nothing. This is the concrete failure README.md warns
    about, pinned as a test so nobody 'optimises' the other two families away.
    """
    eng = _engine(N=64, L=12.0)
    psi = synthetic.vortex_ring(eng, radius=2.5, xi=0.6, sigma=3.0, axis="z")
    s = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG).summary()
    transverse = s["n_pierce_x"] + s["n_pierce_y"]
    assert transverse > 20, s
    assert s["n_pierce_z"] < 0.2 * transverse, s

    only_z = CFG.with_overrides(normals=("z",))
    s_z = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=only_z).summary()
    assert s_z["n_pierce"] < 0.2 * transverse, s_z


def test_crossing_formula_agrees_with_traced_length():
    """Independent cross-check with no linking in it at all -- if these two
    disagree wildly, link_cutoff is wrong."""
    eng = _engine(N=64, L=12.0)
    psi = synthetic.vortex_ring(eng, radius=2.5, xi=0.6, sigma=3.0, axis="z")
    frame = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG)
    s = frame.summary()
    est = line_length_from_crossings(s["n_pierce_x"], s["n_pierce_y"], s["n_pierce_z"], eng.dx)
    traced = s["L_total"]
    assert abs(est - traced) < 0.5 * traced, (est, traced)


def test_two_parallel_vortices_are_counted_as_two():
    eng = _engine()
    psi = synthetic.straight_vortices(eng, axis="z", centres=((-2.0, 0.0), (2.0, 0.0)),
                                       charges=(1, 1), xi=0.6, sigma=3.0)
    s = detect_frame(psi, eng, t=0.0, trajectory=0, cfg=CFG).summary()
    assert s["n_lines"] == 2, s


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} detection tests passed")


if __name__ == "__main__":
    _main()
