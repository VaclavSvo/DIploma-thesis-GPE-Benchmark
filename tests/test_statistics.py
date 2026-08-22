#!/usr/bin/env python3
"""Aggregation tests: counting, bootstrap intervals, onset, control
comparison -- all on synthetic row tables, no physics.

The point is to pin the *statistics protocol* itself: that misaligned
trajectories are refused rather than silently averaged, that the sustain
requirement suppresses single-frame detector twitches, and that a run only
counts as a result when the mean-field control stays at zero.

Standalone (`python tests/test_statistics.py`) or under pytest.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nucleation import statistics as st
from nucleation.io import read_observables_csv, write_observables_csv


def _rows(counts_per_traj, times, lengths=None):
    rows = []
    for tr, counts in enumerate(counts_per_traj):
        for j, t in enumerate(times):
            rows.append(dict(t=float(t), trajectory=tr,
                             n_pierce=int(counts[j]) * 10,
                             n_lines=int(counts[j]),
                             L_total=float(counts[j] if lengths is None else lengths[tr][j])))
    return rows


def test_to_matrix_shapes_and_order():
    times = [0.0, 1.0, 2.0]
    rows = _rows([[0, 1, 2], [0, 0, 3]], times)
    t, v = st.to_matrix(rows, "n_lines")
    assert list(t) == times
    assert v.shape == (2, 3)
    assert v[1, 2] == 3


def test_to_matrix_refuses_misaligned_trajectories():
    rows = _rows([[0, 1]], [0.0, 1.0]) + [dict(t=5.0, trajectory=1, n_lines=2,
                                                n_pierce=20, L_total=2.0)]
    try:
        st.to_matrix(rows, "n_lines")
    except ValueError:
        return
    raise AssertionError("misaligned time grids must be refused, not averaged")


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_n():
    rng = np.random.default_rng(0)
    small = rng.poisson(3.0, size=(6, 4)).astype(float)
    large = rng.poisson(3.0, size=(200, 4)).astype(float)
    m_s, lo_s, hi_s = st.bootstrap_ci(small, n_resamples=800, seed=1)
    m_l, lo_l, hi_l = st.bootstrap_ci(large, n_resamples=800, seed=1)
    assert np.all(lo_s <= m_s) and np.all(m_s <= hi_s)
    assert np.mean(hi_l - lo_l) < np.mean(hi_s - lo_s), "more trajectories must narrow the CI"


def test_onset_requires_a_sustained_detection():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    counts = np.array([
        [0, 1, 0, 0, 0],      # single-frame twitch -- not a nucleation event
        [0, 0, 1, 1, 1],      # sustained from t=2
        [0, 0, 0, 0, 0],      # never
    ])
    onset = st.nucleation_onset(times, counts, threshold=1, sustain=3)
    assert np.isnan(onset[0]), "a one-frame blip must not count as onset"
    assert onset[1] == 2.0
    assert np.isnan(onset[2])


def test_aggregate_reports_final_counts_and_onsets():
    times = [0.0, 1.0, 2.0, 3.0]
    rows = _rows([[0, 1, 1, 1], [0, 0, 2, 2], [0, 0, 0, 0]], times)
    agg = st.aggregate(rows, observables=("n_lines", "L_total"), n_resamples=500)
    assert agg["n_trajectories"] == 3
    assert list(agg["final_counts"]) == [1, 2, 0]
    assert np.isfinite(agg["t_nucleation"][0])
    assert not np.isfinite(agg["t_nucleation"][2])
    assert "trajectories: 3" in st.format_report(agg)


def test_control_comparison_flags_a_dirty_control():
    times = [0.0, 1.0, 2.0]
    tw = _rows([[0, 2, 3], [0, 1, 4]], times)
    clean = _rows([[0, 0, 0], [0, 0, 0]], times)
    dirty = _rows([[0, 2, 3], [0, 1, 4]], times)

    good = st.compare_to_control(tw, clean, n_resamples=500)
    assert good["control_clean"] and good["separated"]

    bad = st.compare_to_control(tw, dirty, n_resamples=500)
    assert not bad["control_clean"], "a control that nucleates must be flagged"
    assert not bad["separated"]


def test_csv_round_trip_preserves_the_numbers(tmp_path=None):
    import tempfile
    rows = _rows([[0, 1, 2]], [0.0, 0.5, 1.0])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "obs.csv")
        write_observables_csv(rows, path)
        back = read_observables_csv(path)
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        assert a["trajectory"] == b["trajectory"]
        assert abs(a["t"] - b["t"]) < 1e-12
        assert a["n_lines"] == b["n_lines"]


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} statistics tests passed")


if __name__ == "__main__":
    _main()
