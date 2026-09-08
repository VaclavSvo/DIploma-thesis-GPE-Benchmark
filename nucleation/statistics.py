"""Aggregation across trajectories.

Counts here are small integers and roughly Poisson, and a nucleation run
realistically has tens of trajectories, not thousands -- so "mean +- SEM" is
the wrong interval and is not offered. Bootstrap percentile intervals are
used instead: they make no distributional assumption and degrade honestly
when the sample is small.

Everything in this module works on the plain row-dicts that
NucleationObserver.summary_table() produces, so a run's numbers can be
re-aggregated later from the saved CSV without re-running anything.
"""
from __future__ import annotations

import numpy as np

DEFAULT_OBSERVABLES = ("n_lines", "L_total", "n_pierce")


def to_matrix(rows: list[dict], key: str):
    """Rows -> (times, values[n_trajectories, n_times]).

    Raises if the trajectories were not all recorded on the same time grid,
    which would otherwise average silently misaligned frames.
    """
    trajectories = sorted({r["trajectory"] for r in rows})
    times = sorted({round(float(r["t"]), 9) for r in rows})
    values = np.full((len(trajectories), len(times)), np.nan)
    t_index = {t: i for i, t in enumerate(times)}
    traj_index = {tr: i for i, tr in enumerate(trajectories)}
    for r in rows:
        values[traj_index[r["trajectory"]], t_index[round(float(r["t"]), 9)]] = r[key]
    if np.isnan(values).any():
        raise ValueError(f"'{key}' is missing for some (trajectory, time) pairs -- "
                         "the trajectories were not recorded on a common time grid")
    return np.asarray(times), values


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, ci: float = 0.95,
                 seed: int = 0):
    """Percentile bootstrap CI of the mean over axis 0.

    Returns (mean, lo, hi), each of length values.shape[1].
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    alpha = 0.5 * (1.0 - ci)
    lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)], axis=0)
    return values.mean(axis=0), lo, hi


def nucleation_onset(times: np.ndarray, counts: np.ndarray, threshold: int = 1,
                     sustain: int = 3):
    """First time the count reaches `threshold` and stays there for `sustain`
    consecutive frames, per trajectory. NaN if it never does.

    The sustain requirement is what separates a nucleation event from a
    single-frame detector twitch; with sustain=1 the onset histogram is
    dominated by the earliest false positive in each trajectory.
    """
    onset = np.full(counts.shape[0], np.nan)
    hits = counts >= threshold
    for i, row in enumerate(hits):
        for j in range(row.size - sustain + 1):
            if row[j:j + sustain].all():
                onset[i] = times[j]
                break
    return onset


def aggregate(rows: list[dict], observables=DEFAULT_OBSERVABLES,
              n_resamples: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Ensemble summary of one run: mean and bootstrap CI per observable,
    plus onset times and the distribution of the final count."""
    observables = tuple(observables)
    if not observables:
        raise ValueError("aggregate: need at least one observable")
    out = {"n_trajectories": len({r["trajectory"] for r in rows})}
    times = None
    for key in observables:
        times, values = to_matrix(rows, key)
        mean, lo, hi = bootstrap_ci(values, n_resamples, ci, seed)
        out[key] = dict(mean=mean, ci_lo=lo, ci_hi=hi, per_trajectory=values)
    # Onset and final counts come from "n_lines" when it was asked for, and
    # otherwise from the first observable. Hard-coding "n_lines" here made
    # aggregate() -- and through it compare_to_control(key=...) -- raise
    # KeyError for every observable except that one.
    count_key = "n_lines" if "n_lines" in observables else observables[0]
    counts = out[count_key]["per_trajectory"]
    out["t"] = times
    out["count_key"] = count_key
    out["t_nucleation"] = nucleation_onset(times, counts)
    out["final_counts"] = counts[:, -1]
    return out


def compare_to_control(tw_rows: list[dict], control_rows: list[dict],
                       key: str = "n_lines", **kw) -> dict:
    """The actual result: TW against the noiseless mean-field control.

    The control matters as much as the signal. Run identically-parameterised
    mean-field dynamics through the same detector; it must give ~0. If it
    does not, the detector is firing on something that is not a vortex and
    every TW number is suspect -- so `control_clean` below, not the TW count,
    is what decides whether the run means anything.
    """
    tw = aggregate(tw_rows, observables=(key,), **kw)
    ctrl = aggregate(control_rows, observables=(key,), **kw)
    control_max = float(np.max(ctrl[key]["mean"]))
    return dict(
        t=tw["t"],
        tw_mean=tw[key]["mean"], tw_lo=tw[key]["ci_lo"], tw_hi=tw[key]["ci_hi"],
        control_mean=ctrl[key]["mean"],
        control_max=control_max,
        control_clean=control_max < 0.5,
        separated=bool(np.any(tw[key]["ci_lo"] > control_max)),
    )


def format_report(agg: dict, observable: str = "n_lines") -> str:
    """Compact text summary for the run log."""
    t = agg["t"]
    mean, lo, hi = (agg[observable][k] for k in ("mean", "ci_lo", "ci_hi"))
    onset = agg["t_nucleation"]
    n_onset = int(np.isfinite(onset).sum())
    lines = [
        f"trajectories: {agg['n_trajectories']}",
        f"{observable}: final {mean[-1]:.2f}  95% CI [{lo[-1]:.2f}, {hi[-1]:.2f}]  "
        f"peak {mean.max():.2f} at t={t[int(np.argmax(mean))]:.3f}",
        f"nucleated in {n_onset}/{len(onset)} trajectories"
        + (f", median onset t={np.nanmedian(onset):.3f}" if n_onset else ""),
        f"final per-trajectory counts: {np.asarray(agg['final_counts'], dtype=int).tolist()}",
    ]
    return "\n".join(lines)
