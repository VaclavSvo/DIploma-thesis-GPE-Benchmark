"""Glue between run.py and the rest of this package: build the observer, write
a run's outputs, print its report. Out of run.py so that stays a thin scenario
switch, and out of the detection modules so those hold no opinions about file
layout.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import io, statistics as st
from .observer import NucleationObserver
from .settings import DetectorConfig
from .visualise import save_density_winding_gif, save_momentum_gif
from .vortex_map import save_vortex_map


def healing_length(g: float, n_peak: float) -> float:
    """xi = 1/sqrt(2*g*n) in these natural units (hbar=m=1).

    From balancing the quantum pressure against the interaction energy,
    hbar^2/(2*m*xi^2) = g*n (Pitaevskii & Stringari, "Bose-Einstein
    Condensation", Sec. 5.2; Pethick & Smith, Sec. 6.2). The core is a couple
    of xi across, so dx must be a fraction of it or the windings stop being
    integers.

    This used to return 1/sqrt(g*n) -- the OTHER convention, larger by
    sqrt(2) -- while gpe3d/noise.py's cutoff is built on 1/sqrt(2*g*n). Two
    conventions in one project made run.py's printed dx/xi optimistic by
    sqrt(2), so a grid at dx = 0.4*xi_true reported 0.28 and passed the
    "cores resolved" line. Both now mean the same thing.
    """
    return float("inf") if g * n_peak <= 0 else 1.0 / math.sqrt(2.0 * g * n_peak)


def resolution_note(engine, g: float, n_peak: float) -> dict:
    xi = healing_length(g, n_peak)
    ratio = engine.dx / xi if math.isfinite(xi) and xi > 0 else float("inf")
    return dict(healing_length=xi, dx=engine.dx, dx_over_xi=ratio,
                resolved=bool(ratio <= 1.0 / 3.0))


def make_observer(solver, cfg: DetectorConfig,
                  final_block: int | None = None) -> NucleationObserver:
    """Observer wired to this solver's engine and noise offset.

    The offset (a TW solver's vacuum+thermal ordering correction) is
    subtracted from the density before masking; without it the noise's own
    uniform "particle" floor sits under the mask threshold and drags n_peak
    around. Classical solvers have no such attribute and get 0.
    """
    offset = getattr(solver, "offset_density", None) or 0.0
    return NucleationObserver(solver.engine, cfg, density_offset=float(offset),
                              final_block=final_block)


def write_outputs(observer: NucleationObserver, out_dir: str, meta: dict,
                  gif_fps: int = 60, box_extent=None, duration: float | None = None,
                  vortex_map: bool = True) -> dict:
    """CSV of observables + npz of pierce points + metadata + two movies. No
    fields: psi and the winding volumes are hundreds of MB each at production
    N, and both movies re-render from the points and slices alone.

    The movies answer different questions and both are cheap:
      density_winding.gif  the close-up -- what is happening on the recorded
                           mid-planes, with the sign of every core.
      vortex_map.gif       the overview -- where the cores are in the whole
                           box, down each axis, plus the count against time.
                           The one that stays readable once a tangle has
                           hundreds of lines in it.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = observer.summary_table()
    npz_path = os.path.join(out_dir, "pierce_points.npz")
    paths = {
        "observables": io.write_observables_csv(rows, os.path.join(out_dir, "observables.csv")),
        "pierce_points": io.write_pierce_points(observer.pierce_arrays(), npz_path),
        "metadata": io.write_run_metadata(meta, os.path.join(out_dir, "run_metadata.json")),
    }
    if observer.times:
        paths["gif"] = save_density_winding_gif(
            observer.slices, observer.times, observer.cfg.slice_planes,
            os.path.join(out_dir, "density_winding.gif"),
            fps=gif_fps, extent=box_extent, duration=duration,
            log_density=observer.cfg.density_log,
            log_floor_frac=observer.cfg.density_log_floor)
    if observer.times and observer.cfg.momentum_planes:
        # The halo's own view: a shell in k-space, cut as a ring by the plane
        # holding the collision axis. Rendered from stored (N,N) frames, so
        # like the other two movies it costs no GPU time to redo.
        paths["momentum"] = save_momentum_gif(
            observer.slices, observer.times, observer.cfg.momentum_planes,
            os.path.join(out_dir, "momentum.gif"),
            fps=gif_fps, k_extent=observer.momentum_extent, duration=duration,
            floor=observer.cfg.momentum_floor,
            vmax=observer.cfg.momentum_vmax,
            ring=observer.cfg.momentum_ring)
    if vortex_map and observer.frames:
        # One trajectory, like density_winding.gif and for the same reason:
        # the stored frames interleave every trajectory on the same time grid,
        # so animating them all would cut between realisations frame by frame.
        paths["vortex_map"] = save_vortex_map(
            npz_path, os.path.join(out_dir, "vortex_map.gif"),
            extent=box_extent or (-1.0, 1.0), fps=gif_fps, duration=duration,
            trajectory=observer.cfg.slice_trajectory)
    return paths


def report(observer: NucleationObserver) -> str:
    """Text summary for the run log: vortex counts, line length, onsets, and
    the two things that decide whether any of it is believable (integrality
    of the windings, and how many trajectories nucleated at all)."""
    rows = observer.summary_table()
    if not rows:
        return "no frames recorded"
    agg = st.aggregate(rows, observables=("n_lines", "L_total"), n_resamples=1000)
    lines = [st.format_report(agg, "n_lines")]
    L = agg["L_total"]
    lines.append(f"L_total: final {L['mean'][-1]:.3f}  95% CI "
                 f"[{L['ci_lo'][-1]:.3f}, {L['ci_hi'][-1]:.3f}]  peak {L['mean'].max():.3f}")
    worst = max(r["integrality_error"] for r in rows)
    verdict = "OK" if worst < 0.05 else "WARNING: cores are marginally resolved"
    lines.append(f"max winding integrality error: {worst:.3g} -- {verdict}")
    return "\n".join(lines)


def compare_runs(tw_csv: str, control_csv: str, key: str = "n_lines") -> str:
    """Post-hoc TW vs mean-field control comparison from two saved runs.

    The control is the check that matters: identical parameters with the
    noise switched off must give zero. If it does not, the detector is firing
    on something that is not a vortex and the TW numbers mean nothing.
    """
    cmp = st.compare_to_control(io.read_observables_csv(tw_csv),
                                io.read_observables_csv(control_csv), key=key)
    verdict = ("control is clean" if cmp["control_clean"]
               else f"CONTROL NOT CLEAN (peak {cmp['control_max']:.2f}) -- detector suspect")
    sep = ("TW separates from the control outside the CI"
           if cmp["separated"] else "TW does NOT separate from the control")
    return (f"final TW {key}: {cmp['tw_mean'][-1]:.2f} "
            f"[{cmp['tw_lo'][-1]:.2f}, {cmp['tw_hi'][-1]:.2f}]   "
            f"control {cmp['control_mean'][-1]:.2f}\n{verdict}\n{sep}")


def sweep_mask_threshold(psi, engine, cfg: DetectorConfig, thresholds, **detect_kw) -> list[dict]:
    """Re-detect one frozen field at several mask thresholds. The reported
    count must be stable across this sweep; if it is not, the number says where
    the mask was drawn, not how many vortices there are. Costs no GPU time.
    """
    from .detect import detect_frame
    out = []
    for f in thresholds:
        frame = detect_frame(psi, engine, t=0.0, trajectory=0,
                             cfg=cfg.with_overrides(mask_threshold=float(f)), **detect_kw)
        s = frame.summary()
        out.append(dict(mask_threshold=float(f), n_lines=s["n_lines"],
                        n_pierce=s["n_pierce"], L_total=s["L_total"]))
    return out


def format_sweep(rows: list[dict]) -> str:
    header = f"{'f':>8} {'n_lines':>9} {'n_pierce':>9} {'L_total':>10}"
    body = "\n".join(f"{r['mask_threshold']:8.3f} {r['n_lines']:9d} "
                     f"{r['n_pierce']:9d} {r['L_total']:10.3f}" for r in rows)
    counts = np.array([r["n_lines"] for r in rows], dtype=float)
    spread = (counts.max() - counts.min()) / max(counts.mean(), 1e-12)
    verdict = ("stable" if spread < 0.25
               else "NOT STABLE -- the count depends on where the mask is drawn")
    return f"{header}\n{body}\nrelative spread {spread:.1%} -- {verdict}"
