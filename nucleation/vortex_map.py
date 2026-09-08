"""Three-view vortex map animation, rendered from a finished run's
pierce_points.npz alone.

Separate from visualise.py's density+winding movie because that one needs psi,
so it can only be made while a run is in flight and only for the planes it
recorded. This one needs nothing but the saved core coordinates -- a few MB
for a whole run -- so any finished run can be re-rendered afterwards, from any
viewpoint, at any frame rate, with no GPU. That is why points are stored
instead of fields.

It shows the detected cores projected along each axis as a binned core-density
map. Once a tangle has hundreds of lines a density slice is unreadable and
individual markers are a solid blob; a binned map stays legible from three
lines at t=0 to thirteen hundred at t=8.
"""
from __future__ import annotations

import os

import numpy as np

from .visualise import interpolate_frames, save_animation, sub_frames_for

# Projection axis -> (horizontal column, vertical column, label) using
# detect.py's (x, y, z) point columns.
_VIEWS = {
    "x": (1, 2, "view along x   (y-z plane)"),
    "y": (0, 2, "view along y   (x-z plane)"),
    "z": (0, 1, "view along z   (x-y plane)"),
}


def load_run(npz_path: str, trajectory: int | None = 0):
    """(times, [points per frame]) from a saved pierce_points.npz.

    `trajectory` selects ONE realisation; None keeps every stored frame.

    Selecting matters: the observer stores one frame per (trajectory, time),
    so an 8-trajectory run's arrays hold each time eight times over, in
    block-major order. Animating them unfiltered plays t0(traj0), t0(traj1),
    ... -- eight different realisations cutting between each other at every
    time step, with each time repeated eight times on the count-vs-time
    strip's x axis. Vortices nucleate in different places in every
    trajectory, so that is not a noisier version of the right picture; it is
    a different picture at every frame.

    An unknown `trajectory` falls back to the lowest one present rather than
    raising, so an old npz written before trajectories were stored still
    renders.
    """
    data = np.load(npz_path)
    off = data["frame_offset"]
    t_all = np.asarray(data["frame_t"], dtype=float)
    traj = (np.asarray(data["frame_trajectory"]) if "frame_trajectory" in data
            else np.zeros(t_all.size, dtype=int))

    keep = np.arange(t_all.size)
    if trajectory is not None and traj.size:
        sel = np.flatnonzero(traj == trajectory)
        if sel.size == 0:
            sel = np.flatnonzero(traj == traj.min())
        keep = sel
    keep = keep[np.argsort(t_all[keep], kind="stable")]
    pts = [data["points"][off[i]:off[i + 1]] for i in keep]
    return t_all[keep], pts


def frame_histograms(points_per_frame, extent, bins: int, views=("x", "y", "z")):
    """Core-density map per view per frame, as float32 (frames, bins, bins)."""
    edges = np.linspace(extent[0], extent[1], bins + 1)
    out = {}
    for v in views:
        h, w, _ = _VIEWS[v]
        stack = np.empty((len(points_per_frame), bins, bins), dtype=np.float32)
        for i, p in enumerate(points_per_frame):
            if p.shape[0]:
                # histogram2d's first axis is the FIRST argument, and imshow
                # draws the first axis vertically -- so pass vertical first.
                stack[i] = np.histogram2d(p[:, w], p[:, h], bins=(edges, edges))[0]
            else:
                stack[i] = 0.0
        out[v] = stack
    return out


def save_vortex_map(npz_path: str, out_path: str, extent=(-8.0, 8.0), bins: int = 128,
                     fps: int = 60, duration: float | None = None,
                     views=("x", "y", "z"), gamma: float = 0.5,
                     clip_percentile: float = 99.5, dpi: int = 90,
                     counts_panel: bool = True, trajectory: int | None = 0) -> str:
    """Render the animation.

    duration -- playback seconds. Default: the run's own simulated time, so
                the movie runs at 1x like run.py's gifs. Sub-frames are
                chosen to hit `fps` over that duration.
    gamma    -- power-law colour scale. 0.5 keeps three vortices at t=0
                visible on the same fixed scale as thirteen hundred at t=8;
                a linear scale renders the whole first half black.
    trajectory -- which realisation to animate (see load_run); None means
                all stored frames, which only makes sense for a single-
                trajectory run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import PowerNorm

    times, pts = load_run(npz_path, trajectory)
    counts = np.array([p.shape[0] for p in pts], dtype=float)
    hists = frame_histograms(pts, extent, bins, views)

    if times.size < 1:
        raise ValueError(f"{npz_path} holds no frames for trajectory {trajectory}")
    span = float(times[-1] - times[0]) or 1.0
    sub = sub_frames_for(len(times), fps, duration if duration else span)
    frames = {v: interpolate_frames(hists[v], sub) for v in views}
    t_frames = interpolate_frames(times, sub)
    c_frames = interpolate_frames(counts, sub)
    n_frames = len(t_frames)

    nonzero = np.concatenate([frames[v][frames[v] > 0].ravel() for v in views])
    vmax = float(np.percentile(nonzero, clip_percentile)) if nonzero.size else 1.0
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1.0))

    rows = 2 if counts_panel else 1
    fig = plt.figure(figsize=(4.1 * len(views), 5.9 if counts_panel else 4.7),
                     layout="constrained")
    gs = fig.add_gridspec(rows, len(views), height_ratios=([4, 1] if counts_panel else [1]))
    ext = (extent[0], extent[1], extent[0], extent[1])

    ims = []
    for k, v in enumerate(views):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(frames[v][0], origin="lower", cmap="viridis", norm=norm,
                       extent=ext, interpolation="bilinear")
        h, w, label = _VIEWS[v]
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("xyz"[h])
        ax.set_ylabel("xyz"[w])
        ax.set_aspect("equal")
        ims.append(im)

    marker = trace = None
    if counts_panel:
        # Log scale: the count runs from a handful of cores at onset to ~1e5
        # in the developed tangle, and on a linear axis the entire nucleation
        # event -- the part worth watching -- is a flat line against the axis.
        axc = fig.add_subplot(gs[1, :])
        axc.plot(times, np.maximum(counts, 0.9), color="#2b6cb0", lw=1.6)
        axc.fill_between(times, np.maximum(counts, 0.9), 0.9,
                         color="#2b6cb0", alpha=0.15)
        marker = axc.axvline(times[0], color="#b2182b", lw=1.4)
        trace, = axc.plot([times[0]], [max(counts[0], 0.9)], "o", color="#b2182b", ms=5)
        axc.set_yscale("log")
        axc.set_xlim(times[0], times[-1])
        axc.set_ylim(0.9, counts.max() * 2.0 or 10.0)
        axc.set_xlabel("t")
        axc.set_ylabel("detected cores")
        axc.spines[["top", "right"]].set_visible(False)

    suptitle = fig.suptitle("", fontsize=12)

    def update(i):
        for k, v in enumerate(views):
            ims[k].set_data(frames[v][i])
        suptitle.set_text(f"t = {t_frames[i]:.3f}     cores = {c_frames[i]:,.0f}")
        artists = list(ims) + [suptitle]
        if counts_panel:
            marker.set_xdata([t_frames[i], t_frames[i]])
            trace.set_data([t_frames[i]], [max(c_frames[i], 0.9)])
            artists += [marker, trace]
        return artists

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps)
    return save_animation(anim, fig, out_path, fps, dpi)


# ---------------------------------------------------------------------------
# CLI: re-render any finished run without touching the GPU
#   python -m nucleation.vortex_map outputs/nucleation_collision/pierce_points.npz \
#          outputs/nucleation_collision/vortex_map.gif --half-box 8 --fps 60
# ---------------------------------------------------------------------------

def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", help="a run's pierce_points.npz")
    ap.add_argument("out", help="output .gif or .mp4")
    ap.add_argument("--half-box", type=float, default=8.0,
                    help="L/2 of the run (config.py's L divided by two)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--duration", type=float, default=None,
                    help="playback seconds; default = the run's simulated time (1x)")
    ap.add_argument("--bins", type=int, default=160)
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--views", default="xyz", help="which projections, e.g. 'xz'")
    ap.add_argument("--no-counts", action="store_true", help="drop the count-vs-time strip")
    ap.add_argument("--trajectory", type=int, default=0,
                    help="which trajectory to animate (-1 = all stored frames)")
    a = ap.parse_args(argv)
    path = save_vortex_map(a.npz, a.out, extent=(-a.half_box, a.half_box), bins=a.bins,
                            fps=a.fps, duration=a.duration, views=tuple(a.views),
                            dpi=a.dpi, counts_panel=not a.no_counts,
                            trajectory=None if a.trajectory < 0 else a.trajectory)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
