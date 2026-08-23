"""The density + winding movie, plus the frame-rate and animation-writing
helpers nucleation/vortex_map.py also uses.

One column per recorded plane: |psi|^2 on top, the plaquette winding on the
same plane below, with every detected core ringed and coloured by the sign of
its circulation. That sign is information a density slice does not carry -- a
dark spot in |psi|^2 is equally consistent with a sound pulse or a grey
soliton.

Single trajectory throughout, never an ensemble average: averaging is exactly
what destroys this signal.

The detector runs at the block cadence, far too coarse to watch (60 frames
over T=8 plays at 7.5 fps), so recorded frames are cross-faded up to the
requested fps at render time. Playback length equals the simulated time, so
the movie runs at 1x like run.py's own gifs.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

# Axis labels for the two in-plane directions of each plane family, in
# (horizontal, vertical) order as imshow lays the array out -- the array's
# last axis runs horizontally. Matches nucleation/winding.py's slice_plane.
_PLANE_AXES = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}

POSITIVE_COLOUR = "#e05c4a"
NEGATIVE_COLOUR = "#4a86e0"


# ---------------------------------------------------------------------------
# Frame-rate helpers
# ---------------------------------------------------------------------------

def sub_frames_for(n_recorded: int, fps: float, duration: float) -> int:
    """How many render frames per recorded frame to hit `fps` over `duration`
    seconds of playback."""
    if n_recorded < 2:
        return 1
    return max(1, round(fps * duration / (n_recorded - 1)))


def _render_positions(n: int, sub: int):
    """Fractional recorded-frame index of every rendered frame."""
    return np.linspace(0, n - 1, (n - 1) * sub + 1)


def interpolate_frames(stack, sub: int):
    """Cross-fade `sub-1` frames between every recorded pair. Works on a 1D
    series (times, counts) or a stack of 2D frames.

    This is a blend, not a physical interpolation of the flow: every frame
    carrying real data is one of the originals, and the ones between exist so
    the eye can follow a core instead of watching it teleport.
    """
    stack = np.asarray(stack, dtype=np.float32 if np.ndim(stack) > 1 else float)
    if sub <= 1 or stack.shape[0] < 2:
        return stack
    pos = _render_positions(stack.shape[0], sub)
    lo = np.clip(np.floor(pos).astype(int), 0, stack.shape[0] - 1)
    hi = np.clip(lo + 1, 0, stack.shape[0] - 1)
    frac = (pos - lo).astype(stack.dtype).reshape(-1, *([1] * (stack.ndim - 1)))
    return stack[lo] * (1.0 - frac) + stack[hi] * frac


def hold_frames(stack, sub: int):
    """Nearest recorded frame for every rendered frame -- no blending. For
    quantities where an intermediate value is meaningless, i.e. the integer
    winding."""
    stack = np.asarray(stack)
    if sub <= 1 or stack.shape[0] < 2:
        return stack
    idx = np.clip(np.rint(_render_positions(stack.shape[0], sub)).astype(int),
                  0, stack.shape[0] - 1)
    return stack[idx]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _mp4_to_gif(src: str, dst: str, fps: int) -> None:
    """Convert with one shared, optimised palette.

    matplotlib's PillowWriter quantises each frame independently against a
    fixed palette: on a viridis field that means visible banding, colour
    crawl between frames, and a larger file. A generated palette fixes all
    three.
    """
    palette = dst[:-4] + ".palette.png"
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    subprocess.run(base + ["-vf", f"fps={fps},palettegen=stats_mode=diff", palette],
                   check=True)
    subprocess.run(base + ["-i", palette, "-lavfi",
                           f"fps={fps}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
                           dst], check=True)
    os.remove(palette)


def save_animation(anim, fig, out_path: str, fps: int, dpi: int):
    """Write a FuncAnimation to .gif or .mp4, using ffmpeg when it's there."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    want_gif = out_path.lower().endswith(".gif")

    if want_gif and _have_ffmpeg():
        tmp = out_path[:-4] + ".tmp.mp4"
        anim.save(tmp, writer=animation.FFMpegWriter(fps=fps, bitrate=6000), dpi=dpi)
        plt.close(fig)
        _mp4_to_gif(tmp, out_path, fps)
        os.remove(tmp)
        return out_path

    writer = (animation.PillowWriter(fps=fps) if want_gif
              else animation.FFMpegWriter(fps=fps, bitrate=6000))
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# The movie
# ---------------------------------------------------------------------------

def save_density_winding_gif(slices: dict, times, planes, path: str, fps: int = 60,
                              extent=None, dens_percentile: float = 99.5,
                              duration: float | None = None, dpi: int = 90,
                              marker_size: float = 90.0) -> str:
    """Write the movie.

    slices   -- NucleationObserver.slices: (normal, "density"|"winding") -> list
                of 2D arrays, one per recorded time.
    planes   -- plane normals to show, one figure column each.
    extent   -- (lo, hi) physical box coordinate for the axes; None = indices.
    duration -- playback seconds; default is the run's own simulated time (1x).

    Returns the path written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    planes = [p for p in planes if (p, "density") in slices]
    if not planes:
        raise ValueError("no recorded slices to animate -- check cfg.slice_planes")

    times = np.asarray(times, dtype=float)
    span = float(times[-1] - times[0]) if times.size > 1 else 1.0
    sub = sub_frames_for(times.size, fps, duration if duration else span)

    dens = {p: interpolate_frames(slices[(p, "density")], sub) for p in planes}
    # Winding is topological: a blended half-integer is meaningless, so the
    # nearest recorded frame is held instead. Cores step while the density
    # flows, which is the honest rendering of an integer-valued quantity.
    wind = {p: hold_frames(slices[(p, "winding")], sub) for p in planes}
    t_frames = interpolate_frames(times, sub)
    n_frames = len(t_frames)

    vmax = float(np.percentile(np.concatenate([d.ravel() for d in dens.values()]),
                               dens_percentile)) or 1.0
    ext = None if extent is None else (extent[0], extent[1], extent[0], extent[1])
    shape = dens[planes[0]][0].shape

    def marker_coords(w):
        """Nonzero plaquettes as (x, y, sign) in the panel's coordinates.

        A vortex occupies ONE plaquette, so at production grid sizes it is a
        single pixel in a 400px panel -- invisible. The image carries the
        field; these markers are what make the cores findable by eye, which is
        the whole point of the figure.
        """
        rows, cols = np.nonzero(w)
        signs = w[rows, cols]
        if extent is None:
            return cols.astype(float), rows.astype(float), signs
        lo, hi = extent
        return (lo + (cols + 0.5) * (hi - lo) / shape[1],
                lo + (rows + 0.5) * (hi - lo) / shape[0], signs)

    # Planes across, quantities down: density on the top row, winding on the
    # bottom. Landscape rather than one tall column per plane -- three planes
    # stacked vertically is 13 inches of figure that fits nothing.
    fig = plt.figure(figsize=(4.2 * len(planes), 8.9), layout="constrained")
    axes = fig.subplots(2, len(planes), squeeze=False)
    panels = []
    for col, p in enumerate(planes):
        h, v = _PLANE_AXES[p]
        ax_d, ax_w = axes[0][col], axes[1][col]
        im_d = ax_d.imshow(dens[p][0], origin="lower", cmap="viridis",
                           vmin=0.0, vmax=vmax, extent=ext, interpolation="bilinear")
        im_w = ax_w.imshow(wind[p][0].astype(np.float32), origin="lower", cmap="coolwarm",
                           vmin=-1.5, vmax=1.5, interpolation="nearest", extent=ext)
        dots = ax_w.scatter([], [], s=marker_size, marker="o", facecolors="none",
                            linewidths=1.7)
        ax_d.set_title(f"$|\\psi|^2$      {p} = 0", fontsize=11)
        title_w = ax_w.set_title("", fontsize=11)
        for ax in (ax_d, ax_w):
            ax.set_xlabel(h)
            ax.set_ylabel(v)
            ax.set_aspect("equal")
        panels.append((p, im_d, im_w, dots, title_w))

    suptitle = fig.suptitle("", fontsize=14)

    def update(i):
        artists = [suptitle]
        for p, im_d, im_w, dots, title_w in panels:
            w = wind[p][i]
            im_d.set_data(dens[p][i])
            im_w.set_data(w.astype(np.float32))
            x, y, signs = marker_coords(w)
            dots.set_offsets(np.column_stack([x, y]) if x.size else np.empty((0, 2)))
            dots.set_edgecolors([POSITIVE_COLOUR if s > 0 else NEGATIVE_COLOUR
                                 for s in signs])
            title_w.set_text(f"winding      {p} = 0      {signs.size} "
                             f"core{'' if signs.size == 1 else 's'}")
            artists += [im_d, im_w, dots, title_w]
        suptitle.set_text(f"t = {t_frames[i]:.3f}")
        return artists

    update(0)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps)
    return save_animation(anim, fig, path, fps, dpi)
