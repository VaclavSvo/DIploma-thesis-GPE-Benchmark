"""The per-frame recorder that turns an evolving wavefunction into vortex
numbers. Both evolution loops (gpe3d/evolve.py, gpe3d/ensemble.py) take an
optional `observer`; that hook was the only change either needed, so a
nucleation run is gated exactly like any other.

It works per trajectory. The ensemble runner's density output is a mean over
trajectories, and vortices nucleate in different places in every one, so that
mean is a smooth blob with no cores in it. Detection therefore happens on each
trajectory's own field before averaging, and the movie follows one trajectory.
"""
from __future__ import annotations

import numpy as np

from gpe3d.backend import xp, to_numpy
from . import winding as W
from .detect import detect_frame


def plane_index(engine, normal: str, coord: float = 0.0) -> int:
    """Grid index of the plane nearest `coord` along `normal`.

    Not simply N//2: the grid is linspace(-L/2, L/2, N) with both endpoints
    included, so for even N no sample sits exactly at 0 and N//2 is half a
    cell off. This picks the genuinely nearest plane.
    """
    x0, dx = -engine.L / 2.0, engine.dx
    return int(min(engine.N - 1, max(0, round((coord - x0) / dx))))


class NucleationObserver:
    """Collects vortex detections and movie slices as a run proceeds.

    Attributes after a run:
      frames  -- list[VortexFrame], one per (trajectory, recorded time)
      slices  -- dict: (normal, "density"|"winding") -> list of 2D arrays,
                 in recorded-frame order, for the followed trajectory
      times   -- list[float], the times those slices belong to
    """

    def __init__(self, engine, cfg, density_offset: float = 0.0,
                 final_block: int | None = None):
        """final_block -- index of the run's last block, if known. Detection
        always fires there regardless of detect_stride: the final count and
        line length are what the run reports, and with an even stride and an
        odd block count they would otherwise come from the second-to-last
        frame. Same reasoning as gpe3d/ensemble.py forcing fresh diagnostics
        on its last block."""
        self.engine = engine
        self.cfg = cfg
        self.final_block = final_block
        self.density_offset = float(density_offset)
        self.frames = []
        self.times = []
        self.slices = {}
        self._plane_idx = {n: plane_index(engine, n) for n in cfg.slice_planes}

    # -- the hook the evolution loops call ---------------------------------
    def on_frame(self, state, block_idx: int, traj_offset: int = 0) -> None:
        psi = state.psi
        batch = psi if psi.ndim == 4 else psi[None, ...]

        if block_idx % self.cfg.detect_stride == 0 or block_idx == self.final_block:
            for b in range(batch.shape[0]):
                self.frames.append(detect_frame(
                    batch[b], self.engine, t=state.t, trajectory=traj_offset + b,
                    cfg=self.cfg, density_offset=self.density_offset))

        follow = self.cfg.slice_trajectory - traj_offset
        if (0 <= follow < batch.shape[0]) and block_idx % self.cfg.slice_stride == 0:
            self._record_slices(batch[follow], state.t)

    # -- movie slices -------------------------------------------------------
    def _record_slices(self, psi_traj, t: float) -> None:
        """Density and winding on each requested plane, masked the same way
        the counter masks.

        The masking is not cosmetic. An unmasked winding panel lights up
        wherever the density is near zero -- the phase there is numerical
        noise, and in a TW run it is genuine vacuum noise -- so the picture
        would show cores the count does not include, and the two outputs of
        the same run would disagree. Same threshold, same closing, same
        plaquette rule as nucleation/detect.py, applied in 2D.
        """
        self.times.append(float(t))
        cfg = self.cfg
        for normal in cfg.slice_planes:
            psi2d = W.slice_plane(psi_traj, normal, self._plane_idx[normal])
            dens2d = self.engine._density(psi2d) - self.density_offset
            axes = W.slice_axes(normal)

            mask = W.density_mask(dens2d, float(xp.max(dens2d)), cfg.mask_threshold,
                                  cfg.mask_close_passes, axes=axes)
            w2d = xp.rint(W.winding_slice(psi2d, normal))
            w2d = xp.where(W.plaquette_mask(mask, *axes), w2d, 0)

            # float16 density / int8 winding: display-only arrays, and a
            # 3-plane movie at N=384 would otherwise run to hundreds of MB.
            self.slices.setdefault((normal, "density"), []).append(
                to_numpy(dens2d).astype(np.float16))
            self.slices.setdefault((normal, "winding"), []).append(
                to_numpy(w2d).astype(np.int8))

    # -- results ------------------------------------------------------------
    def summary_table(self) -> list[dict]:
        """One row per (trajectory, time), sorted -- ready for CSV or for
        nucleation/statistics.py."""
        rows = [f.summary() for f in self.frames]
        rows.sort(key=lambda r: (r["trajectory"], r["t"]))
        return rows

    def pierce_arrays(self) -> dict:
        """Every frame's pierce points packed into flat arrays plus an index,
        for np.savez_compressed. Kilobytes per frame -- unlike psi or the
        winding fields, which are never stored."""
        counts = np.array([f.points.shape[0] for f in self.frames], dtype=np.int64)
        return dict(
            frame_t=np.array([f.t for f in self.frames], dtype=np.float32),
            frame_trajectory=np.array([f.trajectory for f in self.frames], dtype=np.int32),
            frame_offset=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            points=(np.concatenate([f.points for f in self.frames])
                    if self.frames else np.empty((0, 3), np.float32)),
            charges=(np.concatenate([f.charges for f in self.frames])
                     if self.frames else np.empty(0, np.int8)),
            normals=(np.concatenate([f.normals for f in self.frames])
                     if self.frames else np.empty(0, np.uint8)),
        )
