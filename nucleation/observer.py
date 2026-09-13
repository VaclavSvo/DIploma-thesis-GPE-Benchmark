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

from gpe3d.backend import xp, to_numpy, fftn
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


def momentum_plane(engine, psi_traj, normal: str):
    """Mode populations |alpha_k|^2 on the k-plane through k_<normal> = 0.

    The paper's Figs. 1-2 (Norrie/Ballagh/Gardiner, PRA 73, 043617) are exactly
    this: the scattering halo is a SHELL in momentum space, so a k-plane cuts it
    as a ring, while in coordinate space it is a diffuse glow spread under two
    bright clouds. It is the view the halo is actually visible in.

    Computed without ever building the full (N,N,N) transform. Setting one
    component to zero kills that exponential, so the 3D transform restricted to
    k_n = 0 is the 2D transform of psi summed along the real-space axis n:

        sum_r psi(r) e^{-i(k_u u + k_v v)} = FFT2[ sum_n psi ](k_u, k_v)

    -- a reduction plus one (N,N) transform, instead of an (N,N,N) complex
    temporary and a full 3D FFT, per plane per recorded frame.

    Returned in mode-population units, not raw FFT magnitude: on the orthonormal
    plane-wave basis alpha_k = (dV/sqrt(V)) * FFT[psi]_k, so
    |alpha_k|^2 = (dV/N^3) * |FFT|^2, which sums to the atom number. That makes
    the numbers directly comparable with the paper's (3.2e4 for a mode at the
    centre of a condensate packet; 1/2 per mode for the Truncated-Wigner vacuum,
    which is the floor the halo grows out of).

    fftshift'ed, so k = 0 sits at the centre of the array. The surviving axes
    keep their original order, i.e. the same (horizontal, vertical) convention
    visualise._PLANE_AXES uses for the coordinate-space panels.
    """
    axis = W._NORMAL_AXIS[normal]
    projected = psi_traj.sum(axis=axis)
    amp = fftn(projected, axes=(-2, -1))
    dens = amp.real ** 2 + amp.imag ** 2
    dens *= engine.dV / (engine.N ** 3)
    return xp.fft.fftshift(dens, axes=(-2, -1))


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

        # Momentum panels: one shared crop window (the grid is cubic, so both
        # transverse axes use it) and the k limits that go with it. Cropping
        # here rather than at render time also keeps the stored frames small.
        self.momentum_extent = None
        self._k_slice = None
        if cfg.momentum_planes:
            k = np.fft.fftshift(np.fft.fftfreq(engine.N, d=engine.dx)) * 2.0 * np.pi
            k_max = cfg.momentum_k_max or float(np.abs(k).max())
            idx = np.nonzero(np.abs(k) <= k_max)[0]
            if idx.size < 2:
                raise ValueError(f"momentum_k_max={cfg.momentum_k_max} keeps only "
                                 f"{idx.size} modes -- raise it or lower N")
            self._k_slice = slice(int(idx[0]), int(idx[-1]) + 1)
            self.momentum_extent = (float(k[idx[0]]), float(k[idx[-1]]))

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
            # n_peak must be the FOLLOWED TRAJECTORY'S 3D peak, the same
            # reference detect_frame() masks against -- see _record_slices.
            # Reuse it from this block's detection when there was one.
            n_peak = None
            for f in reversed(self.frames):
                if f.t == state.t and f.trajectory == self.cfg.slice_trajectory:
                    n_peak = f.n_peak
                    break
            if n_peak is None:
                # max(dens - c) == max(dens) - c, so the offset needs no
                # full-grid subtraction of its own.
                n_peak = float(xp.max(self.engine._density(batch[follow]))) - self.density_offset
            self._record_slices(batch[follow], state.t, n_peak)

    # -- movie slices -------------------------------------------------------
    def _record_slices(self, psi_traj, t: float, n_peak: float) -> None:
        """Density and winding on each requested plane, masked against the
        same threshold the counter uses.

        The masking is not cosmetic. An unmasked winding panel lights up
        wherever the density is near zero -- the phase there is numerical
        noise, and in a TW run it is genuine vacuum noise -- so the picture
        would show cores the count does not include, and the two outputs of
        the same run would disagree.

        `n_peak` is therefore the trajectory's 3D peak density, passed in by
        on_frame(). It used to be this plane's own max, which is a different
        (smaller) number on every plane that does not cut the densest part of
        the cloud -- so the panel's threshold was mask_threshold * n_peak_plane
        rather than mask_threshold * n_peak_volume, i.e. strictly more
        permissive than the counter exactly where the cloud is thin.

        What still differs, deliberately: the counter also requires a 3D local
        density dip (cfg.require_density_dip), which has no 2D equivalent that
        is the same criterion rather than a new one. The panel is therefore an
        UPPER BOUND on what the counter accepts -- a ringed core in the movie
        may have failed the dip test in 3D. The count in observables.csv is
        the number, the panel is where it is happening.
        """
        self.times.append(float(t))
        cfg = self.cfg
        for normal in cfg.slice_planes:
            psi2d = W.slice_plane(psi_traj, normal, self._plane_idx[normal])
            dens2d = self.engine._density(psi2d) - self.density_offset
            axes = W.slice_axes(normal)

            mask = W.density_mask(dens2d, n_peak, cfg.mask_threshold,
                                  cfg.mask_close_passes, axes=axes)
            w2d = xp.rint(W.winding_slice(psi2d, normal))
            w2d = xp.where(W.plaquette_mask(mask, *axes), w2d, 0)

            # float16 density / int8 winding: display-only arrays, and a
            # 3-plane movie at N=384 would otherwise run to hundreds of MB.
            # Clipped into float16's range first: an under-resolved run develops
            # local density spikes past 65504, the cast turns those into inf,
            # interpolate_frames turns inf*0 into NaN, and the movie writer then
            # died with "Invalid vmin or vmax" -- losing every output of a run
            # whose physics had already finished. The counter works on the real
            # float32 field, so this only ever saturates the picture.
            self.slices.setdefault((normal, "density"), []).append(
                np.clip(to_numpy(dens2d), -65000.0, 65000.0).astype(np.float16))
            self.slices.setdefault((normal, "winding"), []).append(
                to_numpy(w2d).astype(np.int8))

        # float32, not the float16 the density panels use: mode populations run
        # from 1/2 (bare vacuum) to ~3e4 (condensate centre), and float16 tops
        # out at 65504 -- close enough to that peak to risk clipping the one
        # feature the panel is normalised against.
        for normal in cfg.momentum_planes:
            pk = momentum_plane(self.engine, psi_traj, normal)
            pk = pk[..., self._k_slice, self._k_slice]
            self.slices.setdefault((normal, "momentum"), []).append(
                to_numpy(pk).astype(np.float32))

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
