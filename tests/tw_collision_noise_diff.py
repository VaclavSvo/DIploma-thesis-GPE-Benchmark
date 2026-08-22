#!/usr/bin/env python3
"""Diagnostic (not a gate/test): isolate what TW vacuum noise actually adds
in the tw_collision scenario. Runs the SAME initial mean-field cloud shapes
through both the noiseless classical CollidingCondensatesSolver and one
single TW noise trajectory (TWCollidingCondensatesSolver, n_traj=1 -- NOT
the ensemble mean run.py normally records, since averaging over
N_TRAJECTORIES suppresses exactly the stochastic structure we're trying to
see here, ~1/sqrt(n_traj)), then plots (TW density - classical density)
instead of the raw density.

Why the ordinary tw_collision gif looks like plain GPE: the collision's own
deterministic motion (clouds flying across the box, colliding) is orders of
magnitude bigger than the noise's perturbation (depletion_fraction is kept
<1% by design here -- much smaller than tw_trapped_dipole's ~4-5%, see
README.md), and in tw_trapped_dipole there's no
competing deterministic motion at all (a plain trap ground state is
stationary), so the same-size noise perturbation is far easier to see
there. This script removes that huge shared background by diffing, so
only the noise-driven divergence between the two trajectories remains.

Both solvers get an IDENTICAL dt (forced onto the classical solver via
params["dt"] = the TW solver's own dt) so their per-block snapshots land
at exactly the same simulated times -- otherwise the diff would be
contaminated by the two runs simply taking different-sized steps, not by
physics.

Run: `python tests/tw_collision_noise_diff.py` from the project root.
Saves outputs/tw_collision_noise_diff.gif (3 panels: classical, TW single
trajectory, and their difference on a diverging colormap).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gpe3d.backend import to_numpy
from gpe3d.units import (derive_natural_units, velocity_to_natural,
                          NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M)
from gpe3d.solver import CollidingCondensatesSolver
from gpe3d.tw_solver import TWCollidingCondensatesSolver

# Same physical parameters as config.py's "tw_collision" scenario block --
# kept independent of config.py itself (not import-coupled to whatever
# SCENARIO happens to be selected there) so this script always runs the
# same comparison regardless of what config.py is currently set to.
OMEGA_X_REAL_HZ = 4.57
TRAP_ANISOTROPY = np.sqrt(8.0)
RELATIVE_COLLISION_VELOCITY_REAL = 4.0e-3
_units = derive_natural_units(NA23_ATOM_MASS_U, NA23_SCATTERING_LENGTH_M, OMEGA_X_REAL_HZ)

N_PER_CONDENSATE = 5.0e3
N_PARTICLES = 2.0 * N_PER_CONDENSATE
G = _units.g
OMEGA = (1.0, 1.0 / TRAP_ANISOTROPY, 1.0 / TRAP_ANISOTROPY)
_v_rel_natural = velocity_to_natural(RELATIVE_COLLISION_VELOCITY_REAL, _units)
COLLISION_HALF_VELOCITY = (_v_rel_natural / 4.0, 0.0, 0.0)
COLLISION_SEPARATION = (6.0, 0.0, 0.0)

N, L = 256, 16.0                # matches config.py's tw_collision grid
SPLITSTEP_ORDER = 4
IMAG_TIME_STEPS = 1000
T_TOTAL = 5.0
N_FRAMES = 60
SEED = 0
NOISE_CUTOFF_MULTIPLIER = 2.0
SCALE_CUTOFF_TO_GRID = True
CUTOFF_SCALE_N_REF = 96
CUTOFF_SCALE_EXPONENT = 2.0 / 3.0
CUTOFF_NYQUIST_SAFETY = 2.0
ORDER4_DT_MULTIPLIER = 100.0    # TW-safe value (gpe3d/tw_solver.py's own
                                # default) -- used for BOTH solvers here,
                                # not just the TW one, so dt/step count
                                # matches exactly (see module docstring).


def _common_params(order4_dt_multiplier=ORDER4_DT_MULTIPLIER, imag_time_steps=IMAG_TIME_STEPS):
    return dict(
        n_particles_per_cloud=N_PER_CONDENSATE,
        n_particles=N_PARTICLES,
        imag_time_steps=imag_time_steps,
        verbose=False,
        separation=COLLISION_SEPARATION,
        half_velocity=COLLISION_HALF_VELOCITY,
        order4_dt_multiplier=order4_dt_multiplier,
    )


def run_diagnostic(N=N, L=L, T_TOTAL=T_TOTAL, n_frames=N_FRAMES,
                    imag_time_steps=IMAG_TIME_STEPS, seed=SEED, verbose=True):
    """Returns (times, classical_frames, tw_frames, tw_solver, tw_state) --
    mid-plane (z=N//2) density slices, one 2D array per recorded frame.
    tw_solver/tw_state are also returned so the caller can print
    physical_particle_number() diagnostics (depletion_fraction etc.)."""
    classical = CollidingCondensatesSolver(N=N, L=L, g=G, omega=OMEGA, splitstep_order=SPLITSTEP_ORDER)
    tw = TWCollidingCondensatesSolver(N=N, L=L, g=G, omega=OMEGA, splitstep_order=SPLITSTEP_ORDER)

    tw_params = dict(_common_params(imag_time_steps=imag_time_steps),
                      cutoff_multiplier=NOISE_CUTOFF_MULTIPLIER,
                      scale_cutoff_to_grid=SCALE_CUTOFF_TO_GRID,
                      cutoff_scale_n_ref=CUTOFF_SCALE_N_REF,
                      cutoff_scale_exponent=CUTOFF_SCALE_EXPONENT,
                      cutoff_nyquist_safety=CUTOFF_NYQUIST_SAFETY,
                      seed=seed)
    tw.prepare_mean_field(tw_params)
    tw_state = tw.sample_chunk(1, seed=seed)  # single trajectory -- raw noise, not ensemble-averaged

    # Force the classical run onto EXACTLY the same dt as the TW run (see
    # module docstring) instead of letting it pick its own via choose_dt.
    classical_state = classical.init_state(dict(_common_params(imag_time_steps=imag_time_steps),
                                                  dt=tw.engine.dt))

    n_blocks = n_frames - 1
    steps_per_block = max(1, round(T_TOTAL / tw.engine.dt / n_blocks))
    if verbose:
        print(f"N={N} L={L} G={G:.4g} dt={tw.engine.dt:.4e} "
              f"steps_per_block={steps_per_block} n_blocks={n_blocks}")

    mid = N // 2

    def slice_of(engine, psi):
        dens = to_numpy(engine._density(psi))
        return dens[mid] if dens.ndim == 3 else dens.mean(axis=0)[mid]

    times, classical_frames, tw_frames = [], [], []

    def record():
        times.append(classical_state.t)
        classical_frames.append(slice_of(classical.engine, classical_state.psi))
        tw_frames.append(slice_of(tw.engine, tw_state.psi))

    record()
    for _ in range(n_blocks):
        classical.call(classical_state, steps_per_block)
        tw.call(tw_state, steps_per_block)
        record()

    if verbose:
        phys = tw.physical_particle_number(tw_state)
        print(f"single-trajectory depletion_fraction={phys['depletion_fraction']:.2%}  "
              f"n_vacuum_offset={phys['n_vacuum_offset']:.2f}")

    classical.engine.free()
    tw.engine.free()
    return times, classical_frames, tw_frames


def make_diff_gif(times, classical_frames, tw_frames, path, fps=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    diffs = [tw - cl for tw, cl in zip(tw_frames, classical_frames)]
    vmax_dens = max(f.max() for f in classical_frames + tw_frames)
    vmax_diff = max(np.abs(d).max() for d in diffs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    ims = [
        axes[0].imshow(classical_frames[0], origin="lower", cmap="viridis", vmin=0, vmax=vmax_dens),
        axes[1].imshow(tw_frames[0], origin="lower", cmap="viridis", vmin=0, vmax=vmax_dens),
        axes[2].imshow(diffs[0], origin="lower", cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff),
    ]
    for ax, title in zip(axes, ["classical (no noise)", "TW (1 trajectory)", "TW − classical (noise only)"]):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    suptitle = fig.suptitle(f"t = {times[0]:.3f}")
    fig.colorbar(ims[2], ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()

    def update(i):
        ims[0].set_data(classical_frames[i])
        ims[1].set_data(tw_frames[i])
        ims[2].set_data(diffs[i])
        suptitle.set_text(f"t = {times[i]:.3f}")
        return ims + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=1000.0 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    times, classical_frames, tw_frames = run_diagnostic()
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "tw_collision_noise_diff.gif")
    make_diff_gif(times, classical_frames, tw_frames, out_path)
    final_diff = tw_frames[-1] - classical_frames[-1]
    print(f"saved {out_path}")
    print(f"final-frame diff: max|diff|={np.abs(final_diff).max():.4g}, "
          f"rms diff={np.sqrt((final_diff ** 2).mean()):.4g}, "
          f"peak classical density={classical_frames[-1].max():.4g}")
