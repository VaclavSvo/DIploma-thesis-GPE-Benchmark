#!/usr/bin/env python3
"""Runner for every scenario. All parameters live in config.py -- set its
SCENARIO, edit that row, then `python run.py`. No command-line flags.

Adding a scenario: write a solver class implementing the 4-slot contract from
gpe3d/base.py, add one row to SOLVERS below and one to config.SCENARIOS.
Nothing else here needs to change.
"""
from __future__ import annotations
import dataclasses
import math
import os
import time

import config
from gpe3d.backend import HAS_GPU, check_fft_grid_size
from gpe3d.solver import ClassicalTrappedGPESolver, CollidingCondensatesSolver
from gpe3d.tw_solver import TWTrappedGPESolver, TWCollidingCondensatesSolver
from gpe3d import evolve, ensemble, gates
from nucleation import runner as nucleation_runner
from nucleation.settings import from_config as nucleation_config

# (solver kind, is_tw) -> solver class.
SOLVERS = {
    ("trapped", False): ClassicalTrappedGPESolver,
    ("trapped", True): TWTrappedGPESolver,
    ("collision", False): CollidingCondensatesSolver,
    ("collision", True): TWCollidingCondensatesSolver,
}
NUCLEATION_SCENARIOS = ("nucleation_collision", "nucleation_control")


def _spec() -> dict:
    """The selected scenario's row. Read live rather than cached, so a caller
    that flips config.SCENARIO (tests/smoke_nucleation.py) still gets the right
    solver."""
    return config.SCENARIOS[config.SCENARIO]


def check_momentum_gates() -> bool:
    """Are the momentum/angular-momentum gates physically meaningful here?

    Free space (V=0) is translation invariant, so a noiseless collision
    genuinely conserves both. A trapped cloud only does so at rest and centred
    -- displace or kick it and momentum oscillates, which is correct physics
    (gates.run_gates' docstring). TW runs are always excluded: vacuum noise is
    a random, non-symmetric perturbation, so exact conservation is not expected
    trajectory-by-trajectory nor in a modest ensemble.
    """
    if _spec()['tw']:
        return False
    if _spec()['solver'] == "collision":
        return True
    return (config.INITIAL_OFFSET == (0.0, 0.0, 0.0)
            and config.INITIAL_VELOCITY == (0.0, 0.0, 0.0))


def characteristic_velocity() -> float:
    """Largest deliberate velocity kick this scenario imparts, for the
    pre-flight resolution check. TW noise has no bulk kick to check this way --
    its own resolution safety is the cutoff-vs-Nyquist margin inside
    gpe3d/noise.py."""
    if _spec()['solver'] == "collision":
        return max(abs(v) for v in config.COLLISION_HALF_VELOCITY)
    return max(abs(v) for v in config.INITIAL_VELOCITY)


def build_solver_and_state():
    solver = SOLVERS[(_spec()['solver'], _spec()['tw'])](
        N=config.N, L=config.L, g=config.G, omega=config.OMEGA,
        splitstep_order=config.SPLITSTEP_ORDER)

    if _spec()['tw']:
        # Ground state + dt + noise cutoff mask only, NOT a trajectory batch:
        # ensemble.run_tw_ensemble() draws and evolves the trajectories itself,
        # in BATCH_SIZE chunks, so only one chunk is ever resident.
        solver.prepare_mean_field(config.SOLVER_PARAMS)
        return solver, None
    return solver, solver.init_state(config.SOLVER_PARAMS)


def _section(title: str) -> None:
    print(f"\n-- {title} --")


def _block_schedule(dt: float, nucleate: bool) -> tuple[int, int, int]:
    """(n_blocks, steps_per_block, total_steps).

    A nucleation run sets its frame count explicitly: detection costs three
    full-grid winding sweeps per trajectory per frame, so inheriting the gif's
    playback rate would spend more time detecting than evolving. Otherwise one
    frame per 1/GIF_FPS of *simulated* time, which makes the gif play at 1x.
    """
    if nucleate:
        n_blocks = max(2, int(getattr(config, "NUCLEATION_FRAMES", 60))) - 1
    elif config.SAVE_GIF:
        n_blocks = max(2, round(config.T_TOTAL * config.GIF_FPS)) - 1
    else:
        n_blocks = max(1, 3 * config.SNAPSHOT_ROWS - 1)
    steps_per_block = max(1, round(config.T_TOTAL / dt / n_blocks))
    return n_blocks, steps_per_block, steps_per_block * n_blocks


def main():
    nucleate = config.SCENARIO in NUCLEATION_SCENARIOS
    dx = config.L / config.N

    _section("Setup")
    print(f"scenario: {config.SCENARIO}   backend: {'CuPy (GPU)' if HAS_GPU else 'NumPy (CPU)'}   "
          f"splitstep_order: {config.SPLITSTEP_ORDER}")
    print(f"grid: N={config.N}  (N^3={config.N ** 3:,} points)   L={config.L}   dx={dx:.4e}")
    check_fft_grid_size(config.N)  # perf warning only

    # Pre-flight, before the (possibly slow) ground-state prep.
    k_char = characteristic_velocity()
    if k_char > 0:
        res = gates.resolution_gate("velocity_resolution", k_char, math.pi / dx)
        print(f"[{'PASS' if res.passed else 'FAIL'}] {res.name}: {res.detail}")
        if not res.passed:
            print("ABORTING before building the ground state -- raise N or lower L in "
                  "config.py, then rerun.")
            return False

    check_momentum = check_momentum_gates()
    t0 = time.time()
    solver, state = build_solver_and_state()
    if not _spec()['tw'] and _spec()['solver'] == "trapped" and not check_momentum:
        print("note: INITIAL_OFFSET/INITIAL_VELOCITY set -> skipping momentum/angular-momentum "
              "gates (a displaced/kicked cloud is expected to oscillate, not stay constant)")

    _section("Ground state")
    print(solver.units())
    print(f"ground state prepared in {time.time() - t0:.2f}s")

    dt = solver.engine.dt
    n_blocks, steps_per_block, total_steps = _block_schedule(dt, nucleate)

    if _spec()['tw']:
        _section("TW ensemble")
        temp_note = (f"temperature_natural={solver.temperature_natural:.4g} (finite-T)"
                     if solver.temperature_natural > 0.0 else "temperature_natural=0.0 (vacuum-only)")
        print(f"noise cutoff: {solver.n_cutoff_modes}/{config.N ** 3} modes included   {temp_note}")
        print(f"n_traj={config.N_TRAJECTORIES}   batch_size={config.BATCH_SIZE}")
    else:
        _section("Evolution setup")
    print(f"dt={dt:.4e}   total_steps={total_steps}   n_blocks={n_blocks}")

    # Vortex detection rides along with the evolution: it needs each
    # trajectory's own psi, which only exists inside the loop. The ensemble
    # runner's outputs are means over trajectories, and vortices nucleate in
    # different places every time, so an averaged density has no cores at all.
    observer = None
    if nucleate:
        detector_cfg = nucleation_config(config)
        observer = nucleation_runner.make_observer(solver, detector_cfg, final_block=n_blocks)
        _section("Nucleation detector")
        print(f"planes: {','.join(detector_cfg.normals)}   "
              f"mask f={detector_cfg.mask_threshold} (closed {detector_cfg.mask_close_passes} cells)   "
              f"density dip: {detector_cfg.require_density_dip}")
        print(f"link cutoff: {detector_cfg.link_cutoff_dx:g} dx   "
              f"movie planes: {','.join(detector_cfg.slice_planes)} "
              f"(trajectory {detector_cfg.slice_trajectory})")
        psi_ref = solver.psi_mean if _spec()['tw'] else state.psi
        res = nucleation_runner.resolution_note(
            solver.engine, config.G, float(solver.engine._density(psi_ref).max()))
        print(f"healing length xi={res['healing_length']:.4g}   dx/xi={res['dx_over_xi']:.3g} "
              f"-- {'cores resolved' if res['resolved'] else 'WARNING: dx > xi/3, cores under-resolved'}")

    # The ensemble-mean density movie is redundant (and misleading) for a
    # nucleation run -- density_winding.gif replaces it.
    track_density = config.SAVE_GIF and observer is None

    t_run0 = time.time()
    if _spec()['tw']:
        history, densities = ensemble.run_tw_ensemble(
            solver, config.SOLVER_PARAMS, n_blocks, steps_per_block,
            n_trajectories=config.N_TRAJECTORIES, batch_size=config.BATCH_SIZE,
            track_density=track_density, observer=observer)
    else:
        history, densities = evolve.run_and_record(solver, state, n_blocks, steps_per_block,
                                                     track_density=track_density,
                                                     observer=observer)
    wall = time.time() - t_run0

    _section("Evolution")
    print(f"real-time evolution: {total_steps} steps in {wall:.2f}s "
          f"({total_steps / wall:.1f} steps/s)")

    # r_char: cloud size scale for the angular-momentum gate = the harmonic
    # oscillator ground-state width 1/sqrt(2*omega).
    r_char = 1.0 / (2.0 * min(config.OMEGA)) ** 0.5
    results = gates.run_gates(history, tol=config.TOL, r_char=r_char, check_momentum=check_momentum)
    _section("Gates")
    print(gates.report(results))

    depletion_ok = True
    if _spec()['tw']:
        # Norm/energy passing says the propagator behaved; it says nothing
        # about whether the cutoff/n_particles combination is inside TWA's
        # small-depletion validity regime. Folded into the run's pass/fail.
        depl = history["phys_depletion_fraction"][-1]
        depletion_ok = depl < 0.10
        _section("TW diagnostics")
        print(f"[{'PASS' if depletion_ok else 'FAIL'}] depletion_fraction: {depl:.2%} "
              f"(n_phys={history['phys_n_phys'][-1]:.1f} vs "
              f"n_particles_target={config.N_PARTICLES:.0f}) -- "
              f"{'OK' if depletion_ok else 'WARNING: outside TWA small-depletion regime'}")

    os.makedirs(config.OUT_DIR, exist_ok=True)
    if observer is not None:
        _section("Vortices")
        print(nucleation_runner.report(observer))
        _section("Output")
        meta = dict(
            scenario=config.SCENARIO, N=config.N, L=config.L, dx=solver.engine.dx,
            g=config.G, omega=config.OMEGA, T_TOTAL=config.T_TOTAL, dt=dt,
            n_blocks=n_blocks, steps_per_block=steps_per_block,
            n_trajectories=config.N_TRAJECTORIES if _spec()['tw'] else 1,
            splitstep_order=config.SPLITSTEP_ORDER,
            detector=dataclasses.asdict(detector_cfg),
            depletion_fraction=(history["phys_depletion_fraction"][-1] if _spec()['tw'] else 0.0),
            effective_cutoff_multiplier=getattr(solver, "effective_cutoff_multiplier", None),
        )
        paths = nucleation_runner.write_outputs(
            observer, os.path.join(config.OUT_DIR, config.SCENARIO), meta,
            gif_fps=getattr(config, "NUCLEATION_GIF_FPS", 60),
            box_extent=(-config.L / 2.0, config.L / 2.0),
            duration=config.T_TOTAL,   # 1x playback, same rule as the other gifs
            vortex_map=getattr(config, "NUCLEATION_VORTEX_MAP", True))
        for name, path in paths.items():
            print(f"saved {name}: {path}")
    elif config.SAVE_GIF:
        _section("Output")
        gif_path = os.path.join(config.OUT_DIR, "density_slice.gif")
        _save_gif(densities, history["t"], gif_path, config.GIF_FPS)
        print(f"saved {gif_path}  ({len(densities)} frames @ {config.GIF_FPS}fps "
              f"= {len(densities) / config.GIF_FPS:.2f}s, sim T_TOTAL={config.T_TOTAL:.2f}s)")

    solver.engine.free()
    return all(r.passed for r in results) and depletion_ok


def _save_gif(densities, times, path, fps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    vmax = max(d.max() for d in densities)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(densities[0], origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    title = ax.set_title(f"t = {times[0]:.3f}")

    def update(i):
        im.set_data(densities[i])
        title.set_text(f"t = {times[i]:.3f}")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(densities), interval=1000.0 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
