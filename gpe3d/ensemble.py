"""Chunked Truncated-Wigner ensemble runner.

Prepares the mean-field ground state once, then draws and evolves
`batch_size`-sized chunks of trajectories sequentially, folding each into a
running weighted mean. Peak memory scales with batch_size instead of
n_trajectories, at the cost of more sequential chunks -- the same total FFT
work either way.

Chunking is exact, not an approximation: every quantity combined here is a
per-trajectory mean within its chunk, and the overall mean is the chunk-size-
weighted average of those (a sum over a partition equals the sum of partial
sums). Constants like depletion_fraction trivially survive the same average.
"""
from .backend import xp, to_numpy


def run_tw_ensemble(solver, params: dict, n_blocks: int, steps_per_block: int,
                     n_trajectories: int, batch_size: int,
                     track_density: bool = False, z_slice: int | None = None,
                     diagnostics_every: int = 1, observer=None):
    """Evolve ceil(n_trajectories/batch_size) chunks through the same
    n_blocks x steps_per_block schedule, returning (history, densities) as
    ensemble means. Chunk seeds are offset off params["seed"] so the whole
    ensemble stays reproducible. The memory pool is released between chunks.

    diagnostics_every -- recompute get_derived_quantities()/
        physical_particle_number() only every N blocks (they carry a real
        transient peak); intermediate blocks repeat the last values at the new
        timestamp. The first and last block of every chunk are always fresh --
        they are the drift reference and the reported final state -- so this
        only trades away interior time resolution. Density frames are
        unaffected, and history/densities stay n_blocks+1 long either way.

    observer -- optional object with on_frame(state, block_idx, traj_offset),
        called once per recorded block per chunk on that chunk's LIVE psi,
        before any averaging. Necessary for vortex detection: vortices
        nucleate in different places in every trajectory, so the ensemble-mean
        density is a smooth blob with no cores in it. traj_offset is the
        global index of the chunk's first trajectory.
    """
    if batch_size < 1:
        raise ValueError(f"run_tw_ensemble: batch_size must be >= 1, got {batch_size}")
    if diagnostics_every < 1:
        raise ValueError(f"run_tw_ensemble: diagnostics_every must be >= 1, got {diagnostics_every}")

    solver.prepare_mean_field(params)  # no-op if run.py already did it
    # The solver's own vacuum+thermal ordering correction, so the density
    # slices get the same correction as physical_particle_number()'s n_phys.
    offset = solver.offset_density
    base_seed = params.get("seed", 0)

    n_done = 0
    chunk_idx = 0
    t_values = None
    history_sum = None   # key -> running weighted sum per block
    density_sum = None   # running weighted-sum 2D array per block
    n_chunks = -(-n_trajectories // batch_size)  # ceil

    while n_done < n_trajectories:
        chunk_size = min(batch_size, n_trajectories - n_done)
        # Seed offset big enough that no chunk's trajectories overlap the next
        # chunk's for any realistic batch_size.
        state = solver.sample_chunk(chunk_size, seed=base_seed + 1_000_000 * (chunk_idx + 1))

        chunk_history = {"t": []}
        chunk_densities = [] if track_density else None
        last_raw, last_phys = None, None  # carried forward on skipped blocks

        def record(block_idx: int, force: bool = False):
            nonlocal last_raw, last_phys
            chunk_history["t"].append(state.t)
            if force or last_raw is None or block_idx % diagnostics_every == 0:
                last_raw = solver.get_derived_quantities(state)
                last_phys = solver.physical_particle_number(state)
            for k, v in last_raw.items():
                chunk_history.setdefault(k, []).append(v / chunk_size)
            for k, v in last_phys.items():
                chunk_history.setdefault(f"phys_{k}", []).append(v)
            if track_density:
                # Slice psi FIRST, then square: _density(state.psi)[:, n] built
                # a full (B,N,N,N) float32 (226MB at N=384) to keep one plane,
                # inside the diagnostics peak this runner exists to avoid.
                n = z_slice if z_slice is not None else solver.engine.N // 2
                psi_b = state.psi
                plane = psi_b[:, n] if psi_b.ndim == 4 else psi_b[n]
                dens = solver.engine._density(plane)
                mid_slice = (to_numpy(dens.mean(axis=0)) if plane.ndim == 3
                             else to_numpy(dens))
                chunk_densities.append(mid_slice - offset)
            if observer is not None:
                observer.on_frame(state, block_idx, traj_offset=n_done)

        record(0, force=True)
        for i in range(n_blocks):
            solver.call(state, steps_per_block)
            record(i + 1, force=(i == n_blocks - 1))

        if history_sum is None:
            t_values = chunk_history["t"]
            history_sum = {k: [0.0] * len(v) for k, v in chunk_history.items() if k != "t"}
            if track_density:
                density_sum = [None] * len(chunk_densities)

        for k, vals in chunk_history.items():
            if k == "t":
                continue
            for i, v in enumerate(vals):
                history_sum[k][i] += v * chunk_size
        if track_density:
            for i, frame in enumerate(chunk_densities):
                weighted = frame * chunk_size
                density_sum[i] = weighted if density_sum[i] is None else density_sum[i] + weighted

        # Folded into the running sums -- drop the chunk and release the pool
        # before the next one allocates its own (batch_size,N,N,N) working set.
        del state, chunk_history, chunk_densities
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            pass

        n_done += chunk_size
        chunk_idx += 1
        print(f"  TW ensemble: chunk {chunk_idx}/{n_chunks} done "
              f"({n_done}/{n_trajectories} trajectories)")

    history = {"t": t_values}
    for k, sums in history_sum.items():
        history[k] = [s / n_trajectories for s in sums]
    densities = ([frame / n_trajectories for frame in density_sum]
                 if track_density else None)
    return history, densities
