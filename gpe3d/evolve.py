"""The shared "advance in blocks, record diagnostics at each boundary" loop,
used by run.py and the tests so it is written in exactly one place.
"""


def run_and_record(solver, state, n_blocks: int, steps_per_block: int,
                    track_density: bool = False, z_slice: int | None = None,
                    observer=None):
    """Advance `state` through n_blocks x steps_per_block sub-steps, recording
    solver.get_derived_quantities() at t=0 and after every block.

    Returns (history, densities):
      history   -- dict of equal-length lists: "t" plus every key
                   get_derived_quantities() returns.
      densities -- 2D mid-plane (z=z_slice, default N//2) density snapshots if
                   track_density, else None.

    observer -- optional object with on_frame(state, block_idx), called at t=0
                and after every block on the live psi, before anything is
                reduced. nucleation/observer.py uses this to detect vortices in
                step with the evolution.
    """
    history = {"t": []}
    densities = [] if track_density else None

    def record(block_idx: int):
        history["t"].append(state.t)
        for k, v in solver.get_derived_quantities(state).items():
            history.setdefault(k, []).append(v)
        if track_density:
            # get_density_slice_numpy, not get_density_numpy()[n]: the latter
            # builds the whole (N,N,N) density and copies all of it to the
            # host to keep one plane -- 226MB per frame at N=384.
            n = z_slice if z_slice is not None else solver.engine.N // 2
            densities.append(solver.engine.get_density_slice_numpy(n))
        if observer is not None:
            observer.on_frame(state, block_idx)

    record(0)
    for i in range(n_blocks):
        solver.call(state, steps_per_block)
        record(i + 1)

    return history, densities
