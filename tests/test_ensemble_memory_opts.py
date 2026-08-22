"""Regression tests for the GTX 1660 Ti memory optimization pass
(README.md items 1 and 4) -- gpe3d/ensemble.py's
batch_size chunking exactness and the new diagnostics_every cadence knob.
Item 2 (physics.py's order=4 kinetic operators no longer persisted on self)
is covered by tests/test_existence.py::test_set_dt_only_builds_active_order_operators
and tests/test_tw_milestone_c.py::test_batched_matches_sequential_order4
instead -- this file is ensemble.py-specific, since gpe3d/ensemble.py had no
dedicated automated test before this optimization pass touched it.

Run: python tests/test_ensemble_memory_opts.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gpe3d.tw_solver import TWTrappedGPESolver
from gpe3d import ensemble

# Same real dilute-gas coupling as tests/test_tw_milestone_d.py (Na-23,
# arXiv:cond-mat/0602061's reference trap frequency) -- not g=1, see that
# file's module docstring for why.
_G_REAL = 0.003523257353164062


def _params(seed=0):
    return dict(n_particles=1.0e4, imag_time_steps=400, seed=seed, cutoff_multiplier=2.0)


def test_chunked_accumulation_matches_manual_reference():
    """gpe3d/ensemble.py's module docstring claims chunking reproduces the
    true ensemble mean exactly via weighted averaging ("sum of a partition
    equals the sum of partial sums, not an approximation") -- config.py's
    BATCH_SIZE=2->1 change (item 1) leans on that accumulation arithmetic
    being correct, and it had no dedicated automated test before this
    optimization pass touched the file. Checked here against a hand-rolled
    reference that replays the exact same prepare_mean_field()/
    sample_chunk()/call() sequence and chunk seeding
    (base_seed + 1_000_000*(chunk_idx+1), matching gpe3d/ensemble.py) and
    sums the raw (un-normalized, per-chunk) get_derived_quantities() output
    by hand.

    NOTE: this is NOT a check that two DIFFERENT batch_size choices draw
    the same trajectories -- they don't (chunk seeding/shape depends on
    batch_size, so different batch_size choices are independent draws from
    the same distribution, not the same realization). That's expected, not
    a bug: it was tried first while writing this test and correctly showed
    ~1e-3 relative disagreement, consistent with genuine statistical
    scatter across independent small-ensemble draws, not a reduction bug.
    """
    n_trajectories = 4
    batch_size = 2
    n_blocks, steps_per_block = 3, 2
    params = _params()
    base_seed = params["seed"]

    solver = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    history, _ = ensemble.run_tw_ensemble(
        solver, params, n_blocks, steps_per_block,
        n_trajectories=n_trajectories, batch_size=batch_size, track_density=False)
    solver.engine.free()

    # Manual reference: same chunk sizes/seeds ensemble.py's own loop would use.
    solver_ref = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    solver_ref.prepare_mean_field(params)
    n_done, chunk_idx = 0, 0
    sums = {}
    while n_done < n_trajectories:
        chunk_size = min(batch_size, n_trajectories - n_done)
        state = solver_ref.sample_chunk(chunk_size, seed=base_seed + 1_000_000 * (chunk_idx + 1))
        for _ in range(n_blocks):
            solver_ref.call(state, steps_per_block)
        raw_final = solver_ref.get_derived_quantities(state)
        for k, v in raw_final.items():
            sums[k] = sums.get(k, 0.0) + v  # raw v already sums over this chunk's trajectories
        n_done += chunk_size
        chunk_idx += 1
    solver_ref.engine.free()

    for k, total in sums.items():
        expected = total / n_trajectories
        got = history[k][-1]
        rel = abs(got - expected) / max(abs(expected), 1e-30)
        assert rel < 1e-5, f"{k}: manual reference={expected}, run_tw_ensemble={got} (rel={rel:.2e})"
    print("PASS: run_tw_ensemble's weighted-average accumulation matches a hand-rolled reference")


def test_diagnostics_every_default_matches_explicit_every_block():
    """diagnostics_every defaults to 1 -- must be identical to not passing
    the parameter at all, so every existing caller/test gets zero behavior
    change from this optimization pass."""
    n_trajectories = 3
    n_blocks, steps_per_block = 4, 2

    solver_a = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    history_a, _ = ensemble.run_tw_ensemble(
        solver_a, _params(), n_blocks, steps_per_block,
        n_trajectories=n_trajectories, batch_size=2, track_density=False)
    solver_a.engine.free()

    solver_b = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    history_b, _ = ensemble.run_tw_ensemble(
        solver_b, _params(), n_blocks, steps_per_block,
        n_trajectories=n_trajectories, batch_size=2, track_density=False,
        diagnostics_every=1)
    solver_b.engine.free()

    for key in history_a:
        assert history_a[key] == history_b[key], f"diagnostics_every=1 (implicit vs explicit) diverged on {key}"
    print("PASS: diagnostics_every=1 (default) is identical to the pre-optimization code path")


def test_diagnostics_every_preserves_length_and_endpoints():
    """diagnostics_every>1 must not change history/densities list lengths
    (gates.py and run.py's _save_gif both assume len(history[k]) ==
    n_blocks+1 == len(densities) -- neither file is touched by this
    optimization pass, so that invariant must hold exactly) and must
    always give a FRESH (not carried-forward) value at the first and last
    recorded block of every chunk -- those are the two points gates.py's
    drift checks and run.py's final depletion report actually read.
    Density frames (a single, much cheaper _density() call, not
    compute_diagnostics()'s derivative arrays) must be completely
    unaffected -- same physics either way, diagnostics cadence never
    touches state evolution.
    """
    n_trajectories = 2
    n_blocks, steps_per_block = 6, 2

    solver_full = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    history_full, densities_full = ensemble.run_tw_ensemble(
        solver_full, _params(), n_blocks, steps_per_block,
        n_trajectories=n_trajectories, batch_size=1, track_density=True)
    solver_full.engine.free()

    solver_sparse = TWTrappedGPESolver(N=16, L=10.0, g=_G_REAL, omega=(1.0, 1.0, 1.0))
    history_sparse, densities_sparse = ensemble.run_tw_ensemble(
        solver_sparse, _params(), n_blocks, steps_per_block,
        n_trajectories=n_trajectories, batch_size=1, track_density=True,
        diagnostics_every=3)
    solver_sparse.engine.free()

    assert len(history_sparse["t"]) == len(history_full["t"]) == n_blocks + 1
    assert len(densities_sparse) == len(densities_full) == n_blocks + 1

    for key in ("norm", "E_total", "phys_n_phys"):
        assert history_sparse[key][0] == history_full[key][0], f"{key}[0] not fresh under diagnostics_every=3"
        assert history_sparse[key][-1] == history_full[key][-1], f"{key}[-1] not fresh under diagnostics_every=3"

    for i in range(len(densities_full)):
        assert np.allclose(densities_sparse[i], densities_full[i], rtol=1e-5, atol=1e-8), \
            f"density frame {i} differs under diagnostics_every=3 -- should be unaffected"
    print("PASS: diagnostics_every preserves history/densities length, endpoints stay fresh, "
          "and density frames (needed for a smooth gif) are unaffected")


if __name__ == "__main__":
    test_chunked_accumulation_matches_manual_reference()
    test_diagnostics_every_default_matches_explicit_every_block()
    test_diagnostics_every_preserves_length_and_endpoints()
    print("Ensemble memory optimization regression tests: all passed.")
