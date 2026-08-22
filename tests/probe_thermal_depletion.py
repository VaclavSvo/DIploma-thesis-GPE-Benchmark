"""Fast depletion_fraction probe for the ACTIVE TW scenario (currently
tw_trapped_thermal) -- skips run.py's ~800s real-time evolution entirely.

depletion_fraction = n_vacuum_offset / n_particles_target
(gpe3d/tw_solver.py's physical_particle_number()) depends ONLY on
solver.offset_density and solver.n_particles_target, both fixed the
instant prepare_mean_field() returns -- NOT on state.psi, so it is
bit-identical at t=0 and at the end of any evolution regardless of
T_TOTAL/n_traj. Confirmed directly this session (sandbox, N=16): t=0 and
post-evolution values matched to the last bit. So this script gets the
EXACT SAME number run.py prints at the end of a full production run, in
just the ground-state-prep time alone (~35s at N=256 on a GPU, vs ~800s+
for a full T_TOTAL=5 real-time evolution) -- not an approximation, and
not a shortcut that trades away any accuracy.

Motivation: the real GPU run logged in README.md (N=256,
tw_trapped_thermal, T/Tc=10%) came back depletion_fraction=10.23%,
just over run.py's <10% gate. config.py's own tw_trapped_thermal block
already names the two levers to fix this, in preference order: raise
N_PARTICLES first (doesn't cost the T/Tc=10% thermal-signal visibility
that was explicitly requested), then lower CUTOFF_SCALE_EXPONENT. This
script tries candidates along both, at the REAL production N=256/imag_time
convergence (not an under-converged small-N approximation like the
sandbox-only probes earlier this project used), so the numbers here are
directly trustworthy -- no "confirm on your GPU" caveat needed beyond
running this once.

Run with: python tests/probe_thermal_depletion.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import run as run_module
from gpe3d.backend import HAS_GPU, check_fft_grid_size
from gpe3d.units import condensate_critical_temperature_natural

DEPLETION_GATE = 0.10
TARGET_MARGIN = 0.01   # aim for depletion <= gate - margin (<=9%), real
                        # headroom rather than barely scraping under -- a
                        # single N=256 data point (10.23%) is already close
                        # to the gate, so "just barely passing" on this
                        # same probe would carry real risk of flipping back
                        # to FAIL from ordinary trajectory-to-trajectory or
                        # convergence noise.

# In config.py's documented lever-preference order: N_PARTICLES first
# (free -- doesn't cost thermal-signal visibility), CUTOFF_SCALE_EXPONENT
# second (does trade away some of the "higher-N resolves more structure"
# benefit). Edit this list to try other candidates.
CANDIDATES = [
    dict(label="baseline (current config)"),
    dict(label="N_PARTICLES x1.15", n_particles_mult=1.15),
    dict(label="N_PARTICLES x1.3", n_particles_mult=1.3),
    dict(label="CUTOFF_SCALE_EXPONENT=0.6", cutoff_scale_exponent=0.6),
    dict(label="CUTOFF_SCALE_EXPONENT=0.5", cutoff_scale_exponent=0.5),
    dict(label="N_PARTICLES x1.15 + EXPONENT=0.6", n_particles_mult=1.15, cutoff_scale_exponent=0.6),
]


def probe(candidate: dict) -> dict:
    spec = run_module.SCENARIOS[config.SCENARIO]
    if not spec["is_tw"]:
        raise RuntimeError(f"probe_thermal_depletion.py needs a TW scenario -- "
                            f"config.SCENARIO={config.SCENARIO!r} is not one")

    n_particles = config.N_PARTICLES * candidate.get("n_particles_mult", 1.0)
    params = dict(config.SOLVER_PARAMS)
    params["n_particles"] = n_particles
    if "cutoff_scale_exponent" in candidate:
        params["cutoff_scale_exponent"] = candidate["cutoff_scale_exponent"]

    # Keep T/Tc fixed at config.TEMPERATURE_FRACTION_TC as N_PARTICLES
    # varies -- same formula config.py's tw_trapped_thermal block itself
    # uses, so a changed n_particles candidate reflects what actually
    # editing config.py and rerunning would give, not a stale temperature
    # left over from the baseline N_PARTICLES.
    if "temperature_natural" in params and hasattr(config, "TEMPERATURE_FRACTION_TC"):
        Tc = condensate_critical_temperature_natural(config.OMEGA, n_particles)
        params["temperature_natural"] = config.TEMPERATURE_FRACTION_TC * Tc

    solver = spec["solver_class"](N=config.N, L=config.L, g=config.G, omega=config.OMEGA,
                                   splitstep_order=config.SPLITSTEP_ORDER)
    solver.prepare_mean_field(params)

    V = (solver.engine.N ** 3) * solver.engine.dV
    n_vacuum_offset = solver.offset_density * V
    depletion_fraction = n_vacuum_offset / solver.n_particles_target
    result = dict(n_particles=n_particles, temperature_natural=params.get("temperature_natural", 0.0),
                  n_cutoff_modes=solver.n_cutoff_modes, depletion_fraction=depletion_fraction,
                  effective_cutoff_multiplier=solver.effective_cutoff_multiplier)
    solver.engine.free()
    return result


if __name__ == "__main__":
    print(f"backend: {'CuPy (GPU)' if HAS_GPU else 'NumPy (CPU)'}  scenario: {config.SCENARIO}  "
          f"N={config.N}  L={config.L}  gate: depletion_fraction < {DEPLETION_GATE:.0%} "
          f"(targeting <= {DEPLETION_GATE - TARGET_MARGIN:.0%} for real margin)")
    check_fft_grid_size(config.N)

    rows = []
    for c in CANDIDATES:
        t0 = time.time()
        r = probe(c)
        wall = time.time() - t0
        ok = r["depletion_fraction"] < DEPLETION_GATE
        margin_ok = r["depletion_fraction"] <= DEPLETION_GATE - TARGET_MARGIN
        flag = "PASS+margin" if margin_ok else ("PASS (tight)" if ok else "FAIL")
        print(f"\n[{c['label']}]  wall={wall:.1f}s")
        print(f"  n_particles={r['n_particles']:.0f}   temperature_natural={r['temperature_natural']:.4g}   "
              f"n_cutoff_modes={r['n_cutoff_modes']}   effective_cutoff_multiplier={r['effective_cutoff_multiplier']:.4g}")
        print(f"  depletion_fraction={r['depletion_fraction']:.2%}  -- {flag}")
        rows.append((c["label"], r, ok, margin_ok))

    print("\n=== Summary ===")
    for label, r, ok, margin_ok in rows:
        print(f"  {label:45s} depletion={r['depletion_fraction']:6.2%}  "
              f"{'PASS+margin' if margin_ok else ('PASS' if ok else 'FAIL')}")

    winners = [row for row in rows if row[3]]
    if winners:
        best = min(winners, key=lambda row: row[1]["depletion_fraction"])
        print(f"\nRecommended: {best[0]} (depletion={best[1]['depletion_fraction']:.2%}, "
              f"real margin below the {DEPLETION_GATE:.0%} gate).")
        print("Next step: apply that change in config.py's tw_trapped_thermal block, then "
              "confirm with one full run.py run (gates + this same depletion number) before "
              "trusting it in production -- this probe skips evolution, not the norm/energy gates.")
    else:
        print("\nNo candidate cleared the gate with margin -- try larger N_PARTICLES "
              "multipliers or a lower CUTOFF_SCALE_EXPONENT (edit CANDIDATES above).")
