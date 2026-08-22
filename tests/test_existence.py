"""Phase-1 existence checks: physics gates + one analytic cross-check.

Run directly (no pytest needed): `python tests/test_existence.py`
Or with pytest: `pytest tests/test_existence.py -v`
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpe3d.solver import ClassicalTrappedGPESolver
from gpe3d import evolve, gates


def test_gates_pass_interacting_trapped_condensate():
    """Small grid, g=1 (interacting), harmonic trap: norm/energy/P/L must
    hold to the project's +-5% margin (Q3.4) over a real-time run.
    """
    solver = ClassicalTrappedGPESolver(N=32, L=12.0, g=1.0, omega=(1.0, 1.0, 1.0))
    state = solver.init_state(dict(n_particles=1.0, imag_time_steps=1200))

    dt = solver.engine.dt
    n_blocks, steps_per_block = 8, max(1, round(2.0 / dt / 8))
    history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block)

    results = gates.run_gates(history, tol=0.05, r_char=1.0 / (2.0) ** 0.5)
    print(gates.report(results))
    assert all(r.passed for r in results), gates.report(results)


def test_noninteracting_ground_state_matches_analytic():
    """g=0 isotropic 3D QHO: E0 = 1.5*omega, per-axis density std = 1/sqrt(2*omega).
    Cross-checks the imaginary-time relaxation against a known closed form
    (Q9.1/Q6.3: validate against analytic solutions before trusting the
    interacting/TW runs).
    """
    from gpe3d.backend import to_numpy
    omega = 1.0
    solver = ClassicalTrappedGPESolver(N=48, L=14.0, g=0.0, omega=(omega,) * 3)
    state = solver.init_state(dict(n_particles=1.0, imag_time_steps=3000))

    d = solver.get_derived_quantities(state)
    E0_theory = 1.5 * omega
    assert abs(d["E_total"] - E0_theory) / E0_theory < 0.02, \
        f"E0={d['E_total']:.4f} vs theory {E0_theory:.4f}"

    dens = to_numpy(solver.engine._density(state.psi))
    X = to_numpy(solver.engine.X)
    var_x = float((X ** 2 * dens).sum()) * solver.engine.dV  # <x^2>, norm=1
    sigma_theory = (1.0 / (2.0 * omega)) ** 0.5
    sigma_numeric = var_x ** 0.5
    assert abs(sigma_numeric - sigma_theory) / sigma_theory < 0.03, \
        f"sigma_x={sigma_numeric:.4f} vs theory {sigma_theory:.4f}"

    print(f"E0: numeric={d['E_total']:.4f} theory={E0_theory:.4f}  "
          f"sigma_x: numeric={sigma_numeric:.4f} theory={sigma_theory:.4f}")


def test_fused_nonlinear_kick_matches_unfused():
    """The GPU nonlinear-kick kernel (gpe3d.physics._fused_nonlinear_kick,
    cupy.fuse()'d for a single-kernel-launch speedup) must reproduce the
    plain unfused reference expression exactly. This sandbox has no GPU, so
    this only actually exercises the fused path when run on a CuPy machine
    (your laptop) -- it no-ops (prints a skip note, still "passes") on the
    NumPy backend, since NumPy has no .fuse() and the project never takes
    that code path there. Run this on your laptop after `pip install
    cupy-cuda12x` and BEFORE trusting production runs, since the fusion
    itself was written without the ability to execute it here.
    """
    from gpe3d.backend import HAS_GPU, xp
    if not HAS_GPU:
        print("skip: no GPU backend here, fused kernel path not exercised "
              "(run this on your CuPy machine to actually check it)")
        return

    from gpe3d.physics import _fused_nonlinear_kick, GPEPhysics3D
    eng = GPEPhysics3D(N=16, L=8.0, g=1.3)
    shape = (16, 16, 16)
    psi = (xp.random.standard_normal(shape) + 1j * xp.random.standard_normal(shape)).astype(xp.complex64)
    V = (xp.random.standard_normal(shape) ** 2).astype(xp.float32)
    dt = 0.0137

    fused = _fused_nonlinear_kick(psi, V, eng.g, dt)
    dens = eng._density(psi)
    ref = psi * xp.exp(-1j * dt * (V + eng.g * dens))

    max_err = float(xp.max(xp.abs(fused - ref)))
    assert fused.dtype == ref.dtype == xp.complex64, \
        f"dtype mismatch: fused={fused.dtype}, ref={ref.dtype}"
    assert max_err < 1e-4, f"fused nonlinear kick diverges from reference: max_err={max_err}"
    print(f"PASS: fused nonlinear kick matches reference (max_err={max_err:.2e}, dtype={fused.dtype})")


def test_set_dt_only_builds_active_order_operators():
    """GPEPhysics3D.set_dt() must build ONLY the kinetic-operator set the
    active order's run_block() dispatch actually PERSISTS on self --
    _kin_half/_kin_full for order=2. order=4's own pair (_kin4_edge/
    _kin4_mid), and _kin4_edge_full, are deliberately never stored on self
    at all anymore (GTX 1660 Ti memory optimization pass,
    README.md item 2) -- they're built fresh
    as LOCAL variables inside _run_block_4th() (via
    _run_block_4th()), once per run_block() call, and go out of
    scope (reclaimed by the backend's memory pool) as soon as that call
    returns, instead of sitting resident in VRAM for the whole run
    (including during compute_diagnostics()'s own transient peak between
    blocks). This is a pure memory regression guard -- it can't be caught
    by the physics/gate tests above since building/persisting unused or
    extra arrays doesn't change any result, only VRAM footprint. Also
    checks that calling set_dt() again with a different order (e.g.
    sweep_order4_dt.py's order=2/order=4 back-to-back runs on one engine)
    clears the previous order's operators rather than accumulating both,
    and that running an actual order=4 block leaves no kin4 arrays behind
    afterward either.
    """
    from gpe3d.physics import GPEPhysics3D
    from gpe3d.backend import xp
    eng = GPEPhysics3D(N=16, L=8.0, g=1.0)

    eng.set_dt(1e-3, order=2)
    assert hasattr(eng, "_kin_half") and hasattr(eng, "_kin_full")
    assert not any(hasattr(eng, a) for a in ("_kin4_edge", "_kin4_edge_full", "_kin4_mid")), \
        "order=2 set_dt() built order=4-only operators"

    eng.set_dt(1e-3, order=4)
    assert not any(hasattr(eng, a) for a in ("_kin4_edge", "_kin4_edge_full", "_kin4_mid")), \
        "order=4 set_dt() must NOT persist kin4 operators on self -- they're built fresh per " \
        "run_block() call instead (see _run_block_4th()), so a bare set_dt() call " \
        "shouldn't allocate them at all"
    assert not hasattr(eng, "_kin_half") and not hasattr(eng, "_kin_full"), \
        "order=4 set_dt() left order=2 operators allocated (the memory regression this test guards)"

    # Actually run an order=4 block and confirm the kin4 operators still
    # aren't left behind on self afterward -- _run_block_4th()'s
    # whole point is that they're local to _run_block_4th, not a delayed
    # first-call cache.
    eng.V = xp.zeros((eng.N, eng.N, eng.N), dtype=xp.float32)
    eng.psi = xp.exp(-0.5 * (eng.X ** 2 + eng.Y ** 2 + eng.Z ** 2)).astype(xp.complex64)
    eng.run_block(1)
    assert not any(hasattr(eng, a) for a in ("_kin4_edge", "_kin4_edge_full", "_kin4_mid")), \
        "running an order=4 block left kin4 operators persisted on self -- _run_block_4th() " \
        "should only ever produce local variables"

    eng.set_dt(1e-3, order=2)
    assert hasattr(eng, "_kin_half") and hasattr(eng, "_kin_full")
    assert not any(hasattr(eng, a) for a in ("_kin4_edge", "_kin4_edge_full", "_kin4_mid")), \
        "switching back to order=2 left stale order=4 operators allocated"

    print("PASS: set_dt() builds only order=2's persistent kinetic operators; order=4's kin4 "
          "operators are never persisted on self, before or after running a real block")


def test_yoshida4_convergence_order():
    """The 4th-order Yoshida split-step (config.SPLITSTEP_ORDER=4,
    gpe3d.physics.GPEPhysics3D._run_block_4th) must actually converge at
    4th order in dt -- not just "look reasonable" at one dt, which a
    coefficient/sign bug in the Yoshida composition could still pass. Cross-
    check against Kohn's theorem: in a purely harmonic trap, the dipole
    moment <x>(t) = x0*cos(omega*t) EXACTLY, for *any* interaction strength
    g (see README.md's offset/velocity section) -- a genuine closed-form
    reference, not just a finer-resolution numerical one, so this also
    doubles as a check that order=4 hasn't introduced a physics bug.

    Runs the same offset cloud to a fixed T with a few different step counts
    (dt = T/n_steps) at order=2 and order=4, and checks the empirical
    convergence rate log2(err[n]/err[2n]) against each method's expected
    order -- a genuine implementation bug (e.g. a wrong Yoshida coefficient)
    shows up as a rate that silently drops back toward 2 (or worse), which a
    single-dt "is it close enough" check would not catch.
    """
    from gpe3d.backend import to_numpy
    omega, x0, T = 1.0, 1.5, 1.3
    x_theory = x0 * math.cos(omega * T)

    def final_x_error(order, n_steps):
        solver = ClassicalTrappedGPESolver(N=40, L=14.0, g=1.0, omega=(omega,) * 3,
                                            splitstep_order=order)
        state = solver.init_state(dict(n_particles=1.0, imag_time_steps=1500,
                                        initial_offset=(x0, 0.0, 0.0),
                                        dt=T / n_steps, verbose=False))
        state = solver.call(state, n_steps)
        X = to_numpy(solver.engine.X)
        dens = to_numpy(solver.engine._density(state.psi))
        x_num = float((X * dens).sum()) * solver.engine.dV
        return abs(x_num - x_theory)

    for order, expected_rate in ((2, 2.0), (4, 4.0)):
        # (2, 4, 8): the 4th-order method converges so fast that finer step
        # counts (16, 32) fall below this grid/ground-state setup's ~1e-5
        # accuracy floor and the empirical rate collapses toward 0 -- not a
        # bug, just no longer measuring time-integration error at that
        # point. (2,4,8) keeps both orders cleanly in their asymptotic dt^n
        # regime (checked directly: order=2 rates 2.06/2.02, order=4 rates
        # 4.08/3.97 across this range).
        step_counts = (2, 4, 8)
        errs = [final_x_error(order, n) for n in step_counts]
        # Empirical rate from the two finest step counts (least affected by
        # any residual non-asymptotic behaviour at coarse dt).
        rate = math.log(errs[-2] / errs[-1]) / math.log(2.0)
        print(f"order={order}: errors={[f'{e:.2e}' for e in errs]}  "
              f"empirical rate={rate:.2f} (expect ~{expected_rate})")
        # Loose margin (-1.0): this is a 2-point empirical rate estimate from
        # a real PDE integrator, not exact asymptotics -- the point is to
        # catch a method silently behaving like a LOWER order than intended,
        # not to pin the rate to 2 decimal places.
        assert rate > expected_rate - 1.0, \
            f"order={order} split-step converging at rate {rate:.2f}, expected ~{expected_rate} " \
            f"-- check the Yoshida coefficients in physics.py"


def test_order4_dt_retune_keeps_gates_passing():
    """Phase-4 perf pass: choose_dt(order=4) multiplies the order=2 dt by
    physics._ORDER4_DT_MULTIPLIER. The real calibration evidence lives in
    that constant's docstring (tests/sweep_order4_dt.py run on real
    hardware, collision scenario, N=128/256/512 -- 100 was chosen with
    deliberate headroom below the largest value that actually passed, since
    all of that sweep data is from gentle collision dynamics and sharper
    physics not yet added will likely eat into the margin). This test is a
    cheap CPU-only sanity check on top of that, not the primary evidence:
    (1) the multiplier is actually being applied (order=4's auto-picked dt >
    order=2's, by the expected factor), and (2) conservation still holds at
    this small grid even at the now-much-larger multiplier. If this ever
    fails, the fix is to lower _ORDER4_DT_MULTIPLIER (and re-run
    sweep_order4_dt.py on real hardware), not to loosen this test's
    tolerance.
    """
    for N in (32, 48):
        dts = {}
        for order in (2, 4):
            solver = ClassicalTrappedGPESolver(N=N, L=12.0, g=1.0, omega=(1.0, 1.0, 1.0),
                                                splitstep_order=order)
            state = solver.init_state(dict(n_particles=1.0, imag_time_steps=800, verbose=False))
            dts[order] = solver.engine.dt

            n_blocks, steps_per_block = 8, max(1, round(2.0 / solver.engine.dt / 8))
            history, _ = evolve.run_and_record(solver, state, n_blocks, steps_per_block)
            results = gates.run_gates(history, tol=0.05, r_char=1.0 / 2.0 ** 0.5)
            assert all(r.passed for r in results), \
                f"N={N} order={order}: {gates.report(results)}"
            solver.engine.free()

        ratio = dts[4] / dts[2]
        print(f"N={N}: dt(order=2)={dts[2]:.4e}  dt(order=4)={dts[4]:.4e}  ratio={ratio:.2f}")
        # _ORDER4_DT_MULTIPLIER raised 100 -> 400 (explicit instruction, phase 2) --
        # expected ratio updated to match. Bounds widened proportionally (+-5%).
        assert 380.0 < ratio < 420.0, \
            f"N={N}: order=4/order=2 dt ratio={ratio:.2f}, expected ~400.0 -- " \
            f"_ORDER4_DT_MULTIPLIER may have changed without this test being updated"

    print("PASS: order=4 dt retune keeps all conservation gates passing at ~400x the order=2 dt")


if __name__ == "__main__":
    test_noninteracting_ground_state_matches_analytic()
    print("PASS: noninteracting ground state matches analytic QHO result")
    test_gates_pass_interacting_trapped_condensate()
    print("PASS: all physics gates pass for interacting trapped condensate")
    test_set_dt_only_builds_active_order_operators()
    test_fused_nonlinear_kick_matches_unfused()
    test_yoshida4_convergence_order()
    print("PASS: split-step convergence order (2nd and 4th) matches theory")
    test_order4_dt_retune_keeps_gates_passing()
