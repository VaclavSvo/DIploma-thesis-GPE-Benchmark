"""Classical (no-TW) GPE solvers filling the 4-slot contract from base.py.
TW noise is added by sibling solvers in tw_solver.py, not by editing these.
"""
from .backend import xp
from .base import State
from .physics import GPEPhysics3D, choose_dt
from .potentials import harmonic_trap


class _GPESolverBase:
    """Engine construction, units label, dt selection and call()/
    get_derived_quantities() -- identical for every solver. Subclasses
    implement init_state() and end it with self._finalize_state(psi0, params).
    """

    def __init__(self, N: int, L: float, g: float = 1.0, omega=(1.0, 1.0, 1.0),
                 splitstep_order: int = 2):
        self.N, self.L, self.g, self.omega = N, L, g, omega
        self.splitstep_order = splitstep_order  # 2 = Strang, 4 = Yoshida
        self.engine = GPEPhysics3D(N=N, L=L, g=g)
        self.V = None   # set by each subclass

    def units(self) -> dict:
        return dict(hbar=1.0, m=1.0, system="natural units (hbar=m=1)")

    def init_state(self, params: dict) -> State:
        raise NotImplementedError

    def _finalize_state(self, psi0, params: dict) -> None:
        """Common tail of init_state(): pick dt (choose_dt, or an explicit
        params["dt"]) and load psi0 into the engine.

        params["order4_dt_multiplier"] passes straight through to choose_dt,
        so a solver whose dynamics weren't part of the global calibration
        (gpe3d/tw_solver.py) can supply its own instead of inheriting one
        tuned on different physics.
        """
        n_peak = float(self.engine._density(psi0).max())
        real_dt = params.get("dt") or choose_dt(self.engine.K_sq, self.g, n_peak,
                                                 frac=params.get("dt_accuracy_frac", 0.2),
                                                 order=self.splitstep_order,
                                                 order4_multiplier=params.get("order4_dt_multiplier"))
        self.engine.set_dt(real_dt, order=self.splitstep_order)
        self.engine.psi = psi0

    def call(self, state: State, n_steps: int) -> State:
        self.engine.psi = state.psi
        self.engine.V = self.V
        self.engine.run_block(n_steps)
        state.psi = self.engine.psi
        state.t += n_steps * self.engine.dt
        state.step += n_steps
        return state

    def get_derived_quantities(self, state: State) -> dict:
        self.engine.psi = state.psi
        self.engine.V = self.V
        return self.engine.compute_diagnostics()


class ClassicalTrappedGPESolver(_GPESolverBase):
    """Single trapped condensate. A plain ground state is stationary, so
    config.py's INITIAL_OFFSET/INITIAL_VELOCITY apply engine.shift()/kick()
    to get Kohn-mode sloshing.
    """

    def __init__(self, N: int, L: float, g: float = 1.0, omega=(1.0, 1.0, 1.0),
                 splitstep_order: int = 2):
        super().__init__(N, L, g, omega, splitstep_order)
        self.V = harmonic_trap(self.engine.X, self.engine.Y, self.engine.Z, omega)

    def units(self) -> dict:
        d = super().units()
        d.update(length_unit="1 = harmonic osc. length for omega=1",
                  energy_unit="1 = hbar*omega for omega=1")
        return d

    def init_state(self, params: dict) -> State:
        psi0 = self.engine.ground_state_imaginary_time(
            V=self.V, n_particles=params.get("n_particles", 1.0),
            n_steps=params.get("imag_time_steps", 2000),
            dt=params.get("imag_dt"),
            verbose=params.get("verbose", False),
        )
        psi0 = self.engine.shift(psi0, params.get("initial_offset", (0.0, 0.0, 0.0)))
        psi0 = self.engine.kick(psi0, params.get("initial_velocity", (0.0, 0.0, 0.0)))
        self._finalize_state(psi0, params)
        return State(psi=psi0, t=0.0, step=0)


def two_cloud_collision_psi(engine, single_cloud_psi, params: dict):
    """Two shifted/kicked copies of an already-relaxed single-cloud ground
    state, closing in -- the "prepare in a trap, release, collide" protocol.
    Shared by CollidingCondensatesSolver and its TW sibling, which differ only
    in whether noise is added afterwards.

    params:
      separation             (dx,dy,dz) between cloud centres, default (3,0,0)
      half_velocity          each cloud's COM-frame velocity, default (1,0,0);
                             the clouds close in at twice this
      n_particles            total atoms in both clouds, default
                             2*n_particles_per_cloud (no loss assumed)
      n_particles_per_cloud  only used for that default
    """
    sep = params.get("separation", (3.0, 0.0, 0.0))
    half_v = params.get("half_velocity", (1.0, 0.0, 0.0))
    offset_plus = tuple(0.5 * s for s in sep)
    offset_minus = tuple(-0.5 * s for s in sep)
    v_plus = tuple(-v for v in half_v)   # cloud at +offset moves in -direction
    v_minus = tuple(v for v in half_v)   # cloud at -offset moves in +direction

    psi_a = engine.kick(engine.shift(single_cloud_psi, offset_plus), v_plus)
    psi_b = engine.kick(engine.shift(single_cloud_psi, offset_minus), v_minus)
    psi0 = (psi_a + psi_b).astype(psi_a.dtype)

    # Well-separated clouds barely overlap, so the norm should already be
    # right; renormalise anyway, which also catches a too-small separation.
    n_per_cloud = params.get("n_particles_per_cloud", 0.5)
    total_n = params.get("n_particles", 2.0 * n_per_cloud)
    norm = float(xp.sum(engine._density(psi0))) * engine.dV
    if norm > 0.0:
        psi0 = psi0 * (total_n / norm) ** 0.5
    return psi0.astype(xp.complex64)


class CollidingCondensatesSolver(_GPESolverBase):
    """Two ground-state condensates displaced symmetrically, kicked towards
    each other and released into free space (V=0) to collide. V=0 during the
    collision means momentum and angular momentum genuinely are conserved
    here, unlike the trapped-dipole case.
    """

    def __init__(self, N: int, L: float, g: float = 1.0, omega=(1.0, 1.0, 1.0),
                 splitstep_order: int = 2):
        super().__init__(N, L, g, omega, splitstep_order)
        self._trap_shape = harmonic_trap(self.engine.X, self.engine.Y, self.engine.Z, omega)
        # zeros_like(K_sq), not zeros_like(X): X/Y/Z are broadcastable
        # (1,1,N)-style views, so zeros_like(X) would give a (1,1,N) "field" --
        # still correct under broadcasting, but fragile. K_sq is always dense.
        self.V = xp.zeros_like(self.engine.K_sq)

    def init_state(self, params: dict) -> State:
        psi_single = self.engine.ground_state_imaginary_time(
            V=self._trap_shape, n_particles=params.get("n_particles_per_cloud", 0.5),
            n_steps=params.get("imag_time_steps", 2000),
            dt=params.get("imag_dt"),
            verbose=params.get("verbose", False),
        )
        psi0 = two_cloud_collision_psi(self.engine, psi_single, params)
        self._finalize_state(psi0, params)
        return State(psi=psi0, t=0.0, step=0)
