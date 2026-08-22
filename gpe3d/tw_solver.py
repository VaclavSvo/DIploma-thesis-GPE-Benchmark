"""Truncated-Wigner solvers -- siblings of gpe3d/solver.py's classical ones,
same 4-slot contract. call() and get_derived_quantities() are the unmodified
_GPESolverBase versions: TW trajectories obey the same deterministic GPE, and
the stochasticity lives entirely in the initial condition.

_TWEnsembleMixin holds everything the TW scenarios share -- cutoff selection,
chunk sampling, the ordering-correction diagnostic. A concrete solver supplies
only _build_mean_field() and its own self.V.
"""
from .backend import xp
from .base import State
from .solver import _GPESolverBase, two_cloud_collision_psi
from .potentials import harmonic_trap
from .noise import (energy_cutoff_mask, scale_cutoff_multiplier_to_grid,
                     offset_density, sample_noise)

# TW-specific order=4 dt multiplier, deliberately separate from physics.py's
# collision-calibrated global (400). Vacuum noise puts real amplitude on modes
# right up against the cutoff -- exactly the high-k content a large dt
# resolves worst. Direct check at N=32, TWTrappedGPESolver, order=4: the
# global 400 FAILS the energy gate (18-51% drift, non-monotonic in the
# multiplier), while <=25 passes with large margin (25: 0.041%, 12: 0.005%).
# Small-N CPU finding -- re-verify with tests/sweep_order4_dt.py at production
# N on the GPU.
_TW_ORDER4_DT_MULTIPLIER_DEFAULT = 50.0


class _TWEnsembleMixin:
    """Shared TW machinery. Not usable standalone and deliberately without an
    __init__: a concrete solver calls _GPESolverBase.__init__, sets self.V,
    then calls _init_tw_state(), so the order is explicit at the call site.

    A concrete solver must implement
        _build_mean_field(self, params) -> (psi_mean, n_particles_target)
    """

    def _init_tw_state(self) -> None:
        self.mask = None
        self.n_cutoff_modes = None
        self.n_particles_target = None
        self.n_traj = None
        self.psi_mean = None
        self.effective_cutoff_multiplier = None
        self.cutoff_scaling_info = None
        self.temperature_natural = 0.0   # > 0 adds thermal on top of vacuum noise
        self.offset_density = None       # noise's own density contribution

    def _build_mean_field(self, params: dict):
        raise NotImplementedError

    def prepare_mean_field(self, params: dict):
        """Ground state + cutoff mask + dt: everything computed once per run,
        kept apart from sample_chunk() so gpe3d/ensemble.py can draw many
        cheap noise chunks against one expensive imaginary-time relaxation.
        Idempotent, so callers that don't know whether prep already happened
        can call it unconditionally.

        dt is sized from the mean-field peak density, not the noisy psi0 --
        it has to be fixed before any noise is drawn. The difference is the
        percent-level depletion the ordering-correction gate already checks,
        and every run goes through run_gates() anyway.
        """
        if self.psi_mean is not None:
            return self.psi_mean

        psi_mean, n_particles_target = self._build_mean_field(params)
        self.n_particles_target = n_particles_target

        n_peak = float(self.engine._density(psi_mean).max())
        cutoff_multiplier = params.get("cutoff_multiplier", 2.0)
        self.cutoff_scaling_info = None
        if params.get("scale_cutoff_to_grid", False):
            # cutoff_multiplier becomes the floor; the actual cutoff grows
            # with grid resolution. See noise.scale_cutoff_multiplier_to_grid.
            cutoff_multiplier, self.cutoff_scaling_info = scale_cutoff_multiplier_to_grid(
                self.engine.K_sq, self.g, n_peak,
                floor_multiplier=cutoff_multiplier,
                N_ref=params.get("cutoff_scale_n_ref"),
                growth_exponent=params.get("cutoff_scale_exponent", 2.0 / 3.0),
                safety_margin=params.get("cutoff_nyquist_safety", 2.0),
            )
            info = self.cutoff_scaling_info
            print(f"  scale_cutoff_to_grid: E_floor={info['E_floor']:.4g}  "
                  f"E_target={info['E_target']:.4g} (growth={info['growth']:.3g})  "
                  f"E_ceiling={info['E_ceiling']:.4g}  "
                  f"-> effective cutoff_multiplier={cutoff_multiplier:.4g} "
                  f"({'ceiling-limited' if info['ceiling_limited'] else 'on target'})")
        self.effective_cutoff_multiplier = cutoff_multiplier

        self.mask = energy_cutoff_mask(self.engine.K_sq, self.g, n_peak, cutoff_multiplier)
        self.n_cutoff_modes = int(xp.asarray(self.mask).sum())
        self.psi_mean = psi_mean.astype(xp.complex64)

        self.temperature_natural = float(params.get("temperature_natural", 0.0))
        self.offset_density = offset_density(self.engine.K_sq, self.engine.dV,
                                              self.mask, self.temperature_natural)

        # order4_dt_multiplier only matters at splitstep_order==4; set here so
        # TW runs don't silently inherit physics.py's collision-tuned global.
        params = dict(params)
        params.setdefault("order4_dt_multiplier", _TW_ORDER4_DT_MULTIPLIER_DEFAULT)
        self._finalize_state(self.psi_mean, params)

        # The imaginary-time loop churns thousands of (N,N,N) temporaries
        # through the pool; release them before the first chunk allocates.
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            pass
        return self.psi_mean

    def sample_chunk(self, n_traj_chunk: int, seed: int) -> State:
        """One noise-trajectory chunk against the already-prepared mean field.
        One noise draw plus one ifftn -- cheap next to ground-state prep, so
        gpe3d/ensemble.py can call it repeatedly with different seeds.
        """
        if self.psi_mean is None:
            raise RuntimeError("sample_chunk: call prepare_mean_field() first")
        rng = xp.random.default_rng(seed)
        delta_psi = sample_noise(self.engine.K_sq, self.engine.dV, self.mask,
                                  n_traj=n_traj_chunk, rng=rng,
                                  T=self.temperature_natural)   # (n_traj_chunk, N, N, N)

        if n_traj_chunk == 1:
            psi0 = (self.psi_mean + delta_psi[0]).astype(xp.complex64)
        else:
            psi0 = (self.psi_mean[None, ...] + delta_psi).astype(xp.complex64)

        self.n_traj = n_traj_chunk
        self.engine.psi = psi0
        return State(psi=psi0, t=0.0, step=0)

    def init_state(self, params: dict) -> State:
        """Single-batch entry point: all n_traj trajectories at once. For
        memory-chunked production ensembles use gpe3d/ensemble.run_tw_ensemble()
        instead, which drives prepare_mean_field()+sample_chunk() itself.
        """
        self.prepare_mean_field(params)
        return self.sample_chunk(params.get("n_traj", 1), seed=params.get("seed", 0))

    def physical_particle_number(self, state: State) -> dict:
        """Ordering-corrected particle numbers, for one trajectory
        (psi.ndim==3) or a batch (ndim==4, averaged over the leading axis):

          n_measured_raw     <integral |psi_W|^2 dV>, contaminated by the
                             noise's own (Wigner-ordered) particle content.
          n_vacuum_offset    that contamination, offset_density * V -- constant
                             across trajectories and ensemble sizes. Vacuum
                             only at T=0, vacuum+thermal above it; the name is
                             kept for backward compatibility with callers.
          n_phys             n_measured_raw - n_vacuum_offset, which should
                             recover n_particles_target up to statistical
                             error if TWA is behaving.
          depletion_fraction n_vacuum_offset / n_particles_target. Must be
                             percent-level; order-1 means the run is outside
                             TWA's validity regime whatever the other gates
                             say.
        """
        psi = state.psi
        dens = self.engine._density(psi)
        if psi.ndim == 4:
            n_measured = float(xp.mean(xp.sum(dens, axis=(-3, -2, -1)))) * self.engine.dV
        else:
            n_measured = float(xp.sum(dens)) * self.engine.dV

        V = (self.engine.N ** 3) * self.engine.dV
        n_vacuum_offset = self.offset_density * V
        depletion = (n_vacuum_offset / self.n_particles_target
                      if self.n_particles_target else float("nan"))
        return dict(n_measured_raw=n_measured, n_vacuum_offset=n_vacuum_offset,
                    n_phys=n_measured - n_vacuum_offset, depletion_fraction=depletion)


class TWTrappedGPESolver(_TWEnsembleMixin, _GPESolverBase):
    """Trapped condensate + TW noise. state.psi is (N,N,N) for n_traj=1, or a
    batched (n_traj,N,N,N) sharing one mean field with independent draws."""

    def __init__(self, N: int, L: float, g: float = 1.0, omega=(1.0, 1.0, 1.0),
                 splitstep_order: int = 2):
        _GPESolverBase.__init__(self, N, L, g, omega, splitstep_order)
        self.V = harmonic_trap(self.engine.X, self.engine.Y, self.engine.Z, omega)
        self._init_tw_state()

    def units(self) -> dict:
        d = super().units()
        d.update(length_unit="1 = harmonic osc. length for omega=1",
                  energy_unit="1 = hbar*omega for omega=1")
        return d

    def _build_mean_field(self, params: dict):
        n_particles = params.get("n_particles", 1.0)
        psi_mean = self.engine.ground_state_imaginary_time(
            V=self.V, n_particles=n_particles,
            n_steps=params.get("imag_time_steps", 2000),
            dt=params.get("imag_dt"),
            verbose=params.get("verbose", False),
        )
        return psi_mean, n_particles


class TWCollidingCondensatesSolver(_TWEnsembleMixin, _GPESolverBase):
    """CollidingCondensatesSolver's geometry (prepare in a trap, displace and
    kick apart, release into free space) plus TW noise on the combined initial
    state. V=0 throughout; the noise is a random, non-symmetric perturbation,
    so momentum/angular-momentum gates should stay off for this solver -- see
    run.py's SCENARIOS table."""

    def __init__(self, N: int, L: float, g: float = 1.0, omega=(1.0, 1.0, 1.0),
                 splitstep_order: int = 2):
        _GPESolverBase.__init__(self, N, L, g, omega, splitstep_order)
        self._trap_shape = harmonic_trap(self.engine.X, self.engine.Y, self.engine.Z, omega)
        self.V = xp.zeros_like(self.engine.K_sq)  # see CollidingCondensatesSolver
        self._init_tw_state()

    def _build_mean_field(self, params: dict):
        n_per_cloud = params.get("n_particles_per_cloud", 0.5)
        psi_single = self.engine.ground_state_imaginary_time(
            V=self._trap_shape, n_particles=n_per_cloud,
            n_steps=params.get("imag_time_steps", 2000),
            dt=params.get("imag_dt"),
            verbose=params.get("verbose", False),
        )
        psi0 = two_cloud_collision_psi(self.engine, psi_single, params)
        return psi0, params.get("n_particles", 2.0 * n_per_cloud)
