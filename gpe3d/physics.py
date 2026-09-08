"""3D split-step Fourier GPE engine.

    i dpsi/dt = (-1/2 grad^2 + V(r) + g|psi|^2) psi      (hbar = m = 1)

Strang (order 2) or Yoshida (order 4) composition, plus imaginary-time
relaxation for the ground state and the conserved-quantity diagnostics.

psi is (N,N,N) for one trajectory or (B,N,N,N) for a batch of Truncated-
Wigner trajectories; every FFT passes axes=_FFT_AXES so only the last three
axes are ever transformed.
"""
import numpy as np

from .backend import xp, HAS_GPU, fftn, ifftn, to_numpy

_FFT_AXES = (-3, -2, -1)

if HAS_GPU:
    # One CUDA kernel instead of ~5 elementwise launches, once per substep.
    # density is abs(psi)**2, not psi.real**2+psi.imag**2: cupy.fuse() can
    # trace ufuncs but not the .real/.imag attributes.
    @xp.fuse()
    def _fused_nonlinear_kick(psi, V, g, dt):
        return psi * xp.exp(-1j * dt * (V + g * xp.abs(psi) ** 2))
else:
    _fused_nonlinear_kick = None

# Yoshida (1990): S4(dt) = S2(theta*dt) . S2((1-2*theta)*dt) . S2(theta*dt).
# The middle sub-step is negative -- unavoidable above 2nd order, and harmless
# here since exp(-1j*tau*(...)) stays unitary for tau < 0.
_YOSHIDA_THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_MID = 1.0 - 2.0 * _YOSHIDA_THETA


def grid_spacing(N: int, L: float) -> float:
    """The grid spacing GPEPhysics3D actually uses, dx = L/(N-1).

    The grid is linspace(-L/2, +L/2, N) with BOTH endpoints kept, so it is
    symmetric under r -> -r (nice for a symmetric trap) but its N points are
    spaced L/(N-1), not L/N. Everything downstream is built from dx and is
    therefore self-consistent -- the FFT's periodic box is N*dx = L*N/(N-1),
    a fraction 1/(N-1) larger than the nominal L (0.4% at N=250).

    Exposed so callers that need dx BEFORE an engine exists (run.py's
    pre-flight resolution gate) cannot drift from the engine's own value:
    using L/N there overstates the Nyquist wavenumber pi/dx by N/(N-1) and
    makes the aliasing gate that much too generous.
    """
    if N < 2:
        raise ValueError(f"grid_spacing: need N >= 2, got {N}")
    return L / (N - 1)


class GPEPhysics3D:

    def __init__(self, N: int, L: float, g: float = 1.0) -> None:
        self.N, self.L, self.g = N, L, g
        self.dt = None
        self._order = 2
        self.psi = None
        self.V = None
        self._build_grid()
        if not HAS_GPU:
            self._ensure_cpu_scratch_buffers((N, N, N))

    # -- Grid (dt-independent) --------------------------------------------
    def _build_grid(self) -> None:
        # X/Y/Z and KX/KY/KZ are broadcastable (N,1,1)/(1,N,1)/(1,1,N) views,
        # not dense (N,N,N) arrays: every use is elementwise against a real
        # (N,N,N) field, so broadcasting gives the same result for a fraction
        # of the memory (~600MB saved at N=256). K_sq genuinely mixes all
        # three axes and is used every substep, so it stays dense.
        x = xp.linspace(-self.L / 2, self.L / 2, self.N, dtype=xp.float32)
        # dx from the exact L/(N-1), not from float32 x[1]-x[0]: the latter is
        # one rounded difference of two rounded float32s and came out ~4e-6
        # relative off the spacing it is meant to describe, which then went
        # into dV (3x that), into every norm and energy, and into the k grid.
        self.dx = grid_spacing(self.N, self.L)
        self.dV = self.dx ** 3
        self.Z = x.reshape(self.N, 1, 1)
        self.Y = x.reshape(1, self.N, 1)
        self.X = x.reshape(1, 1, self.N)

        k = xp.fft.fftfreq(self.N, d=self.dx).astype(xp.float32) * (2.0 * xp.pi)
        self.KZ = k.reshape(self.N, 1, 1)
        self.KY = k.reshape(1, self.N, 1)
        self.KX = k.reshape(1, 1, self.N)
        self.K_sq = self.KX ** 2 + self.KY ** 2 + self.KZ ** 2

    def _ensure_cpu_scratch_buffers(self, shape: tuple) -> None:
        """Scratch for the CPU nonlinear kick, reallocated only when the
        shape changes -- (N,N,N) single trajectory or (B,N,N,N) batch. Not
        used on GPU, where _fused_nonlinear_kick always wins and 4 unused
        full-grid arrays measurably slow the run down."""
        if getattr(self, "_dens_buf", None) is not None and self._dens_buf.shape == shape:
            return
        self._dens_buf = xp.empty(shape, dtype=xp.float32)
        self._imag_sq_buf = xp.empty(shape, dtype=xp.float32)
        self._theta_buf = xp.empty(shape, dtype=xp.float32)
        self._phase_buf = xp.empty(shape, dtype=xp.complex64)

    @staticmethod
    def _density(psi):
        return psi.real ** 2 + psi.imag ** 2

    # -- Ground state via imaginary time ------------------------------------
    def ground_state_imaginary_time(self, V, n_particles: float = 1.0,
                                     n_steps: int = 2000, dt: float | None = None,
                                     tol: float = 1e-8, init_width: float | None = None,
                                     check_every: int = 50, verbose: bool = False):
        """Relax to the trap ground state in imaginary time (t -> -i*tau),
        renormalising to n_particles every step since imaginary time is
        dissipative. Stops once the energy changes by < tol (relative).
        """
        self.V = V
        if dt is None:
            # Imaginary time is unconditionally stable; dt only sets
            # convergence speed/accuracy.
            dt = 0.5 / (0.5 * float(xp.max(self.K_sq)))
        if init_width is None:
            init_width = self.L / 8.0

        r2 = self.X ** 2 + self.Y ** 2 + self.Z ** 2
        psi = xp.exp(-0.5 * r2 / init_width ** 2).astype(xp.complex64)
        psi *= self._norm_factor(psi, n_particles)

        kin_half_im = xp.exp(-0.25 * dt * self.K_sq).astype(xp.complex64)  # 0.25: see set_dt
        prev_E = None
        for it in range(n_steps):
            psi = ifftn(self._kin_multiply(fftn(psi, axes=_FFT_AXES), kin_half_im), axes=_FFT_AXES)
            psi = psi * xp.exp(-dt * (V + self.g * self._density(psi))).astype(xp.complex64)
            psi = ifftn(self._kin_multiply(fftn(psi, axes=_FFT_AXES), kin_half_im), axes=_FFT_AXES)
            psi *= self._norm_factor(psi, n_particles)

            if it % check_every == 0 or it == n_steps - 1:
                E = self._energy(psi, V)
                if prev_E is not None and abs(E - prev_E) < tol * max(abs(E), 1e-30):
                    if verbose:
                        print(f"imaginary-time converged at step {it}, E={E:.6f}")
                    break
                prev_E = E

        self.psi = psi.astype(xp.complex64)
        return self.psi

    def _norm_factor(self, psi, n_particles: float) -> float:
        norm = float(xp.sum(self._density(psi))) * self.dV
        return xp.sqrt(n_particles / norm) if norm > 0.0 else 1.0

    # -- Initial-condition shaping (applied once, before real time) ---------
    def shift(self, psi, offset):
        """Rigid translation by (dx,dy,dz) via the Fourier shift theorem --
        exact on a periodic grid, no interpolation. Displacing the ground
        state excites the dipole mode, which by Kohn's theorem oscillates at
        the bare trap frequency regardless of g."""
        dx0, dy0, dz0 = offset
        if dx0 == 0.0 and dy0 == 0.0 and dz0 == 0.0:
            return psi
        phase = xp.exp(-1j * (self.KX * dx0 + self.KY * dy0 + self.KZ * dz0))
        return ifftn(phase * fftn(psi, axes=_FFT_AXES), axes=_FFT_AXES).astype(xp.complex64)

    def kick(self, psi, velocity):
        """Galilean boost exp(i*v.r): |psi|^2 unchanged, net momentum p = v
        (hbar=m=1)."""
        vx, vy, vz = velocity
        if vx == 0.0 and vy == 0.0 and vz == 0.0:
            return psi
        phase = xp.exp(1j * (vx * self.X + vy * self.Y + vz * self.Z))
        return (psi * phase).astype(xp.complex64)

    # -- Time step (dt-dependent operators) ---------------------------------
    def set_dt(self, dt: float, order: int = 2) -> None:
        """Kinetic operator T = -1/2 grad^2 has Fourier eigenvalue 0.5*K_sq,
        so the half-dt propagator is exp(-1j*0.25*dt*K_sq). A coefficient of
        0.5 here (the 2D reference notebook's convention) silently implements
        T = -grad^2, i.e. half the intended mass -- verified against a single
        free-particle Fourier mode.

        Only order=2's operators are persisted (2 complex64 (N,N,N) arrays,
        ~2.1GB at N=512); order=4 builds its own pair per run_block() call so
        they are not resident during compute_diagnostics()' own peak.
        """
        self.dt = dt
        self._order = order
        for attr in ("_kin_half", "_kin_full"):
            if hasattr(self, attr):
                delattr(self, attr)

        if order == 2:
            self._kin_half = xp.exp(-1j * 0.25 * dt * self.K_sq).astype(xp.complex64)
            self._kin_full = (self._kin_half * self._kin_half).astype(xp.complex64)
        elif order != 4:
            raise ValueError(f"set_dt: order must be 2 or 4, got {order!r}")

    @staticmethod
    def _kin_multiply(psi_k, kin):
        """In-place on CPU (removes one allocation from fftn's output,
        measured 1.3-1.65x faster over run_block); out-of-place on GPU, where
        in-place measured as a net regression on a GTX 1660 Ti."""
        if HAS_GPU:
            return kin * psi_k
        psi_k *= kin
        return psi_k

    def _nonlinear_kick(self, psi, dt: float | None = None):
        """dt=None uses self.dt (order 2). order=4 passes its Yoshida
        sub-step taus explicitly, including the negative middle one."""
        if dt is None:
            dt = self.dt

        if _fused_nonlinear_kick is not None:
            return _fused_nonlinear_kick(psi, self.V, self.g, dt)

        # CPU: build density/phase into preallocated scratch and mutate psi in
        # place (psi is always a fresh ifftn output from run_block, never
        # aliased). Must stay equivalent to
        #   psi * exp(-1j*dt*(V + g*(psi.real**2 + psi.imag**2)))
        self._ensure_cpu_scratch_buffers(psi.shape)
        dens, imag_sq = self._dens_buf, self._imag_sq_buf
        theta, phase = self._theta_buf, self._phase_buf

        xp.square(psi.real, out=dens)
        xp.square(psi.imag, out=imag_sq)
        dens += imag_sq                # |psi|^2
        theta[...] = self.V
        dens *= self.g
        theta += dens
        theta *= -dt                   # -dt*(V + g|psi|^2)
        xp.cos(theta, out=phase.real)
        xp.sin(theta, out=phase.imag)
        psi *= phase
        return psi

    # -- Real-time propagation ----------------------------------------------
    def run_block(self, n_steps: int) -> None:
        """Advance psi by exactly n_steps sub-steps.

        n_steps == 0 must be a no-op. Without this guard both compositions
        below still apply their leading half/edge kinetic factor and, at
        order 4, return psi with an UNPAIRED exp(-i*theta*dt*K^2/4) on it --
        a silently corrupted state rather than an untouched one.
        """
        if n_steps <= 0:
            if n_steps < 0:
                raise ValueError(f"run_block: n_steps must be >= 0, got {n_steps}")
            return
        if self._order == 4:
            self._run_block_4th(n_steps)
        else:
            self._run_block_2nd(n_steps)

    def _kin_step(self, psi, kin):
        """One kinetic half/full step: FFT, multiply, inverse FFT."""
        return ifftn(self._kin_multiply(fftn(psi, axes=_FFT_AXES), kin), axes=_FFT_AXES)

    def _run_block_2nd(self, n_steps: int) -> None:
        """Strang with merged half-kicks:
        K(dt/2) [N(dt) K(dt)]*(n-1) N(dt) K(dt/2)."""
        psi = self._kin_step(self.psi, self._kin_half)
        for _ in range(n_steps - 1):
            psi = self._kin_step(self._nonlinear_kick(psi), self._kin_full)
        psi = self._kin_step(self._nonlinear_kick(psi), self._kin_half)
        self.psi = psi

    def _run_block_4th(self, n_steps: int) -> None:
        """Yoshida-4: three Strang sub-steps per outer step, merged into

            K_edge N(theta*dt) K_mid N(mid*dt) K_mid N(theta*dt) K_edge

        with K_edge merging into K_edge^2 between consecutive outer steps.
        3x the work of a Strang substep at O(dt^4) instead of O(dt^2), so it
        affords a much larger dt. tests/test_existence.py checks the
        convergence *rate*, which is what a coefficient/sign bug would break.

        The kinetic operators are locals, reclaimed when this returns.
        """
        dt_edge = _YOSHIDA_THETA * self.dt
        dt_mid = _YOSHIDA_MID * self.dt          # negative, see module header
        kin_edge = xp.exp(-1j * 0.25 * _YOSHIDA_THETA * self.dt * self.K_sq).astype(xp.complex64)
        kin_mid = xp.exp(-1j * 0.25 * (1.0 - _YOSHIDA_THETA) * self.dt * self.K_sq).astype(xp.complex64)
        kin_edge_full = (kin_edge * kin_edge) if n_steps > 1 else None

        psi = self._kin_step(self.psi, kin_edge)
        for step in range(n_steps):
            psi = self._kin_step(self._nonlinear_kick(psi, dt=dt_edge), kin_mid)
            psi = self._kin_step(self._nonlinear_kick(psi, dt=dt_mid), kin_mid)
            psi = self._nonlinear_kick(psi, dt=dt_edge)
            psi = self._kin_step(psi, kin_edge if step == n_steps - 1 else kin_edge_full)
        self.psi = psi

    # -- Diagnostics ---------------------------------------------------------
    #
    # Two paths, because the two callers want different things:
    #
    #   _kinetic_energy()  energy only (the imaginary-time convergence check
    #                      and gates.run_gates' reference). Parseval in k
    #                      space: ONE forward FFT, no inverse, no derivative
    #                      array at all.
    #   _diagnostics_from_currents()  the full report, which needs angular
    #                      momentum and therefore the current in REAL space.
    #                      One shared forward FFT + one inverse per axis.
    #
    # Both were previously the same routine doing three forward FFTs -- one
    # per axis, transforming the identical psi three times.

    def _k_marginals(self, dens_k):
        """|psi_k|^2 collapsed onto each k axis: (Sx, Sy, Sz), each length N
        (already summed over any leading batch axis).

        Two full reductions per axis and no (N,N,N) temporary at all, which
        is what makes the Parseval route below cheap in memory as well as in
        transforms.
        """
        nd = dens_k.ndim
        keep = {"x": nd - 1, "y": nd - 2, "z": nd - 3}
        return tuple(dens_k.sum(axis=tuple(a for a in range(nd) if a != keep[c]))
                     for c in ("x", "y", "z"))

    def _kinetic_energy(self, psi) -> float:
        """E_kin = (1/2) integral |grad psi|^2 dV, from ONE forward FFT.

        By Parseval, with the unnormalised (numpy/cupy) forward transform,

            E_kin = (dV / N^3) * sum_k (K^2 / 2) |psi_k|^2

        and K^2 = kx^2 + ky^2 + kz^2 separates, so the k-space density can be
        reduced straight onto three length-N marginals -- no inverse
        transform, no complex64 derivative array, nothing of size N^3 alive
        except |psi_k|^2, which is freed before anything else is allocated.

        Six FFTs and a 24 B/point transient become one FFT and 4 B/point;
        measured ~4.9x faster than the real-space form it replaces. Agrees
        with it to 2e-7 relative (float32 round-off), checked against the
        pre-optimisation reference in tests/test_diagnostics_memory_opt.py.

        The marginals accumulate in float32, exactly as the old full-grid
        reductions did, and are contracted with the k axes in float64.
        """
        dens_k = self._density(fftn(psi, axes=_FFT_AXES))
        Sx, Sy, Sz = (to_numpy(m).astype(np.float64) for m in self._k_marginals(dens_k))
        del dens_k
        kx, ky, kz = (to_numpy(K).astype(np.float64).ravel()
                      for K in (self.KX, self.KY, self.KZ))
        return 0.5 * (self.dV / float(self.N ** 3)) * float(
            np.dot(kx * kx, Sx) + np.dot(ky * ky, Sy) + np.dot(kz * kz, Sz))

    def _diagnostics_from_currents(self, psi):
        """(E_kin, (Px,Py,Pz), (Lx,Ly,Lz)) from the real-space derivatives.

        Angular momentum weights the current by position, so it cannot be had
        in k space cheaply -- this path keeps an inverse FFT per axis. What it
        drops is the forward FFT per axis: psi_k is transformed once and
        reused, 6 transforms down to 4, at exactly the same peak (psi_k is
        resident across the loop where the extra forward output used to be).

        Every reduction here is arithmetically identical to the pre-
        optimisation reference, deliberately. In particular each L component
        is ONE elementwise difference reduced once, never a difference of two
        separate reductions: L is strongly cancelling (|Lz| ~ 0.4 against
        per-term sums of order 1e3 on a generic field), and splitting it
        costs ~4 significant digits -- measured 9e-5 relative error, against
        a 1e-5 regression tolerance.

        The current uses Im(conj(a)*b) = a.real*b.imag - a.imag*b.real, which
        touches only float32 temporaries; the complex form conjures two extra
        complex64 (N,N,N) arrays first.
        """
        psi_k = fftn(psi, axes=_FFT_AXES)
        dV = self.dV
        E_kin = 0.0
        j = {}
        for letter, K_axis in (("x", self.KX), ("y", self.KY), ("z", self.KZ)):
            dpsid = ifftn(1j * K_axis * psi_k, axes=_FFT_AXES)
            E_kin += float(xp.sum(self._density(dpsid))) * dV * 0.5
            jj = psi.real * dpsid.imag
            jj -= psi.imag * dpsid.real
            j[letter] = jj
            del dpsid
        del psi_k

        jx, jy, jz = j["x"], j["y"], j["z"]
        P = (float(xp.sum(jx)) * dV, float(xp.sum(jy)) * dV, float(xp.sum(jz)) * dV)
        L = (float(xp.sum(self.Y * jz - self.Z * jy)) * dV,
             float(xp.sum(self.Z * jx - self.X * jz)) * dV,
             float(xp.sum(self.X * jy - self.Y * jx)) * dV)
        return E_kin, P, L

    def _potential_interaction_energy(self, dens, V) -> tuple[float, float]:
        E_pot = float(xp.sum(V * dens)) * self.dV
        E_int = float(0.5 * self.g * xp.sum(dens ** 2)) * self.dV
        return E_pot, E_int

    def _energy_terms(self, psi, V) -> tuple[float, float, float]:
        dens = self._density(psi)
        E_pot, E_int = self._potential_interaction_energy(dens, V)
        del dens   # freed before the spectral pass, so the two never overlap
        return self._kinetic_energy(psi), E_pot, E_int

    def _energy(self, psi, V) -> float:
        return sum(self._energy_terms(psi, V))

    def compute_diagnostics(self) -> dict:
        """Norm, energy (kin/pot/int), linear and angular momentum. Conserved
        to machine precision for a time-independent potential -- macroscopic
        drift means the run is under-resolved, not that conservation is only
        approximate."""
        psi, V, dV = self.psi, self.V, self.dV

        dens = self._density(psi)
        norm = float(xp.sum(dens)) * dV
        E_pot, E_int = self._potential_interaction_energy(dens, V)
        del dens

        E_kin, (Px, Py, Pz), (Lx, Ly, Lz) = self._diagnostics_from_currents(psi)

        return dict(
            norm=norm, E_total=E_kin + E_pot + E_int,
            E_kin=E_kin, E_pot=E_pot, E_int=E_int,
            Px=Px, Py=Py, Pz=Pz,          # <p> = integral of j dV      (hbar=1)
            Lx=Lx, Ly=Ly, Lz=Lz,          # <L> = integral of r x j dV  (hbar=1)
        )

    def get_density_numpy(self):
        return to_numpy(self._density(self.psi))

    def get_density_slice_numpy(self, index: int, axis: int = -3):
        """One 2D density plane on the host, WITHOUT materialising the full
        (N,N,N) density first.

        The obvious `get_density_numpy()[index]` computes an N^3 float32 array
        and copies all of it across the PCIe bus to throw away all but one
        plane -- 226 MB per movie frame at N=384. Slicing psi first makes it
        0.6 MB.
        """
        return to_numpy(self._density(xp.take(self.psi, index, axis=axis)))

    def free(self) -> None:
        for attr in ('psi', 'V', 'X', 'Y', 'Z', 'KX', 'KY', 'KZ', 'K_sq',
                     '_kin_half', '_kin_full',
                     '_dens_buf', '_imag_sq_buf', '_theta_buf', '_phase_buf'):
            if hasattr(self, attr):
                delattr(self, attr)
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except AttributeError:
            pass  # NumPy backend has no memory pool


# order=4 dt multiplier on top of the order=2 heuristic. Yoshida-4 costs 3x a
# Strang step, so a net win needs a ratio > 3. Swept on real hardware with the
# "collision" scenario (tests/sweep_order4_dt.py): N=128 passed to 24, N=256
# to 248, N=512 passed at 400 but FAILED at 1000 (3.3% energy drift). 400 is
# the largest tested value that passed -- no deliberate headroom -- and it is
# calibrated on smooth noiseless dynamics only; TW noise needs its own, much
# smaller number (see gpe3d/tw_solver.py). Re-sweep rather than hand-editing
# this when N or config.TOL changes substantially.
_ORDER4_DT_MULTIPLIER = 400.0


def choose_dt(K_sq, g: float, n_peak_estimate: float, frac: float = 0.2,
              order: int = 2, order4_multiplier: float | None = None) -> float:
    """Accuracy-safe real-time dt: keep the phase accumulated per step within
    `frac` radians for both the fastest Fourier mode and the interaction term
    at peak density. An accuracy heuristic, not a stability condition --
    split-step is norm-preserving for any dt.

    order=4 multiplies by order4_multiplier (default _ORDER4_DT_MULTIPLIER).
    Always confirm against run_gates() for a new scenario/g/N combination.
    """
    E_kin_max = 0.5 * float(xp.max(K_sq))
    dt_kin = frac / E_kin_max if E_kin_max > 0 else float("inf")
    E_int_max = abs(g) * n_peak_estimate
    dt_nl = frac / E_int_max if E_int_max > 0 else float("inf")

    dt = min(dt_kin, dt_nl)
    if order == 4:
        dt *= order4_multiplier if order4_multiplier is not None else _ORDER4_DT_MULTIPLIER
    elif order != 2:
        raise ValueError(f"choose_dt: order must be 2 or 4, got {order!r}")
    return dt
