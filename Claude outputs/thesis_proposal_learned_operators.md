# Thesis proposal — learned operators inside a structure-preserving kinetic solver

**Three models, one method, and a roadmap from a working solver to a much cheaper one.**

Draft, September 2026. Supersedes and generalises the single-model collision-operator proposal.

---

## 0. The thesis in one paragraph

Kinetic solvers of the generalized-hydrodynamics family all have the same shape: cheap
structure-preserving transport in phase space, plus one or two expensive operators evaluated at
every grid point and every time step. This thesis replaces those operators with **learned
surrogates that satisfy the operator's structural identities by construction**, and — where the
exact operator admits a cheap residual — uses the surrogate as a *preconditioner or initial guess*
rather than as an answer, so the classical iteration still delivers the accuracy and the learning
buys only speed. The method is developed and validated on a model where every answer is known
exactly (hard rods), then applied to the two expensive operators of the Lieb–Liniger GHD solver:
the **dressing** and the **collision integral**. The deliverable is a solver that is measurably
cheaper at matched accuracy, together with an honest characterisation of where the learned
components fail.

---

## 1. The method: the acceleration ladder

This is the organising idea of the whole thesis and the thing to say first in any talk.

A learned component can enter a classical solver at four levels of aggression. They differ enormously
in risk, and almost everyone in the ML-for-PDE literature starts at the top rung, which is why so
many results are unstable.

| Rung | What is learned | Accuracy risk | Typical gain | When to use |
|---|---|---|---|---|
| **R0** | initial guess / warm start for an iterative solve | **none** — the iteration still converges to tolerance | 1.5–3× fewer iterations | always, if the operator is solved iteratively |
| **R1** | a preconditioner | **none** at convergence | 2–10× fewer iterations | when the system is ill-conditioned |
| **R2** | the operator itself, with a cheap residual used as a per-call certificate and fallback | bounded, and *measured* every step | 10–100× | when a residual is cheap relative to the solve |
| **R3** | the operator, uncertified | unbounded | maximal | only after R2 has mapped the failure modes |

**The rule this thesis adopts:** *if the exact operator has a residual that is cheap relative to
solving it, never start above R1.* An accuracy risk becomes a pure speed play, and the worst
possible outcome of the thesis stops being "the method is unreliable" and becomes "the speed-up was
only 1.8×", which is still a result.

R0 and R1 have a real literature to build on — see [NOWS: neural operator warm starts for
accelerating iterative solvers, arXiv:2511.02481](https://arxiv.org/html/2511.02481v1),
[Neural preconditioning via Krylov subspace geometry, arXiv:2507.15452](https://arxiv.org/abs/2507.15452),
[graph neural preconditioners, arXiv:2406.00809](https://arxiv.org/html/2406.00809v1) — and
essentially none of it has been applied to kinetic or hydrodynamic solvers of this type.

The second organising principle, from the earlier discussion: **learn operators, not solutions.**
Spectral bias makes networks bad at representing oscillatory *fields*; it does not stop them
representing a *smooth operator acting on* such a field. Every target below is an operator.

---

## 2. Model A — the hard-rod gas: the trainer

**Purpose: build and validate the entire pipeline on a model where every answer is known.**

A gas of $N$ rods of length $a$ on a line, colliding elastically. It is integrable (collisions
permute velocities), it has an exact GHD description, and — this is why it is the right first
model — **the dressing collapses to moments**. The scattering shift is the constant $-a$, so the
self-consistent integral equation over rapidity degenerates into a relation involving only the
density $n=\int\rho\,d\theta$ and current $j=\int\theta\rho\,d\theta$. There is no linear system to
solve at all.

$$\partial_t \rho(x,\theta) + \partial_x\!\big(v^{\rm eff}[\rho](x,\theta)\,\rho\big) = 0,
\qquad v^{\rm eff} \ \text{a closed expression in } (\theta, n, j)$$

> Transcribe the exact form of $v^{\rm eff}$ from Doyon & Spohn before writing any code. The
> structure is stated here; the formula is not, on purpose.

**Why it is the ideal trainer:**

- **Three independent ground truths.** (i) The contraction map to a free gas gives closed-form
  solutions for domain-wall initial data. (ii) Event-driven hard-rod molecular dynamics is a
  hundred lines and gives the microscopic answer with no hydrodynamic assumption. (iii) The GHD
  solution itself. A solver that reproduces all three is genuinely validated.
- **The full pipeline in miniature.** Semi-Lagrangian transport on $(x,\theta)$, positivity,
  conservation tests, an invariant test-suite, the certificate/fallback machinery — all of it can be
  built and debugged here, then reused unchanged.
- **It carries a real learned-operator target anyway.** Add an external field and hard rods stop
  being integrable: a Boltzmann-type collision term appears and drives thermalisation through
  distinct stages ([three-stage thermalization of a quasi-integrable system,
  PRResearch 6, 023083 (2024)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.023083)).
  So Model C's method can be rehearsed here with MD as ground truth.
- **It is live.** Recent work on hard-rod hydrodynamics ([arXiv:2407.17067](https://arxiv.org/html/2407.17067),
  [arXiv:2603.18522](https://arxiv.org/html/2603.18522)) means a good solver is of interest on its own.

**Deliverable:** a validated 1D GHD solver, an invariant test-suite, and a benchmark set. Roughly
two months, and it is a thesis chapter even if everything after it fails.

---

## 3. Model B — Lieb–Liniger GHD: the **dressing** operator

**The safest high-value target, and the one I would put the most weight on.**

In GHD proper the effective velocity requires, at every spatial point and every step, the solution
of a linear integral equation in rapidity,

$$h^{\rm dr} = (1 - T n)^{-1} h,$$

with $T$ the differential scattering kernel and $n$ the local occupation function. Discretised this
is a dense $N_\theta \times N_\theta$ solve repeated $N_x$ times per step.

### Why this operator is unusually well suited to learning

1. **It is linear in the source and nonlinear only in $n$.** So the learning problem is not "map a
   function to a function" but "map $n$ to a *linear operator*", which can then be applied exactly
   linearly. That is a very strong architectural constraint and it removes a whole class of possible
   errors. A natural ansatz is a low-rank correction to the identity,
   $(1-Tn)^{-1} \approx I + U(n)V(n)^{\!\top}$, with the rank a tunable accuracy dial.
2. **The residual is one matrix–vector product.** $\;r = \|(1-Tn)\hat h - h\|$. That is $O(N_\theta^2)$
   against $O(N_\theta^3)$ for the solve — so **every single prediction can be certified at
   inference for free**. This is what puts the target on rungs R0–R2 rather than R3.
3. **The dilute limit is exact by construction.** $n \to 0 \Rightarrow h^{\rm dr} \to h$. Write the
   model in residual form and that limit holds identically, trained or not.
4. **It is local in $x$.** Every spatial grid point of every frame of every trajectory is an
   independent training sample. A single simulation yields $N_x \times N_t$ examples. Data is free
   and abundant, which is rare.
5. **Failure is visible, not silent.** A bad dressing shows up immediately as a residual spike, not
   as a plausible-looking wrong answer three chapters later.

### The three rungs, concretely

- **R0:** predict $h^{\rm dr}$ from $n$, use it to warm-start the fixed-point/Krylov iteration that
  the solver already runs. Measure iteration count. *Cannot* change the answer.
- **R1:** predict a low-rank $\hat{P} \approx (1-Tn)^{-1}$ and use it as a preconditioner.
- **R2:** use the prediction directly, evaluate the residual every call, fall back to the exact
  solve when it exceeds tolerance. Log every fallback — the fallback rate as a function of physical
  parameters *is* a result.

---

## 4. Model C — dissipative GHD: the **collision** operator

The other expensive term in the same solver, and the one that carries the physics of thermalisation.
The dissipative GHD equations add a nonlocal diffusion operator and a Boltzmann-type collision
integral to the transport ([Møller, Besse, Mazets, Stimming, Mauser, J. Comput. Phys. (2023),
arXiv:2212.12349](https://arxiv.org/abs/2212.12349)); the collision term is what lets an otherwise
integrable system thermalise.

Unlike the dressing, **this operator has no cheap residual** — there is no linear system whose
satisfaction can be checked. So it sits naturally at R2/R3 and the structural guarantees have to
come from the architecture instead:

| Property | Architectural device |
|---|---|
| occupation bound $0 \le n \le 1$ | sigmoid parameterisation of the output |
| collision invariants annihilated | orthogonal projection onto the complement of $\mathrm{span}\{h_i\}$, exactly as in the worked example |
| correct thermal fixed point | relaxation form $C = (M_\theta[n] - n)/\tau_\theta[n]$ |
| $H$-theorem / entropy production $\ge 0$ | follows from the relaxation form with $\tau > 0$ |
| parity in rapidity | explicit symmetrisation $\tfrac12(\mathcal N[n] + P\,\mathcal N[Pn])$ |

Templates: [RelaxNet, J. Comput. Phys. 490, 112317 (2023)](https://arxiv.org/abs/2211.08149) for the
relaxation decomposition, and [structure-preserving operator learning,
arXiv:2402.16613](https://arxiv.org/pdf/2402.16613) for imposing invariants through an
orthogonalised trunk.

**The physics deliverable, not just the speed one:** the eigenvalues of the *linearised* learned
collision operator are the relaxation rates. Comparing them against the integrability-breakdown
rates of Mazets & Schmiedmayer is both the sharpest accuracy test available and a genuine physics
result. Make it a first-class chapter, not an appendix.

---

## 4b. Does this actually help? A cost model, and one uncomfortable answer

The honest version, because "learned operators accelerate solvers" is a slogan until someone counts
flops. Everything here is an *estimate to be checked in Phase 0*, not a measurement.

### 4b.1 Where the flops are

Per time step, with $N_x$ spatial points, $N_\theta$ rapidity points, $n_\ell$ retained transverse
levels:

| Component | Cost | At $N_x{=}512,\ N_\theta{=}128,\ n_\ell{=}1$ |
|---|---|---|
| Transport (backward SL + interpolation) | $O(N_x N_\theta)$ | ~$10^7$ flops |
| Collision integral | $O(N_x\,n_\ell^2 N_\theta^2)$ | ~$5\times10^7$ |
| **Dressing, dense direct** | $O(N_x N_\theta^3)$ | ~$7\times10^8$ **per pass** |
| Dressing, iterative | $O(N_x N_{\rm iter} N_\theta^2)$ | ~$1.7\times10^8$ at 20 iters |

Two consequences worth stating plainly:

1. **The dressing dominates by one to two orders of magnitude**, and the gap *widens* with $N_\theta$.
   This corrects my earlier guess that the collision integral was the bottleneck — on scaling grounds
   it is not, at $n_\ell = 1$.
2. **The crossover is $n_\ell$.** Collision cost grows as $n_\ell^2$ while dressing grows more slowly,
   so in the dimensional-crossover regime with tens of transverse levels the ordering can invert.
   *Phase 0 has a precise question to answer: at what $n_\ell$ does collision overtake dressing?*

There is a further multiplier specific to this group's scheme. The JCP solver builds its advection
field from a **high-order time-Taylor expansion**, which needs time derivatives of $v^{\rm eff}$ —
and each of those is another dressing-type solve. If so, the dressing cost is multiplied by the
order of the expansion. Worth confirming early: it strengthens the case considerably.

### 4b.2 The uncomfortable answer: try linear algebra first

For Lieb–Liniger the differential scattering kernel is
$$T(\theta,\theta') = \frac{2c}{c^2 + (\theta-\theta')^2},$$
**a function of $\theta - \theta'$ alone.** On a uniform rapidity grid that is a Toeplitz
(convolution) operator, so although $1 - T\,\mathrm{diag}(n)$ is not itself Toeplitz, its
matrix–vector product is: $T(n \odot v)$ is one FFT-based convolution, $O(N_\theta \log N_\theta)$
instead of $O(N_\theta^2)$.

So **if the current code forms and factorises a dense matrix, the first and largest speed-up
available is classical, not learned** — switch to an iterative solve with FFT matvecs. At
$N_\theta = 128$ that is roughly $N_\theta^2 / (N_\theta\log N_\theta) \approx 18\times$ per
iteration, and it costs nothing but a rewrite.

A thesis that reached for a neural network before trying this would deserve the question it would
get in the defence. **Phase 0.5 of the roadmap is therefore: exploit the kernel's structure, and
re-profile.**

### 4b.3 Where learning still has irreducible value

Here is the argument that survives the above, and it is the strongest one in this proposal:

> FFT structure reduces the **cost per iteration**. It does nothing about the **number of
> iterations**. A learned warm start attacks exactly the factor that classical structure
> exploitation cannot touch — and the two gains multiply.

Rung R0 is therefore not a fallback but the natural companion to the Toeplitz rewrite. If the
iteration count drops from ~20 to ~5, that is another 4× on the dominant cost, **with the converged
answer unchanged to solver tolerance** — no accuracy argument to have with anyone.

### 4b.4 Realistic speed-up, with Amdahl applied

Suppose after the Toeplitz rewrite the dressing is a fraction $\phi$ of runtime.

| Scenario | Dressing speed-up | $\phi = 0.9$ | $\phi = 0.7$ | $\phi = 0.4$ |
|---|---|---|---|---|
| R0 warm start, 20→5 iters | 4× | **3.1×** | 2.1× | 1.4× |
| R1 preconditioner, 20→3 | 6.7× | 3.6× | 2.3× | 1.5× |
| R2 direct low-rank surrogate | ~50× | 6.9× | 3.0× | 1.6× |

**The headline: expect 2–7× overall, not 100×.** The ceiling is set by Amdahl, not by how good the
network is — which is why Phase 0 decides whether the thesis is worth doing before any model is
trained.

### 4b.5 Memory: mostly no, and it is better to say so

1D GHD is **flop-bound, not memory-bound**. The state $\rho(x,\theta)$ is $N_x N_\theta n_\ell$
doubles — half a megabyte at the numbers above, 16 MB with 30 transverse levels. The kernel is
128 kB. Nothing here is close to a limit.

The one genuine memory angle is **batched GPU dressing**: holding $N_x$ systems simultaneously costs
$N_x N_\theta^2$ doubles ≈ 67 MB at these sizes, but grows to gigabytes as $N_\theta$ rises, and a
low-rank learned surrogate would replace that with $N_x\, r\, N_\theta$. Real, but secondary.

Memory becomes the binding constraint only in the *extensions*: a 2D soliton gas on a 4D phase space,
or a 3D GPE ensemble (244 MB per trajectory field at $N=248$, which is what limits your nucleation
batch size today). There the relevant technology is adaptive-rank / tensor-train transport — see
[a mass, momentum and energy conserving semi-Lagrangian adaptive-rank method for Vlasov–Poisson,
arXiv:2606.29027](https://arxiv.org/html/2606.29027) and [a semi-Lagrangian Vlasov solver in tensor
train format, SIAM J. Sci. Comput.](https://epubs.siam.org/doi/10.1137/140971270) — **not** learned
operators. Do not conflate the two claims in the proposal.

### 4b.6 What would make the answer "no"

- Phase 0 shows the dressing is under ~40% of runtime → ceiling below 1.6×, and the thesis pivots to
  Model A plus the Model C physics.
- The solver already uses a fast structured dressing → the classical headroom is gone and only the
  iteration-count argument (R0) remains, which is a smaller and harder-won result.
- The iteration already converges in 2–3 steps → nothing for a warm start to save.

Each of these is discoverable in the first six weeks, which is the point of ordering the roadmap this
way.

---

## 4c. What the group's own results contribute

This is not a proposal that arrives from outside; nearly every component exists in the supervisors'
published work already.

**The solver being accelerated.** Møller, Besse, Mazets, Stimming & Mauser, *The dissipative
generalized hydrodynamic equations and their numerical solution*, J. Comput. Phys. (2023) —
[arXiv:2212.12349](https://arxiv.org/abs/2212.12349). Backward semi-Lagrangian transport with a
high-order time-Taylor expansion of the advection fields, IMEX Runge–Kutta and Adams–Moulton
semi-Lagrangian variants, and several literature methods compared for the source terms. This is the
baseline, the code, and the object of Phase 0.

**The transport technology.** Besse's semi-Lagrangian Vlasov work (with Sonnendrücker, JCP 191, 2003)
is the lineage the scheme comes from, and the natural route into the adaptive-rank extension in §4b.5.

**The collision physics and its rate.** Mazets, Schumm & Schmiedmayer, *Breakdown of integrability in
a quasi-one-dimensional ultracold bosonic gas*, PRL 100, 210403 (2008) —
[arXiv:0802.1701](https://arxiv.org/abs/0802.1701) — and Mazets & Schmiedmayer, *Thermalization in a
quasi-1D ultracold bosonic gas*, NJP 12, 055023 (2010) —
[arXiv:0912.4493](https://arxiv.org/abs/0912.4493). These supply the collision integral of Model C
*and* the analytic relaxation rate that the learned operator's linearised spectrum must reproduce.
Without them the ML chapter has no physics test.

**The dimensional-crossover model.** *Extension of the generalized hydrodynamics to the dimensional
crossover regime*, PRL 126, 090602 (2021) — [arXiv:2006.08577](https://arxiv.org/abs/2006.08577) —
is where $n_\ell > 1$ comes from, and therefore where the dressing/collision crossover of §4b.1
actually gets decided.

**The numerical-analysis standards.** Two recent Stimming–Mauser papers set the house style the
thesis should match: *Adaptive absorbing boundary layer for the nonlinear Schrödinger equation*
(with W. Xin), Comput. Methods Appl. Math. (2024), and *A time-splitting method for the
three-dimensional linear Pauli equation* (with T. S. Gutleb and M. Ruggeri), same journal, 2024. Both
are "modify the scheme, preserve the structure, measure the gain" — which is precisely the shape of
the R0/R1 chapters, and a useful precedent when arguing that a learned warm start is a numerical
method rather than a machine-learning application.

**The phase-space theory.** Mauser's Wigner-transform line — Gérard, Markowich, Mauser & Poupaud
(CPAM 1997), and *Coarse-scale representations and smoothed Wigner transforms* with Athanassoulis and
Paul, [arXiv:0804.0259](https://arxiv.org/abs/0804.0259) — is the rigorous backbone for the
Wigner/Vlasov extension, and the reason the "where do I truncate" question has a proper answer in
this group rather than a heuristic one.

---

## 5. The roadmap: turning a working solver into a much cheaper one

Generic, and the part that transfers to any future solver. Each phase has a gate: do not proceed
until it passes.

### Phase 0 — Profile. Do not skip, do not guess.
Instrument the existing solver. Produce a table of wall time per component against
$(N_x, N_\theta)$ and the physical parameters. Then compute the Amdahl ceiling: if the target
operator is 40% of runtime, the *maximum possible* speed-up is 1.7×, and no amount of clever
learning changes that.
**Gate:** a component consumes enough of the runtime to be worth attacking. If nothing does, the
thesis pivots here — and the profile is publishable on its own, because no complexity analysis of
dissipative-GHD solvers exists in print.

### Phase 1 — Isolate the operator behind an interface.
Refactor so the hot operator sits behind an abstract base class with two methods: `__call__` and
`residual`. Exact and learned implementations become drop-in. Nothing else changes; the existing
tests must stay green.
**Gate:** the refactored solver reproduces the old results bit-for-bit (or to stated tolerance).

### Phase 2 — Write the contract before writing the model.
Enumerate what any implementation must satisfy *for any input*: shapes, finiteness, limits,
invariants, symmetries, determinism. Turn each into a test that runs against every implementation,
including deliberately adversarial inputs. This suite is the specification.
**Gate:** the *exact* operator passes its own contract. (It may not — see the worked example, where
the exact BGK operator conserved only to quadrature accuracy. Discovering that is valuable.)

### Phase 3 — Characterise the operator; let it choose the architecture.
| If the operator is… | then… |
|---|---|
| local in $x$ | learn per-point; data multiplies by $N_x$ |
| linear in one argument | learn the *map to a linear operator*, apply linearly |
| a resolvent with a cheap residual | go to R0/R1, not R2 |
| differential (a stencil) | a small convolution beats a neural operator, at a fraction of the cost |
| smooth in its arguments | spectral bias is not a threat |
| equivariant | symmetrise the network |

**Gate:** every structural property is assigned to an architectural device or explicitly declared
unenforceable.

### Phase 4 — Data: sample where you will evaluate.
Cover equilibrium across temperature and coupling; states harvested from real solver trajectories;
and deliberately adversarial states (bimodal, near-empty, near-saturated) held out for testing.
**Gate:** a held-out adversarial set exists and was never trained on.

### Phase 5 — Train, and measure cost from the first line.
The surrogate must be *faster* than what it replaces. Track inference cost alongside accuracy from
the beginning; it constrains architecture size far more than accuracy does.
**Gate:** inference is at least 5× cheaper than the exact call at the target accuracy.

### Phase 6 — Couple in. A-priori accuracy is not a result.
Run long trajectories with the surrogate in the loop. Measure conservation drift, positivity
violations, and time-to-divergence against the exact solver.
**Gate:** a run of $\ge 10^4$ steps is stable, or the instability is characterised.

### Phase 7 — Certify and fall back.
Wire in the residual check (R2) or the preconditioned iteration (R0/R1). Log fallback events.
**Gate:** the hybrid is never worse than the baseline, by construction.

### Phase 8 — Report speed-up at matched accuracy, and the failure map.
Fix the accuracy target first, then measure time. Report out-of-distribution degradation as a curve,
not a number. Publish the failures.

---

## 6. Timeline and chapter map

| Months | Work | Chapter |
|---|---|---|
| 1–2 | Model A: hard-rod GHD solver, MD ground truth, invariant suite | 2. A validated 1D kinetic solver |
| 3 | Phase 0–1 on the group's Lieb–Liniger solver: profile, refactor | 3. Where the time goes |
| 4–6 | Model B at R0/R1: learned warm start and preconditioner for the dressing | 4. Accelerating the dressing |
| 7–8 | Model B at R2: certified surrogate, hybrid, fallback statistics | 5. Certified surrogates |
| 9–10 | Model C: structure-preserving learned collision operator | 6. Learning thermalisation |
| 11 | Physics validation: linearised spectra vs analytic rates | 7. What the learned operator says about the physics |
| 12 | Writing, and the failure map | 8. Limits of the method |

Months 1–3 produce a thesis on their own. Everything after is upside.

---

## 7. Success criteria — fixed before any code

1. Model A reproduces the exact hard-rod solution and MD to stated tolerances.
2. R0/R1 reduce dressing iteration count by a measured factor, with the converged answer unchanged
   to solver tolerance.
3. R2 achieves a stated speed-up **at matched accuracy**, with the fallback rate reported as a
   function of physical parameters.
4. Conservation drift over $10^4$ steps is no worse than the baseline solver's own drift.
5. Structural properties hold to machine precision on adversarial inputs — verified, not asserted.
6. Out-of-distribution behaviour is reported as a degradation curve.
7. The learned collision operator's linearised spectrum is compared against the analytic
   thermalisation rate.

---

## 8. Risks and what each one turns into

| Risk | Becomes |
|---|---|
| No component dominates the runtime | Phase 0's complexity analysis, published as such; thesis pivots to Model A + Model C physics |
| Surrogate not faster at useful accuracy | a measured cost/accuracy crossover curve — a negative result with a number attached |
| Coupled instability | Phase 7's certified hybrid bounds it; the instability analysis becomes a section |
| Poor out-of-distribution behaviour | expected; characterising *where* and relating it to the physics is the contribution |
| ML overshadows the physics | Model A and the spectral validation are physics-first and bracket the thesis at both ends |

---

## 9. Sources

- Doyon, *Lecture notes on generalised hydrodynamics* — [arXiv:1912.08496](https://arxiv.org/abs/1912.08496)
- Doyon & Spohn, *Dynamics of hard rods with initial domain wall state* — [arXiv:1703.05971](https://arxiv.org/abs/1703.05971)
- *Conserved densities of hard rods: microscopic to hydrodynamic solutions* — [arXiv:2407.17067](https://arxiv.org/html/2407.17067)
- *Quasiparticle dynamics and diffusive scale hydrodynamics in an inhomogeneous gas of hard rods* — [arXiv:2603.18522](https://arxiv.org/html/2603.18522)
- Biagetti, Cecile, De Nardis, *Three-stage thermalization of a quasi-integrable system* — [PRResearch 6, 023083 (2024)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.023083)
- Møller, Besse, Mazets, Stimming, Mauser, *The dissipative generalized hydrodynamic equations and their numerical solution* — [arXiv:2212.12349](https://arxiv.org/abs/2212.12349)
- Møller & Schmiedmayer, *iFluid* — [arXiv:2001.02547](https://arxiv.org/abs/2001.02547)
- Bonnemain, Doyon, El, *Generalized hydrodynamics of the KdV soliton gas* — [arXiv:2203.08551](https://arxiv.org/pdf/2203.08551)
- Xiao & Frank, *RelaxNet* — [arXiv:2211.08149](https://arxiv.org/abs/2211.08149)
- Lee, Schotthöfer, Xiao, Krumscheid, Frank, *Structure-preserving operator learning* — [arXiv:2402.16613](https://arxiv.org/pdf/2402.16613)
- *NOWS: neural operator warm starts for accelerating iterative solvers* — [arXiv:2511.02481](https://arxiv.org/html/2511.02481v1)
- *Neural preconditioning via Krylov subspace geometry* — [arXiv:2507.15452](https://arxiv.org/abs/2507.15452)
- *Graph neural preconditioners for iterative solutions of sparse linear systems* — [arXiv:2406.00809](https://arxiv.org/html/2406.00809v1)
- Sonnendrücker, Roche, Bertrand, Ghizzo, *The semi-Lagrangian method for the numerical resolution of the Vlasov equation*, JCP 149, 201 (1999)
