"""Minimal worked example: learning an operator with its structure enforced by construction.

TARGET (the toy version of the GHD collision operator)

    C[f](v) = ( M[f](v) - f(v) ) / tau

the BGK collision operator on a 1D velocity grid, where M[f] is the Maxwellian carrying
the same mass, momentum and energy as f. It has the same signature as the real target --
a function of one variable in, a function of one variable out -- the same structural
identities, and a closed-form ground truth, so every trick can be checked exactly:

    <1, C> = <v, C> = <v^2, C> = 0        (collision invariants)
    C[M] = 0                              (Maxwellian is the fixed point)
    f > 0  =>  the relaxation keeps f > 0

WHAT THIS DEMONSTRATES

Two models trained on identical data:
    plain      -- DeepONet, no structural constraints, conservation left to the loss
    projected  -- the same network, output projected onto the orthogonal complement
                  of span{1, v, v^2} in the quadrature inner product

The table at the end is the whole point. The plain model's conservation error is whatever
training happened to leave; the projected model's is machine epsilon for EVERY input,
including adversarial ones nobody trained on. That difference is not an accuracy
improvement -- it is the difference between a preference and a guarantee.

HONEST LIMITATION, stated because it matters for the real problem: projecting out the
invariants makes the output conserve, but it does NOT force C = 0 at equilibrium. For the
fixed point you need the relaxation form C = (M_theta[f] - f)/tau_theta[f] with M and tau
learned separately (RelaxNet, arXiv:2211.08149). Projection is step one of two.

Run:  python learn_operator_demo.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn

    _Module = nn.Module
except ImportError:  # the numpy core below stays importable and testable without torch
    torch = nn = None

    class _Module:  # placeholder base class
        def __init__(self) -> None:
            raise RuntimeError("this part of the demo needs torch")

SEED = 0


# ---------------------------------------------------------------- grid & exact operator

@dataclass(frozen=True)
class VelocityGrid:
    """Velocity nodes and the quadrature weights that define the inner product.

    The weights are not decoration: every moment, every inner product and the projection
    below must use the SAME quadrature, or the invariants are conserved with respect to a
    rule the operator does not actually satisfy.
    """

    v: np.ndarray
    w: np.ndarray

    @classmethod
    def uniform(cls, v_max: float = 6.0, nv: int = 64) -> "VelocityGrid":
        v = np.linspace(-v_max, v_max, nv)
        dv = v[1] - v[0]
        w = np.full(nv, dv)
        w[0] = w[-1] = dv / 2.0          # trapezoid
        return cls(v=v, w=w)

    def inner(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """<a, b> = sum_i w_i a_i b_i, batched over leading axes."""
        return np.einsum("...i,i,...i->...", a, self.w, b)

    def integrate(self, a: np.ndarray) -> np.ndarray:
        return np.einsum("...i,i->...", a, self.w)


def moments(f: np.ndarray, g: VelocityGrid):
    """(rho, u, T) from a distribution. f may be batched: f[..., nv]."""
    rho = g.integrate(f)
    m = g.integrate(f * g.v)
    e = g.integrate(f * g.v**2)
    u = m / rho
    T = e / rho - u**2
    return rho, u, T


def maxwellian(rho, u, T, g: VelocityGrid) -> np.ndarray:
    rho, u, T = (np.asarray(x)[..., None] for x in (rho, u, T))
    return rho / np.sqrt(2.0 * np.pi * T) * np.exp(-((g.v - u) ** 2) / (2.0 * T))


def bgk(f: np.ndarray, g: VelocityGrid, tau: float = 1.0) -> np.ndarray:
    """The exact operator. Ground truth for training and for every check."""
    return (maxwellian(*moments(f, g), g) - f) / tau


# ---------------------------------------------------------------- structural machinery

def invariant_basis(g: VelocityGrid) -> np.ndarray:
    """Orthonormal basis of span{1, v, v^2} in the quadrature inner product.

    Gram-Schmidt, not the raw monomials: the projection formula below is only exact for an
    orthonormal basis, and {1, v, v^2} is not orthogonal under any quadrature.
    """
    raw = np.stack([np.ones_like(g.v), g.v, g.v**2])
    basis = []
    for phi in raw:
        for b in basis:
            phi = phi - g.inner(b, phi) * b
        phi = phi / np.sqrt(g.inner(phi, phi))
        basis.append(phi)
    return np.stack(basis)


def project_out(c: np.ndarray, basis: np.ndarray, g: VelocityGrid) -> np.ndarray:
    """Remove every component along the collision invariants.

    Exact, linear, cheap, differentiable -- which is why it belongs in the architecture
    rather than in the loss. c may be batched: c[..., nv].
    """
    for b in basis:
        c = c - g.inner(b, c)[..., None] * b
    return c


# ---------------------------------------------------------------- data

def sample_distributions(n: int, g: VelocityGrid, rng, adversarial: bool = False):
    """Random positive f as a mixture of one or two Gaussians.

    The sampling distribution is the single most important design decision in operator
    learning: the model is only trustworthy where you sampled. `adversarial=True` produces
    the far-from-equilibrium bimodal states deliberately EXCLUDED from training, so the
    out-of-distribution column of the results table means something.
    """
    out = np.empty((n, g.v.size))
    for i in range(n):
        k = 2 if (adversarial or rng.random() < 0.4) else 1
        f = np.zeros_like(g.v)
        for _ in range(k):
            if adversarial:
                mu, sig, amp = rng.uniform(-3.5, 3.5), rng.uniform(0.25, 0.6), rng.uniform(0.3, 1.0)
            else:
                mu, sig, amp = rng.uniform(-1.2, 1.2), rng.uniform(0.7, 1.6), rng.uniform(0.3, 1.0)
            f += amp * np.exp(-((g.v - mu) ** 2) / (2.0 * sig**2))
        out[i] = f + 1e-6
    return out


# ---------------------------------------------------------------- model

class DeepONet(_Module):
    """u(y) = sum_k b_k(f) t_k(y).

    branch: the input FUNCTION sampled at fixed sensors -> coefficients
    trunk:  the query POINT -> basis functions

    The branch/trunk split is what makes this an operator rather than a fixed-size vector
    map: the trunk can be evaluated at query points that were never sensors, so the model
    transfers to a finer grid than it was trained on.
    """

    def __init__(self, n_sensors: int, width: int = 128, n_basis: int = 48):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_basis))
        self.trunk = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, n_basis), nn.Tanh())

    def forward(self, f_sensors, y):
        b = self.branch(f_sensors)                  # (B, K)
        t = self.trunk(y[:, None])                  # (Q, K)
        return b @ t.T                              # (B, Q)


def train(model, f_train, c_train, g, basis_t, project: bool, epochs: int = 400, lr: float = 2e-3):
    """Plain supervised fit of the operator. Relative L2, no constraint penalties.

    Note what is NOT here: no conservation term in the loss. The projected model gets its
    conservation from the architecture; the plain model is left to find it on its own,
    which is exactly the comparison being made.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    v_t = torch.tensor(g.v, dtype=torch.float32)
    scale = float(np.abs(c_train).max())
    x = torch.tensor(f_train, dtype=torch.float32)
    y = torch.tensor(c_train / scale, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        out = model(x, v_t)
        if project:
            out = project_out_torch(out, basis_t, g)
        loss = ((out - y) ** 2).mean() / (y**2).mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"    epoch {ep+1:4d}  rel loss {loss.item():.3e}")
    return scale


def project_out_torch(c, basis_t, g: VelocityGrid):
    w = torch.tensor(g.w, dtype=torch.float32)
    for b in basis_t:
        c = c - ((c * w * b).sum(dim=-1, keepdim=True)) * b
    return c


# ---------------------------------------------------------------- evaluation

def conservation_errors(c: np.ndarray, basis: np.ndarray, g: VelocityGrid) -> np.ndarray:
    """max_i |<b_i, c>| / ||c||_w  over the ORTHONORMAL invariant basis, per sample.

    Cauchy-Schwarz bounds this by 1, so it is a genuine relative measure. Using the raw
    monomials {1, v, v^2} instead makes the number scale with v_max^2 and is meaningless.

    Worth knowing before reading the results: the EXACT operator does not hit zero here.
    M[f] is built from analytic formulae, so its *discrete* moments on a truncated grid
    differ slightly from f's, and the exact C inherits that as a quadrature-limited
    conservation error (~1e-4 at v_max=6, falling as the grid is widened). The projected
    network satisfies the discrete identity to machine precision -- i.e. BETTER than the
    ground truth it was trained on. That is the right outcome: what a long run's
    conservation actually depends on is the discrete identity, not the continuum one.
    """
    num = np.abs(np.einsum("bi,i,ki->bk", c, g.w, basis)).max(axis=1)
    norm = np.sqrt(np.einsum("bi,i,bi->b", c, g.w, c))
    return num / (norm + 1e-30)


def main() -> None:
    if torch is None:
        raise SystemExit("torch not installed; the numpy core is still importable")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    g = VelocityGrid.uniform()
    basis = invariant_basis(g)
    basis_t = torch.tensor(basis, dtype=torch.float32)

    f_tr = sample_distributions(2000, g, rng)
    f_id = sample_distributions(300, g, rng)
    f_ood = sample_distributions(300, g, rng, adversarial=True)
    c_tr, c_id, c_ood = (bgk(f, g) for f in (f_tr, f_id, f_ood))

    results = {}
    for name, project in (("plain", False), ("projected", True)):
        print(f"  training {name}")
        model = DeepONet(g.v.size)
        scale = train(model, f_tr, c_tr, g, basis_t, project=project)
        row = {}
        for tag, f, c in (("in-dist", f_id, c_id), ("out-of-dist", f_ood, c_ood)):
            with torch.no_grad():
                pred = model(torch.tensor(f, dtype=torch.float32),
                             torch.tensor(g.v, dtype=torch.float32))
                if project:
                    pred = project_out_torch(pred, basis_t, g)
                pred = pred.numpy() * scale
            row[tag] = (float(np.linalg.norm(pred - c) / np.linalg.norm(c)),
                        float(conservation_errors(pred, basis, g).max()))
        results[name] = row

    print(f"\n{'model':<12}{'set':<14}{'rel L2 err':>12}{'max cons. err':>16}")
    for name, row in results.items():
        for tag, (l2, cons) in row.items():
            print(f"{name:<12}{tag:<14}{l2:12.3e}{cons:16.3e}")
    print("\nThe conservation column is the lesson: trained accuracy for one model, "
          "machine epsilon for the other, on inputs neither model saw.")


if __name__ == "__main__":
    main()
