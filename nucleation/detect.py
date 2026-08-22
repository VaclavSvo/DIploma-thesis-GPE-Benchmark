"""Pierce maps -> vortex lines, counts and line length.

Two stages, both computed every frame:

  Stage 1  count the plaquettes that carry a nonzero winding, per plane
           family. Robust, no linking logic, gives N_pierce(t) immediately.
  Stage 2  merge the pierce points of all three families into one 3D point
           cloud, group them into connected components (= vortex lines), and
           measure each line's length and whether it closes on itself.

Why both: integer counts are small and jumpy, so across a handful of
trajectories their variance swamps the signal. Total vortex line length L(t)
is continuous, varies smoothly, and is the standard quantum-turbulence
observable -- report both, do the statistics on L(t).

Only the pierce-point coordinates (a few hundred (x, y, z, charge) tuples per
frame, kilobytes) and the scalar observables leave this module. psi and the
winding fields are never saved: at N=384 one complex field is ~450MB, and
every figure re-renders from the points alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpe3d.backend import xp, to_numpy
from . import winding as W

# points[:, k] columns, and the integer code stored in `normals`
COL_X, COL_Y, COL_Z = 0, 1, 2
NORMAL_CODE = {"x": 0, "y": 1, "z": 2}


@dataclass
class VortexLine:
    """One connected component of the pierce-point cloud."""
    points: np.ndarray      # (P, 3) physical (x, y, z), ordered along the line
    length: float           # arc length of the traced polyline
    length_mst: float       # total spanning-tree length -- >= length when the
                            # component branches (reconnection, or two lines
                            # passing within the link cutoff)
    closed: bool            # ends meet -> a ring
    mean_charge: float      # signed mean of the constituent windings


@dataclass
class VortexFrame:
    """Everything kept for one (trajectory, time) frame."""
    t: float
    trajectory: int
    points: np.ndarray                 # (P, 3) float32, physical (x, y, z)
    charges: np.ndarray                # (P,)   int8
    normals: np.ndarray                # (P,)   uint8, see NORMAL_CODE
    n_peak: float
    integrality_error: float
    lines: list[VortexLine] = field(default_factory=list)

    # -- scalar observables, the columns that end up in the CSV ------------
    def summary(self) -> dict:
        per_normal = {f"n_pierce_{n}": int((self.normals == c).sum())
                      for n, c in NORMAL_CODE.items()}
        lengths = [ln.length for ln in self.lines]
        closed = [ln for ln in self.lines if ln.closed]
        return dict(
            t=self.t, trajectory=self.trajectory,
            n_pierce=int(self.points.shape[0]), **per_normal,
            n_lines=len(self.lines),
            n_lines_closed=len(closed),
            n_lines_open=len(self.lines) - len(closed),
            L_total=float(sum(lengths)),
            L_closed=float(sum(ln.length for ln in closed)),
            L_max=float(max(lengths)) if lengths else 0.0,
            n_peak=self.n_peak,
            integrality_error=self.integrality_error,
        )


# ---------------------------------------------------------------------------
# Stage 1 -- pierce points
# ---------------------------------------------------------------------------

def _axis_origin(engine) -> float:
    """Physical coordinate of index 0 on every axis (the grid is the same
    linspace on all three -- see GPEPhysics3D._build_grid)."""
    return -engine.L / 2.0


def pierce_points(psi, engine, normal: str, mask, dip_mask=None,
                  integrality_tol: float = 0.25):
    """Pierce points of one plane family.

    Returns (coords (P,3) float32 physical (x, y, z), charges (P,) int8,
    integrality error of this family's winding field).

    `mask` is the *corner* density mask; it is expanded to a plaquette mask
    here (all four corners must be inside the cloud). `dip_mask`, if given,
    additionally requires a local density minimum at at least one of the
    plaquette's corners -- winding plus a density dip is a much stronger
    criterion than either alone.
    """
    u, v = W._TRANSVERSE[normal]
    w = W.winding_volume(psi, normal)
    err = W.integrality_error(w)
    if err > integrality_tol:
        raise ResolutionError(
            f"winding on {normal}-planes is not integer to within {integrality_tol} "
            f"(max deviation {err:.3f}) -- dx is too coarse to resolve the vortex cores. "
            f"Raise N or lower L.")

    charge = xp.rint(w).astype(xp.int8)
    del w
    sel = charge != 0
    sel &= W.plaquette_mask(mask, u, v)
    if dip_mask is not None:
        sel &= W.plaquette_any(dip_mask, u, v)

    iz, iy, ix = (to_numpy(a) for a in xp.nonzero(sel))
    charges = to_numpy(charge[sel]).astype(np.int8)
    del charge, sel

    # Plaquette centre: half a cell up along the two transverse axes, exactly
    # on the plane along the normal axis.
    dx, x0 = engine.dx, _axis_origin(engine)
    half = {"x": (0.0, 0.5, 0.5), "y": (0.5, 0.0, 0.5), "z": (0.5, 0.5, 0.0)}[normal]
    coords = np.empty((ix.size, 3), dtype=np.float32)
    coords[:, COL_X] = x0 + (ix + half[0]) * dx
    coords[:, COL_Y] = x0 + (iy + half[1]) * dx
    coords[:, COL_Z] = x0 + (iz + half[2]) * dx
    return coords, charges, err


class ResolutionError(RuntimeError):
    """Winding numbers came out non-integer -- the grid cannot resolve the
    cores it is being asked to count."""


# ---------------------------------------------------------------------------
# Stage 2 -- linking pierce points into lines
# ---------------------------------------------------------------------------

def _neighbour_pairs(points: np.ndarray, cutoff: float):
    """All index pairs closer than `cutoff`, via a uniform spatial hash.

    O(P * occupancy) instead of O(P^2): at N=384 a busy frame carries a few
    thousand pierce points and the pairwise matrix would be the dominant
    cost of the whole diagnostic.
    """
    if points.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)
    cells = np.floor(points / cutoff).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        buckets.setdefault(c, []).append(i)

    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    cutoff_sq = cutoff * cutoff
    pairs = []
    for cell, members in buckets.items():
        near = []
        for off in offsets:
            other = buckets.get((cell[0] + off[0], cell[1] + off[1], cell[2] + off[2]))
            if other:
                near.extend(other)
        near = np.asarray(near, dtype=np.int64)
        for i in members:
            cand = near[near > i]
            if cand.size == 0:
                continue
            d = points[cand] - points[i]
            hit = cand[(d * d).sum(axis=1) <= cutoff_sq]
            if hit.size:
                pairs.append(np.stack([np.full(hit.size, i, dtype=np.int64), hit], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)


def _components(n: int, pairs: np.ndarray) -> list[np.ndarray]:
    """Connected components via union-find with path compression."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    roots = np.array([find(i) for i in range(n)])
    order = np.argsort(roots, kind="stable")
    roots_sorted = roots[order]
    splits = np.flatnonzero(np.diff(roots_sorted)) + 1
    return np.split(order, splits) if n else []


def _prim_mst(dist: np.ndarray):
    """Minimum spanning tree of a dense distance matrix -> (parent, weight).

    Prim rather than Kruskal because components are small and dense: the
    O(P^2) loop below beats sorting all P^2/2 edges.
    """
    n = dist.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    best[0] = 0.0
    for _ in range(n):
        u = int(np.argmin(np.where(in_tree, np.inf, best)))
        in_tree[u] = True
        upd = (~in_tree) & (dist[u] < best)
        best[upd] = dist[u][upd]
        parent[upd] = u
    return parent, best


def _tree_diameter(adj: list[list[tuple[int, float]]], start: int = 0):
    """Farthest-node path of a weighted tree: two greedy sweeps.

    On an open vortex line this returns the line itself. On a closed loop the
    spanning tree is the loop minus its longest edge, so the path is the loop
    minus one gap -- which is exactly what makes the endpoint-proximity test
    in trace_lines() a reliable ring detector.
    """
    def sweep(src):
        dist = {src: 0.0}
        prev = {src: -1}
        stack = [src]
        while stack:
            u = stack.pop()
            for v, wgt in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + wgt
                    prev[v] = u
                    stack.append(v)
        far = max(dist, key=dist.get)
        return far, dist, prev

    a, _, _ = sweep(start)
    b, dist, prev = sweep(a)
    path = [b]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])
    return path[::-1], dist[b]


def trace_lines(points: np.ndarray, charges: np.ndarray, link_cutoff: float,
                close_cutoff: float | None = None,
                max_component: int = 20000) -> list[VortexLine]:
    """Group pierce points into vortex lines and measure them.

    link_cutoff  -- points closer than this are the same line. A few dx: the
                    three plane families sample the same physical line at
                    spacings of order dx, so ~1.8*dx joins them without
                    bridging two genuinely separate lines.
    close_cutoff -- traced endpoints closer than this mean the line closes on
                    itself (a ring). Defaults to link_cutoff.
    max_component -- components larger than this skip the O(P^2) distance
                    matrix and report a spanning-tree length only, so one
                    pathological frame cannot stall a long run.
    """
    if close_cutoff is None:
        close_cutoff = link_cutoff
    pairs = _neighbour_pairs(points, link_cutoff)
    lines: list[VortexLine] = []

    for comp in _components(points.shape[0], pairs):
        pts = points[comp]
        chg = charges[comp]
        if pts.shape[0] == 1:
            # A single isolated pierce -- a line grazing one plane, or
            # detector noise. Kept (it is a real winding) with zero length,
            # so it shows up in n_lines but cannot inflate L(t).
            lines.append(VortexLine(pts, 0.0, 0.0, False, float(chg[0])))
            continue
        if pts.shape[0] > max_component:
            lines.append(VortexLine(pts, float("nan"), float("nan"), False, float(chg.mean())))
            continue

        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        parent, weight = _prim_mst(d)
        adj: list[list[tuple[int, float]]] = [[] for _ in range(pts.shape[0])]
        for v in range(1, pts.shape[0]):
            u = int(parent[v])
            if u < 0:
                continue
            adj[u].append((v, float(weight[v])))
            adj[v].append((u, float(weight[v])))
        length_mst = float(weight[1:][np.isfinite(weight[1:])].sum())

        path, length = _tree_diameter(adj)
        gap = float(np.linalg.norm(pts[path[0]] - pts[path[-1]]))
        closed = gap <= close_cutoff and len(path) > 3
        if closed:
            length += gap
        lines.append(VortexLine(pts[path], float(length), length_mst, closed,
                                float(chg.mean())))
    return lines


def line_length_from_crossings(n_x: int, n_y: int, n_z: int, dx: float) -> float:
    """Independent estimate of total line length from pierce counts alone.

    A curve of length L crossing a family of planes spaced dx apart makes
    L*<|t.e|>/dx intersections; averaged over orientations <|t.e|> = 1/2, so
    summing the three orthogonal families gives n_x+n_y+n_z = 3L/(2 dx).

    Isotropy-averaged, so it is not a substitute for the traced length -- but
    it needs no linking at all, which makes it the right cross-check on
    whether link_cutoff is set sanely. The two should agree to tens of
    percent; a large disagreement means the linker is merging or splitting
    lines.
    """
    return 2.0 * dx * (n_x + n_y + n_z) / 3.0


# ---------------------------------------------------------------------------
# One frame, end to end
# ---------------------------------------------------------------------------

def detect_frame(psi, engine, t: float, trajectory: int, cfg,
                 density_offset: float = 0.0) -> VortexFrame:
    """Full detection for one single-trajectory (N,N,N) field.

    `cfg` is any object exposing the knobs below -- config.py's NUCLEATION
    block, or nucleation.config.DetectorConfig for tests:
        mask_threshold, mask_close_passes, require_density_dip, dip_factor,
        link_cutoff_dx, close_cutoff_dx, integrality_tol, normals

    `density_offset` is subtracted from |psi|^2 first. For a TW trajectory
    this is solver.offset_density -- the vacuum/thermal noise's own
    "particle" content, which otherwise sits under the mask threshold as a
    uniform floor and drags n_peak around.
    """
    if psi.ndim != 3:
        raise ValueError(f"detect_frame expects one (N,N,N) trajectory, got shape {psi.shape}")

    dens = engine._density(psi)
    if density_offset:
        dens = dens - density_offset
    n_peak = float(xp.max(dens))
    mask = W.density_mask(dens, n_peak, cfg.mask_threshold, cfg.mask_close_passes)
    dip = W.local_density_dip(dens, cfg.dip_factor) if cfg.require_density_dip else None
    del dens

    coords, charges, normals, err = [], [], [], 0.0
    for normal in cfg.normals:
        c, q, e = pierce_points(psi, engine, normal, mask, dip,
                                integrality_tol=cfg.integrality_tol)
        coords.append(c)
        charges.append(q)
        normals.append(np.full(q.size, NORMAL_CODE[normal], dtype=np.uint8))
        err = max(err, e)
    del mask, dip

    points = np.concatenate(coords) if coords else np.empty((0, 3), np.float32)
    charges = np.concatenate(charges) if charges else np.empty(0, np.int8)
    normals = np.concatenate(normals) if normals else np.empty(0, np.uint8)

    frame = VortexFrame(t=t, trajectory=trajectory, points=points, charges=charges,
                        normals=normals, n_peak=n_peak, integrality_error=err)
    frame.lines = trace_lines(points, charges,
                              link_cutoff=cfg.link_cutoff_dx * engine.dx,
                              close_cutoff=cfg.close_cutoff_dx * engine.dx)
    return frame
