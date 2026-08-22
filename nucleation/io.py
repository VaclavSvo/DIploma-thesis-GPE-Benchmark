"""Writing a nucleation run to disk: a CSV of scalar observables, an .npz of
pierce-point coordinates, and the movies. No fields -- at N=384 one complex
field is ~450MB, and every figure re-renders from the points.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np


def write_observables_csv(rows: list[dict], path: str) -> str:
    if not rows:
        raise ValueError("no observables to write -- the observer recorded no frames")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# Columns that must come back as ints, not floats -- listed explicitly rather
# than inferred from the name, because "n_peak" is a density and "n_lines" is
# a count and a prefix rule cannot tell them apart.
INT_COLUMNS = frozenset({
    "trajectory", "n_pierce", "n_pierce_x", "n_pierce_y", "n_pierce_z",
    "n_lines", "n_lines_closed", "n_lines_open",
})


def read_observables_csv(path: str) -> list[dict]:
    """Round-trip of the above, so a finished run can be re-aggregated (a
    different bootstrap seed, a different onset threshold, a different
    control comparison) without touching the GPU again."""
    with open(path, newline="") as fh:
        return [{k: (int(v) if k in INT_COLUMNS else float(v)) for k, v in raw.items()}
                for raw in csv.DictReader(fh)]


def write_pierce_points(arrays: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def frame_points(npz, index: int):
    """Slice one frame's points back out of write_pierce_points' packed
    arrays: (points, charges, normals) for frame `index`."""
    lo, hi = npz["frame_offset"][index], npz["frame_offset"][index + 1]
    return npz["points"][lo:hi], npz["charges"][lo:hi], npz["normals"][lo:hi]


def write_run_metadata(meta: dict, path: str) -> str:
    """Every knob that could change the answer, next to the answer. A vortex
    count is meaningless without the mask threshold, noise cutoff and dx/xi
    that produced it, and the three convergence sweeps need them recorded per
    run to be comparable afterwards.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return path
