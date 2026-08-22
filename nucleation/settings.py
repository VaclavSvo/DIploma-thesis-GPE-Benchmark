"""Detector/recorder knobs as one dataclass, plus a builder that reads them off
config.py. Separate from config.py so the detector can be constructed directly
(tests, notebooks, sweeps) without importing a whole scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DetectorConfig:
    # -- detection ---------------------------------------------------------
    mask_threshold: float = 0.08
    """|psi|^2 > mask_threshold * n_peak counts as "inside the condensate".
    A convergence parameter: sweep it and check the count is stable, or the
    detection is not real."""

    mask_close_passes: int = 3
    """Morphological closing applied to the density mask, in cells. Must
    comfortably exceed the core radius (~xi/dx) or the mask deletes the very
    plaquettes that contain vortices -- see winding.density_mask."""

    require_density_dip: bool = True
    """Also require a local density minimum at the core. Costs one separable
    box filter per frame and removes most surviving false positives."""

    dip_factor: float = 0.75
    """Density must be below this times its local 3x3x3 mean."""

    normals: tuple = ("x", "y", "z")
    """Which plane families to sweep. All three catch lines of any
    orientation; drop to ("z",) only as a memory-pressure fallback, and then
    expect rings in the y-z plane to be missed."""

    integrality_tol: float = 0.25
    """Abort if any winding deviates from an integer by more than this --
    the grid's own resolution check."""

    # -- linking -----------------------------------------------------------
    link_cutoff_dx: float = 1.8
    """Pierce points within this many dx belong to the same line."""

    close_cutoff_dx: float = 2.5
    """Traced endpoints within this many dx -> the line is a closed ring."""

    # -- recording ---------------------------------------------------------
    slice_planes: tuple = ("z",)
    """Which mid-planes to record for the density+winding movie. One row of
    the figure per entry."""

    slice_trajectory: int = 0
    """Which trajectory the movie follows. Never an ensemble average -- that
    is exactly what smears vortices away."""

    slice_stride: int = 1
    """Record a movie frame every N blocks. Raise it if the stored slices get
    large: at N=384 a 3-plane frame is ~1MB even stored as float16/int8."""

    detect_stride: int = 1
    """Run the (more expensive) 3D detection every N blocks."""

    def with_overrides(self, **kw) -> "DetectorConfig":
        return replace(self, **kw)


def from_config(cfg_module) -> DetectorConfig:
    """Build a DetectorConfig from config.py's NUCLEATION_* variables,
    falling back to the dataclass defaults for anything not set there."""
    defaults = DetectorConfig()
    get = lambda name, default: getattr(cfg_module, name, default)  # noqa: E731
    return DetectorConfig(
        mask_threshold=get("NUCLEATION_MASK_THRESHOLD", defaults.mask_threshold),
        mask_close_passes=get("NUCLEATION_MASK_CLOSE_PASSES", defaults.mask_close_passes),
        require_density_dip=get("NUCLEATION_REQUIRE_DENSITY_DIP", defaults.require_density_dip),
        dip_factor=get("NUCLEATION_DIP_FACTOR", defaults.dip_factor),
        normals=tuple(get("NUCLEATION_NORMALS", defaults.normals)),
        integrality_tol=get("NUCLEATION_INTEGRALITY_TOL", defaults.integrality_tol),
        link_cutoff_dx=get("NUCLEATION_LINK_CUTOFF_DX", defaults.link_cutoff_dx),
        close_cutoff_dx=get("NUCLEATION_CLOSE_CUTOFF_DX", defaults.close_cutoff_dx),
        slice_planes=tuple(get("NUCLEATION_SLICE_PLANES", defaults.slice_planes)),
        slice_trajectory=get("NUCLEATION_SLICE_TRAJECTORY", defaults.slice_trajectory),
        slice_stride=get("NUCLEATION_SLICE_STRIDE", defaults.slice_stride),
        detect_stride=get("NUCLEATION_DETECT_STRIDE", defaults.detect_stride),
    )
