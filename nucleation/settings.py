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

    max_pierce_points: int | None = 400_000
    """Skip line tracing on a frame with more pierce points than this.

    A wall-clock guard, not a memory one: linking is host-side, so a frame with
    a million singularities stalls the run for minutes with the GPU idle. That
    many is never a vortex count -- it is the Truncated-Wigner vacuum's speckle
    passing the density mask -- so the honest response is to record the counts,
    say linking was skipped, and tell the user to raise mask_threshold."""

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

    density_log: bool = False
    """Log colour scale on the movie's density panels. A low-density feature
    under a bright cloud -- a collision halo is 2-3 decades down -- is simply
    not there on the default linear scale."""

    density_log_floor: float = 1.0e-4
    """Bottom of that log scale, as a fraction of the frame's 99.5th-percentile
    density. Only read when density_log is True."""

    momentum_planes: tuple = ()
    """Plane normals for the momentum-space movie (momentum.gif), one figure
    column each; empty means don't record it. This is the view a scattering
    halo is a sharp feature in -- a shell in k-space, cut as a ring by the
    plane holding the collision axis. Costs one (N,N) transform per plane per
    recorded frame, no 3D FFT (see observer.momentum_plane)."""

    momentum_k_max: float | None = None
    """Crop the momentum panels to |k| <= this. None keeps out to the Nyquist
    wavenumber, which for a collision run is mostly empty space around a small
    ring. Wavenumber is velocity here (hbar = m = 1)."""

    momentum_floor: float = 0.2
    """Bottom of the momentum panels' log scale, in atoms per mode -- ABSOLUTE,
    not a fraction of the top, because the number that matters is fixed: the
    Truncated-Wigner vacuum sits at exactly 1/2 an atom per mode, and the halo
    is what rises out of it. 0.2 puts the floor just under that, so the vacuum
    is one dim tone and everything brighter is signal. A fraction-of-vmax floor
    would move with the saturation knob and bury the halo."""

    momentum_vmax: float | None = None
    """Top of that scale; None uses the data maximum (~3e4, the condensate
    packet centres). ~1e2 reproduces the source paper's Figs. 1-2, which
    saturate those packets on purpose so the halo reads clearly."""

    momentum_ring: float | None = None
    """Draw a dashed circle at this wavenumber on every momentum panel. Set it
    to the expected halo radius (half the relative collision wavenumber) and
    the panel stops being a picture and becomes a check: elastic scattering
    conserves energy, so the shell must sit on that circle."""

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
        max_pierce_points=get("NUCLEATION_MAX_PIERCE_POINTS", defaults.max_pierce_points),
        link_cutoff_dx=get("NUCLEATION_LINK_CUTOFF_DX", defaults.link_cutoff_dx),
        close_cutoff_dx=get("NUCLEATION_CLOSE_CUTOFF_DX", defaults.close_cutoff_dx),
        slice_planes=tuple(get("NUCLEATION_SLICE_PLANES", defaults.slice_planes)),
        slice_trajectory=get("NUCLEATION_SLICE_TRAJECTORY", defaults.slice_trajectory),
        slice_stride=get("NUCLEATION_SLICE_STRIDE", defaults.slice_stride),
        detect_stride=get("NUCLEATION_DETECT_STRIDE", defaults.detect_stride),
        density_log=get("NUCLEATION_DENSITY_LOG", defaults.density_log),
        density_log_floor=get("NUCLEATION_DENSITY_LOG_FLOOR", defaults.density_log_floor),
        momentum_planes=tuple(get("NUCLEATION_MOMENTUM_PLANES", defaults.momentum_planes)),
        momentum_k_max=get("NUCLEATION_MOMENTUM_K_MAX", defaults.momentum_k_max),
        momentum_floor=get("NUCLEATION_MOMENTUM_FLOOR", defaults.momentum_floor),
        momentum_vmax=get("NUCLEATION_MOMENTUM_VMAX", defaults.momentum_vmax),
        momentum_ring=get("NUCLEATION_MOMENTUM_RING", defaults.momentum_ring),
    )
