"""Vortex nucleation diagnostics: detect, count and measure quantised vortices
in the trajectories the gpe3d/ solvers produce. Nothing here changes the
physics -- these modules only read psi.

  winding.py     plaquette phase winding -> pierce maps
  detect.py      pierce maps -> vortex lines, counts, line length
  observer.py    the per-frame recorder that plugs into the evolution loops
  settings.py    DetectorConfig, and how config.py maps onto it
  statistics.py  aggregation and bootstrap CIs across trajectories
  visualise.py   the density + winding movie (needs psi, so made during a run)
  vortex_map.py  three-view map of every core, from the saved points alone --
                 re-renderable after a run, with a CLI
  io.py          writing a run's results to disk
"""
from .settings import DetectorConfig, from_config
from .observer import NucleationObserver, plane_index
from .detect import VortexFrame, VortexLine, detect_frame, trace_lines
from .winding import winding_volume, winding_slice, integrality_error
from .visualise import save_density_winding_gif
from .vortex_map import save_vortex_map

__all__ = [
    "DetectorConfig", "from_config",
    "NucleationObserver", "plane_index",
    "VortexFrame", "VortexLine", "detect_frame", "trace_lines",
    "winding_volume", "winding_slice", "integrality_error",
    "save_density_winding_gif", "save_vortex_map",
]
