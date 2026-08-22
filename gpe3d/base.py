"""The fixed 4-slot solver contract. The runner and the gates only ever talk
to this interface, never to a solver's internals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class State:
    """Mutable simulation state threaded through call()."""
    psi: Any            # complex64, (N,N,N) or (B,N,N,N) for a TW batch
    t: float = 0.0       # simulated time elapsed (natural units)
    step: int = 0        # sub-step counter


class Solver(Protocol):
    def units(self) -> dict:
        """Unit system + characteristic scales."""
        ...

    def init_state(self, params: dict) -> State:
        """Build the initial wavefunction from params."""
        ...

    def call(self, state: State, n_steps: int) -> State:
        """Advance state by n_steps sub-steps."""
        ...

    def get_derived_quantities(self, state: State) -> dict:
        """Norm, energy, momentum, angular momentum, ... as plain floats."""
        ...
