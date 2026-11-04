# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Article planners and configurable experiments. Simulator imports are optional."""
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from .nashopt_controller import NashOptController
from .linear_nash_mpc_controller import LinearNashMPCController
from .blocked_game_mpc import (
    BlockedGameMPC,
    MoveBlocking,
    CooperativeWeights,
    AdversarialWeights,
)
from .cooperative_equilibrium_mpc import CooperativeEquilibriumMPC
from .config import load_config, ExperimentConfig
from .experiments import run_experiment

__version__ = "1.0.0"
__all__ = [
    "NashOptController",
    "LinearNashMPCController",
    "BlockedGameMPC",
    "MoveBlocking",
    "CooperativeWeights",
    "AdversarialWeights",
    "CooperativeEquilibriumMPC",
    "load_config",
    "ExperimentConfig",
    "run_experiment",
]
