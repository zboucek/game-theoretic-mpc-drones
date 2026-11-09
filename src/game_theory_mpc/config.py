# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Strict, portable configuration for the five article studies (SI units)."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import tomllib


def construct(cls, values):
    try:
        return cls(**values)
    except TypeError as error:
        raise ValueError(f"Invalid {cls.__name__}: {error}") from error


@dataclass(frozen=True)
class ModelConfig:
    states: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    pursuers: int = 0
    blocks: list = field(default_factory=lambda: [1, 1, 2, 4, 6, 6])
    dt: float = 0.1
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    repulsion_weight: float = 1.0
    effort_weight: float = 0.1
    terminal_scale: float = 1.0
    input_bound: float = 2.0
    speed_bound: float = 2.0
    jerk_bound: float = 6.0
    separation: float = 0.55
    altitude_bounds: list = field(default_factory=lambda: [0.35, 1.95])
    max_iterations: int = 200
    response_sweeps: int = 3
    response_fractions: list = field(default_factory=lambda: [0.7, 0.35, 1.0])
    max_sweeps: int = 30
    # All six coefficients are used only when pursuers > 0.
    adversarial_weights: dict = field(
        default_factory=lambda: dict(
            goal=100.0,
            evasion=10.0,
            pursuit=10.0,
            repulsion=2.0,
            obstacle=5.0,
            effort=0.1,
        )
    )


@dataclass(frozen=True)
class ExecutionConfig:
    updates: int = 300
    plant: str = "pybullet"
    pid_hz: int = 120
    physics_hz: int = 240
    pid_mode: str = "limited"
    reserve_retry: bool = True
    physical_speed_bound: float = 2.5
    physical_separation: float = 0.5
    physical_altitude_bounds: list = field(default_factory=lambda: [0.3, 2.0])
    emergency_jerk_bound: float = 20.0
    recovery_speed_margin: float = 0.2


@dataclass(frozen=True)
class ControlledConfig:
    agents: int = 3
    horizon: int = 5
    dt: float = 0.1
    bound: float = 5.0
    separation: float = 0.5
    effort_weight: float = 0.1
    repulsion_offset: float = 0.01
    solver_tolerance: float = 1e-12
    max_evaluations: int = 1000
    solve_limit_s: float = 30.0
    kkt_tolerance: float = 1e-6
    feasibility_tolerance: float = 1e-8
    activity_tolerance: float = 1e-6
    deviation_iterations: int = 200
    deviation_limit_s: float = 5.0
    unique_control_tolerance: float = 1e-5
    updates: int = 60
    starts: list = field(
        default_factory=lambda: [
            "zero",
            "braking",
            "lateral_plus",
            "lateral_minus",
            "vertical_plus",
            "vertical_minus",
        ]
    )
    loop_starts: list = field(default_factory=lambda: ["braking", "lateral_plus"])
    deviations: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int = 5
    warmups: int = 1


@dataclass(frozen=True)
class LaboratoryConfig:
    csv: str = "data/hardware/run.csv"
    horizon: int = 10
    dt: float = 0.1
    q_track: float = 10.0
    q_du: float = 0.1
    backend: str = "fast"
    updates: int = 20
    states: list = field(default_factory=list)
    targets: list = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentConfig:
    study: str
    seed: int = 20261023
    methods: list = field(default_factory=list)
    model: ModelConfig | None = None
    execution: ExecutionConfig | None = None
    controlled: ControlledConfig | None = None
    benchmark: BenchmarkConfig | None = None
    laboratory: LaboratoryConfig | None = None
    cases: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    def validate(self):
        import numpy as np

        allowed = {
            "cooperative",
            "pursuit-evasion",
            "controlled",
            "scalability",
            "laboratory",
        }
        if self.study not in allowed:
            raise ValueError(f"Unknown study: {self.study}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        required = dict(
            cooperative={"model", "execution"},
            **{
                "pursuit-evasion": {"model", "execution"},
                "controlled": {"controlled"},
                "scalability": {"model", "benchmark"},
                "laboratory": {"laboratory"},
            },
        )[self.study]
        present = {
            k
            for k in (
                "model",
                "execution",
                "controlled",
                "benchmark",
                "laboratory",
            )
            if getattr(self, k) is not None
        }
        if present != required:
            raise ValueError(
                f"{self.study} requires exactly these sections: {sorted(required)}"
            )

        def positive(value, name, integer=False):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (integer and type(value) is not int)
            ):
                raise ValueError(
                    f'{name} must be a positive {"integer" if integer else "number"}'
                )

        def geometry(states, targets, pursuers=0):
            x, y = np.asarray(states, float), np.asarray(targets, float)
            if x.ndim != 2 or x.shape[1] != 6 or len(x) < 2 or not np.isfinite(x).all():
                raise ValueError("states must have finite shape (N>=2,6)")
            if type(pursuers) is not int or not 0 <= pursuers < len(x):
                raise ValueError("pursuers must be an integer in [0,N)")
            if y.shape != (len(x) - pursuers, 3) or not np.isfinite(y).all():
                raise ValueError("targets must have finite shape (N-pursuers,3)")
            return len(x)

        if self.methods and (
            len(set(self.methods)) != len(self.methods)
            or set(self.methods) - {"gne", "vgne", "potential", "adversarial_br"}
        ):
            raise ValueError("Invalid or duplicate methods")
        expected_methods = {
            "cooperative": {"gne", "vgne", "potential"},
            "pursuit-evasion": {"adversarial_br"},
            "controlled": {"gne", "vgne", "potential"},
        }
        if self.study in expected_methods and (
            not self.methods or set(self.methods) - expected_methods[self.study]
        ):
            raise ValueError(f"Unsupported methods for {self.study}")
        if self.study in ("scalability", "laboratory") and self.methods:
            raise ValueError(
                "Use case.method for scalability; laboratory uses laboratory.backend"
            )
        if self.model:
            m = self.model
            for k in (
                "dt",
                "input_bound",
                "speed_bound",
                "jerk_bound",
                "separation",
                "effort_weight",
            ):
                positive(getattr(m, k), "model." + k)
            for k in ("max_iterations", "response_sweeps", "max_sweeps"):
                positive(getattr(m, k), "model." + k, True)
            if not m.response_fractions or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 < v <= 1
                for v in m.response_fractions
            ):
                raise ValueError("model.response_fractions must be nonempty and lie in (0, 1]")
            for k in (
                "position_weight",
                "velocity_weight",
                "repulsion_weight",
                "terminal_scale",
            ):
                v = getattr(m, k)
                if (
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v < 0
                ):
                    raise ValueError("Invalid model." + k)
            if not m.blocks or any(type(v) is not int or v <= 0 for v in m.blocks):
                raise ValueError(
                    "model.blocks must contain positive integers; their sum is the horizon"
                )
            if (
                len(m.altitude_bounds) != 2
                or not np.isfinite(m.altitude_bounds).all()
                or m.altitude_bounds[0] >= m.altitude_bounds[1]
            ):
                raise ValueError("Invalid model.altitude_bounds")
            if np.asarray(m.obstacles).size and (
                np.asarray(m.obstacles).ndim != 2
                or np.asarray(m.obstacles).shape[1] != 3
                or not np.isfinite(m.obstacles).all()
            ):
                raise ValueError("obstacles must have finite shape (K,3)")
            from .blocked_game_mpc import AdversarialWeights

            construct(AdversarialWeights, m.adversarial_weights)
            if self.study == "scalability":
                if m.states or m.targets or m.pursuers or m.obstacles:
                    raise ValueError(
                        "For scalability, put geometry in cases, not model"
                    )
            else:
                geometry(m.states, m.targets, m.pursuers)
                if (self.study == "pursuit-evasion") != bool(m.pursuers):
                    raise ValueError(
                        "Pursuit-evasion needs pursuers; cooperation needs zero pursuers"
                    )
        if self.study == "scalability" and self.model.blocks != [1, 1, 2, 4, 6, 6]:
            raise ValueError("Set blocks in each scalability case, not in model")
        if self.execution:
            e = self.execution
            positive(e.updates, "execution.updates", True)
            if e.plant not in ("nominal", "pybullet") or e.pid_mode not in (
                "legacy",
                "feedforward",
                "limited",
            ):
                raise ValueError("Invalid plant or PID mode")
            for k in ("pid_hz", "physics_hz"):
                positive(getattr(e, k), k, True)
            ticks = self.model.dt * e.pid_hz
            if (
                e.physics_hz % e.pid_hz
                or not math.isclose(ticks, round(ticks), abs_tol=1e-10)
                or ticks < 1
            ):
                raise ValueError(
                    "PyBullet timing needs integral physics/PID and PID/planning ratios"
                )
            for k in (
                "physical_speed_bound",
                "physical_separation",
                "emergency_jerk_bound",
                "recovery_speed_margin",
            ):
                positive(getattr(e, k), k)
            if type(e.reserve_retry) is not bool:
                raise ValueError("reserve_retry must be boolean")
            a = e.physical_altitude_bounds
            if (
                len(a) != 2
                or not np.isfinite(a).all()
                or not a[0] < a[1]
                or a[0] > self.model.altitude_bounds[0]
                or a[1] < self.model.altitude_bounds[1]
            ):
                raise ValueError("Physical altitude bounds must contain planned bounds")
            if (
                e.physical_separation > self.model.separation
                or e.physical_speed_bound < self.model.speed_bound
            ):
                raise ValueError("Physical envelope must contain planned bounds")
        if self.controlled:
            c = self.controlled
            if type(c.agents) is not int or c.agents != 3:
                raise ValueError("The controlled study supports exactly three agents")
            for k in ("horizon", "max_evaluations", "deviation_iterations", "updates"):
                positive(getattr(c, k), k, True)
            for k in (
                "dt",
                "bound",
                "separation",
                "effort_weight",
                "repulsion_offset",
                "solver_tolerance",
                "solve_limit_s",
                "kkt_tolerance",
                "feasibility_tolerance",
                "activity_tolerance",
                "deviation_limit_s",
                "unique_control_tolerance",
            ):
                positive(getattr(c, k), k)
            if type(c.deviations) is not bool:
                raise ValueError("deviations must be boolean")
            from experiments.equilibrium_selection.protocol import INITIALIZATIONS

            if (
                not c.starts
                or len(c.starts) != len(set(c.starts))
                or len(c.loop_starts) != len(set(c.loop_starts))
                or set(c.starts) - set(INITIALIZATIONS)
                or not c.loop_starts
                or set(c.loop_starts) - set(INITIALIZATIONS)
            ):
                raise ValueError("Unknown controlled initialization")
        if self.study in ("controlled", "scalability"):
            if not self.cases:
                raise ValueError("Study requires cases")
            ids = set()
            for row in self.cases:
                if (
                    not isinstance(row, dict)
                    or not {"id", "states", "targets"} <= row.keys()
                ):
                    raise ValueError("Each case requires id, states and targets")
                allowed_keys = dict(
                    controlled={"id", "states", "targets", "geometry"},
                    scalability={
                        "id",
                        "states",
                        "targets",
                        "blocks",
                        "pursuers",
                        "method",
                        "parameterization",
                        "obstacles",
                        "velocity_weight",
                    },
                )[self.study]
                if set(row) - allowed_keys:
                    raise ValueError("Unknown case fields")
                name = row.get("id", "")
                if (
                    not name
                    or any(
                        ch
                        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                        for ch in name
                    )
                    or name in ids
                ):
                    raise ValueError("Case IDs must be unique simple file names")
                ids.add(name)
                n = geometry(row["states"], row["targets"], row.get("pursuers", 0))
                if self.study == "controlled" and (
                    n != 3 or row.get("pursuers", 0)
                ):
                    raise ValueError(
                        "The controlled study requires three cooperative agents"
                    )
                if self.study == "controlled" and row.get("geometry") not in (
                    "head_on",
                    "crossing",
                    "three_way",
                ):
                    raise ValueError(
                        "Controlled cases require geometry for initialization axes"
                    )
                if self.study == "scalability":
                    if row.get("method") not in ("gne", "vgne", "adversarial_br") or (
                        row["method"] == "adversarial_br"
                    ) != bool(row.get("pursuers", 0)):
                        raise ValueError("Invalid benchmark method/pursuers")
                    if "velocity_weight" in row and (
                        not isinstance(row["velocity_weight"], (int, float))
                        or not math.isfinite(row["velocity_weight"])
                        or row["velocity_weight"] < 0
                    ):
                        raise ValueError("Invalid case velocity_weight")
                    if not row.get("blocks") or any(
                        type(v) is not int or v < 1 for v in row["blocks"]
                    ):
                        raise ValueError("Benchmark cases require positive blocks")
        elif self.cases:
            raise ValueError("This study uses its section geometry, not cases")
        if self.benchmark:
            positive(self.benchmark.repeats, "repeats", True)
            positive(self.benchmark.warmups, "warmups", True)
        if self.laboratory:
            l = self.laboratory
            for k in ("horizon", "updates"):
                positive(getattr(l, k), k, True)
            for k in ("dt", "q_track", "q_du"):
                positive(getattr(l, k), k)
            if l.backend not in ("fast", "nashopt"):
                raise ValueError("Invalid linear backend")
            geometry(l.states, l.targets)
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        value = tomllib.load(stream)
    for key, cls in dict(
        model=ModelConfig,
        execution=ExecutionConfig,
        controlled=ControlledConfig,
        benchmark=BenchmarkConfig,
        laboratory=LaboratoryConfig,
    ).items():
        if key in value:
            value[key] = construct(cls, value[key])
    return construct(ExperimentConfig, value).validate()
