"""The explicit output contract consumed by the public API."""

from collections.abc import Callable
from typing import Protocol

from mecbench.domain.contracts import FloatArray


class SolverBackend(Protocol):
    x_var: FloatArray | None
    y_var: FloatArray | None
    q_var: FloatArray | None
    score: float | None
    runtime: float | None
    build_time: float
    termination_condition: str | None

    @property
    def run(self) -> Callable[..., object]: ...
