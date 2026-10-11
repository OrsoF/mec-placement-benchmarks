"""Canonical solution data with lazy historical import compatibility."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray


@dataclass(frozen=True, eq=False, init=False)
class Solution:
    """Owned, read-only arc routing, service placement and demand quantities.

    Columns follow ``problem.network_graph.arc_list`` and integer node indices.
    Construction copies the arrays; feasibility is checked by ``validate``.
    """

    x: FloatArray
    y: FloatArray
    q: FloatArray

    def __init__(self, x: ArrayLike, y: ArrayLike, q: ArrayLike) -> None:
        for name, source in (("x", x), ("y", y), ("q", q)):
            values = np.array(source, dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


_LEGACY_EXPORTS: dict[str, str] = {
    "to_int_array": "mecbench.domain.conversions",
    "split_state": "mecbench.domain.conversions",
    "make_state": "mecbench.domain.conversions",
    "validate_compact_shapes": "mecbench.domain.conversions",
    "compact_to_arc": "mecbench.domain.conversions",
    "basic_solution": "mecbench.solvers.search",
    "basic_state": "mecbench.solvers.search",
    "clip_solution": "mecbench.solvers.search",
    "clip_state": "mecbench.solvers.search",
    "greedy_quantity_lift": "mecbench.solvers.search",
    "repair_solution": "mecbench.solvers.search",
    "SolvingInstance": "mecbench.solvers.workspace",
    "residual_arrays": "mecbench.domain.loads",
    "apply_residuals_to_graph": "mecbench.solvers.graph_state",
    "validate_arc_solution": "mecbench.domain.validation",
}

__all__ = ["Solution", *_LEGACY_EXPORTS]


def __getattr__(name: str) -> Any:
    module = _LEGACY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
