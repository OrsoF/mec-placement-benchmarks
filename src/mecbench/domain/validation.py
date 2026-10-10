"""Reference feasibility checks, independent of solver implementations.

Binary routing/placement and continuous quantities match the arc MILP.
Flow conservation permits balanced circulations, as does the formulation.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np


class SolutionValidationError(ValueError):
    """A solution violates its dimensions, variable domains, or constraints."""


@dataclass(frozen=True)
class Violation:
    """A constraint violation with zero-based entity indices and measured values."""

    constraint: str
    message: str
    observed: float | tuple | None = None
    limit: float | tuple | None = None
    demand: int | None = None
    service: int | None = None
    node: int | None = None
    arc: int | None = None
    variable: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    violations: tuple[Violation, ...] = ()

    @property
    def feasible(self) -> bool:
        return not self.violations

    def raise_if_infeasible(self) -> None:
        if self.violations:
            raise SolutionValidationError(self.violations[0].message)


def validate(problem, solution, *, atol=1e-6) -> ValidationReport:
    """Report all violations; malformed arrays stop dependent physical checks.

    ``solution`` exposes x, y and q arrays (normally a ``Solution``). ``None``
    represents absence of a candidate. This never changes the problem or arrays.
    """
    _check_tolerance(atol)
    if solution is None:
        return ValidationReport((Violation("no_solution", "No solution available."),))
    if not all(hasattr(solution, name) for name in ("x", "y", "q")):
        raise TypeError("solution must expose x, y and q arrays, or be None.")
    return _validate_arrays(problem, solution.x, solution.y, solution.q, atol=atol)


def _check_tolerance(atol):
    if not isinstance(atol, Real) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")


def _validate_arrays(problem, x, y, q, *, atol=1e-6) -> ValidationReport:
    _check_tolerance(atol)
    network = problem.network_graph
    demands = problem.total_number_of_demands
    services = problem.number_of_service
    nodes = network.nx_graph.number_of_nodes()
    arcs = network.nx_graph.number_of_edges()
    violations = []
    arrays = {}
    for name, values, shape in (
        ("x", x, (demands, arcs)), ("y", y, (services, nodes)), ("q", q, (demands,)),
    ):
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            violations.append(Violation("numeric", f"{name} must be a numeric array.", variable=name))
            continue
        if values.shape != shape:
            violations.append(Violation(
                "dimensions", f"{name} shape must be {shape}, got {values.shape}.",
                values.shape, shape, variable=name,
            ))
        elif not np.all(np.isfinite(values)):
            violations.append(Violation("finite", f"{name} must contain only finite values.", variable=name))
        else:
            arrays[name] = values
    if violations:
        return ValidationReport(tuple(violations))
    x, y, q = (arrays[name] for name in ("x", "y", "q"))
    for name, values in (("x", x), ("y", y)):
        rounded = np.rint(values)
        if np.any(np.abs(values - rounded) > atol) or np.any((rounded < 0) | (rounded > 1)):
            violations.append(Violation("binary", f"{name} must be binary within tolerance {atol}.", variable=name))
        arrays[name] = rounded
    if violations:
        return ValidationReport(tuple(violations))
    x, y = arrays["x"], arrays["y"]

    def record(mask, constraint, message, observed, limit, entity):
        for index in np.flatnonzero(mask):
            i = int(index)
            violations.append(Violation(
                constraint, message, float(observed[i]), float(limit[i]), **{entity: i},
            ))

    placements = y.sum(axis=1)
    record(placements != 1, "placement", "Each service must be placed on exactly one node.",
           placements, np.ones(services), "service")
    minimum = np.maximum(0, np.asarray(problem.min_flows))
    maximum = np.asarray(problem.max_flows)
    record(q < minimum - atol, "minimum_flow", "Demand quantity below minimum flow.", q, minimum, "demand")
    record(q > maximum + atol, "maximum_flow", "Demand quantity above maximum flow.", q, maximum, "demand")
    node_load = y.T @ np.asarray(problem.needed_resources, dtype=float)
    resources = np.asarray(network.available_resources)
    record(node_load > resources + atol, "resources", "Node resources exceeded.", node_load, resources, "node")
    edge_load = x.T @ q
    capacities = np.asarray(network.capacities)
    record(edge_load > capacities + atol, "capacity", "Arc capacity exceeded.", edge_load, capacities, "arc")

    balance = np.zeros((demands, nodes))
    for a, (source, destination) in enumerate(network.arc_list):
        balance[:, source] += x[:, a]
        balance[:, destination] -= x[:, a]
    expected = -y[np.asarray(problem.demand_to_service, dtype=int)].copy()
    expected[np.arange(demands), np.asarray(problem.origins, dtype=int)] += 1
    for d, node in np.argwhere(np.abs(balance - expected) > atol):
        violations.append(Violation("conservation", "Flow conservation violated.",
                                    float(balance[d, node]), float(expected[d, node]), demand=int(d), node=int(node)))

    arc_latency = np.asarray(network.betas) + np.asarray(network.alphas) * edge_load
    demand_latency = x @ arc_latency
    limits = np.asarray(problem.latencies)
    record(demand_latency > limits + atol, "latency", "Demand latency exceeded under total arc load.",
           demand_latency, limits, "demand")
    return ValidationReport(tuple(violations))


def validate_arc_solution(problem, x, y, q, *, atol=1e-6) -> None:
    """Strict compatibility entry point using the same checks as ``validate``."""
    _validate_arrays(problem, x, y, q, atol=atol).raise_if_infeasible()
