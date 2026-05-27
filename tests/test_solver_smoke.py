import numpy as np

from core.problem_classes.solving_instance import SolvingInstance
from data_io.experiment_io import solve
from solvers.heuristic.heuristic import Solver as HeuristicSolver


def test_solve_supports_legacy_base_heuristic_alias(connected_tree_instance):
    _, solver = solve(
        "unused_when_instance_is_provided",
        "base_heuristic",
        instance=connected_tree_instance,
    )

    assert solver.solver_name == "heuristic"
    assert solver.score >= sum(connected_tree_instance.min_flows)
    assert solver.runtime >= 0


def test_heuristic_solver_smoke_uses_in_memory_instance(connected_tree_instance):
    solver = HeuristicSolver(
        connected_tree_instance,
        limit_shortest_path=1,
        step_no_progression=1,
    )

    solver.run()

    assert np.isclose(solver.score, sum(solver.quantities))
    assert SolvingInstance(connected_tree_instance).test_solution(
        solver.placement,
        solver.routing,
        solver.quantities,
    )


def test_solving_instance_legacy_render_helpers_do_not_require_ilp(
    connected_tree_instance,
):
    solving_instance = SolvingInstance(connected_tree_instance)

    placement, routing, quantities = solving_instance._maximize_quantities_greedy()
    solving_instance.reset_instance()

    assert solving_instance.test_solution(placement, routing, quantities)

