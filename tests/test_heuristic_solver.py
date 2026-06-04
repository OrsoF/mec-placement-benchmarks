import numpy as np

from core.problem_classes.solving_instance import SolvingInstance
from solvers.heuristic.heuristic import Solver as HeuristicSolver
from solvers.heuristic.heuristicprecise import Solver as PreciseHeuristicSolver
from instances.builder import make_tree_instance


def test_heuristic_solver_runs_and_returns_feasible_solution():
    instance = make_tree_instance(number_of_users=2, neighbors_connection=0)
    solver = HeuristicSolver(instance, limit_shortest_path=1, step_no_progression=1)

    solver.run()

    assert solver.score >= sum(instance.min_flows)
    assert solver.runtime >= 0
    assert solver.x_var.shape == (
        instance.total_number_of_demands,
        instance.network_graph.nx_graph.number_of_edges(),
    )
    assert solver.y_var.shape == (
        instance.number_of_service,
        instance.network_graph.nx_graph.number_of_nodes(),
    )
    assert solver.q_var.shape == (instance.total_number_of_demands,)
    assert np.isclose(solver.score, sum(solver.quantities))
    assert SolvingInstance(instance).test_solution(
        solver.placement,
        solver.routing,
        solver.quantities,
    )

    instance._set_arc_solution(solver.x_var, solver.y_var, solver.q_var)
    instance.check_current_solution()


def test_heuristic_solver_is_deterministic_with_fixed_seed():
    instance = make_tree_instance(number_of_users=2, neighbors_connection=0)
    args = dict(
        limit_shortest_path=1,
        step_no_progression=3,
        quantity_increment=1,
        seed=7,
        max_iter=10,
    )

    first = HeuristicSolver(instance, **args).run()
    second = HeuristicSolver(instance, **args).run()

    assert first.placement == second.placement
    assert first.routing == second.routing
    assert first.quantities == second.quantities
    assert first.score == second.score


def test_precise_heuristic_is_quantity_increment_alias():
    instance = make_tree_instance(number_of_users=2, neighbors_connection=0)
    solver = PreciseHeuristicSolver(
        instance,
        limit_shortest_path=1,
        step_no_progression=1,
    )

    assert solver.quantity_increment == 1
    solver.run()

    assert solver.x_var.shape == (
        instance.total_number_of_demands,
        instance.network_graph.nx_graph.number_of_edges(),
    )
