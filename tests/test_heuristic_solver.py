import numpy as np

from core.problem_classes.solving_instance import SolvingInstance
from solvers.heuristic.heuristic import Solver as HeuristicSolver
from instances.tree_builder import make_tree_instance


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
