from itertools import product
import numpy as np
import pytest
from mecbench.domain.validation import SolutionValidationError, validate_arc_solution
from mecbench.domain.solution import SolvingInstance, validate_arc_solution as legacy_validate_arc_solution
from mecbench.solvers.heuristics.fast_heuristic import Solver as FastHeuristic
from tests.helpers import make_problem, route_solution

def test_rl_state_feasibility_uses_reference_validator(shared_arc):
    from mecbench.environments.path_cache import ShortestPathCache
    from mecbench.environments.state_codec import is_feasible_state

    problem, _ = shared_arc
    problem.latencies = [3, 3]
    checker = SolvingInstance(problem)
    cache = ShortestPathCache(checker.graph)
    assert is_feasible_state(problem, checker, cache, [1, 0, 0, 1, 1])
    assert not is_feasible_state(problem, checker, cache, [1, 0, 0, 2, 2])



def test_problem_instance_current_solution_uses_reference_validator():
    from mecbench.domain.problem import Demand
    from mecbench.domain.problem import ProblemInstance
    from mecbench.domain.problem import Service
    from mecbench.domain.problem import ServicePackage

    network = make_problem().network_graph
    service = Service([Demand(0, 1, 10, 3), Demand(0, 1, 10, 3)], network, 2)
    problem = ProblemInstance(network, ServicePackage([service], "shared"), "shared")
    checker = SolvingInstance(problem)
    problem._set_arc_solution(*checker.solution_to_arc_variables([1], [0, 0], [1, 1]))
    problem.check_current_solution()
    problem._set_arc_solution(*checker.solution_to_arc_variables([1], [0, 0], [2, 2]))
    with pytest.raises(SolutionValidationError, match="latency"):
        problem.check_current_solution()
