import pytest
from tests.helpers import make_problem, route_solution

@pytest.fixture
def shared_arc():
    problem = make_problem()
    return problem, route_solution(problem, [[0, 1], [0, 1]], [1], [2, 2])
