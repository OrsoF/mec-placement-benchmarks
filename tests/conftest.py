import importlib
import importlib.util

import pytest

from tests.helpers import make_problem, route_solution


def pytest_addoption(parser):
    parser.addoption(
        "--require-extras",
        action="store_true",
        help="fail instead of skipping missing optional dependencies",
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers("requires"):
        for module in marker.args:
            if importlib.util.find_spec(module) is None:
                message = f"Optional dependency {module!r} required by {item.nodeid}"
                if item.config.getoption("--require-extras"):
                    pytest.fail(message)
                pytest.skip(message)
            # Import failures from installed dependencies are failures, not skips.
            importlib.import_module(module)


@pytest.fixture
def shared_arc():
    problem = make_problem()
    return problem, route_solution(problem, [[0, 1], [0, 1]], [1], [2, 2])
