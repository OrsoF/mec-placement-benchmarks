import os
import sys

import pytest

from instances.builder import make_star_instance, make_tree_instance


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture
def small_tree_instance():
    return make_tree_instance(number_of_users=2, neighbors_connection=0)


@pytest.fixture
def connected_tree_instance():
    return make_tree_instance(number_of_users=3, neighbors_connection=1)


@pytest.fixture
def small_star_instance():
    return make_star_instance(
        number_of_groups=1,
        users_per_group=3,
        neighbors_connection=1,
    )
