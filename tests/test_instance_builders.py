import networkx as nx

from instances.build_parallel import build_star_instance
from instances.build_tree import make_tree_instance


def assert_valid_problem_instance(instance, expected_services, expected_demands):
    graph = instance.network_graph.nx_graph

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert instance.number_of_service == expected_services
    assert instance.total_number_of_demands == expected_demands
    assert len(instance.origins) == expected_demands
    assert len(instance.min_flows) == expected_demands
    assert len(instance.max_flows) == expected_demands
    assert len(instance.latencies) == expected_demands
    assert len(instance.needed_resources) == expected_services
    assert len(instance.network_graph.available_resources) == graph.number_of_nodes()
    assert len(instance.network_graph.capacities) == graph.number_of_edges()
    assert all(capacity > 0 for capacity in instance.network_graph.capacities)
    assert all(resource >= 0 for resource in instance.network_graph.available_resources)

    instance.check_current_solution()


def test_tree_builder_creates_valid_small_instance():
    users = 3

    instance = make_tree_instance(number_of_users=users, neighbors_connection=1)

    assert instance.problem_name == "tree_3_1"
    assert_valid_problem_instance(
        instance,
        expected_services=users,
        expected_demands=users * 2,
    )


def test_parallel_builder_creates_valid_small_instance():
    groups = 2
    users_per_group = 3

    instance = build_star_instance(
        number_of_groups=groups,
        users_per_group=users_per_group,
        neighbors_connection=1,
    )

    assert instance.problem_name == "tree_parallel_6_1"
    assert_valid_problem_instance(
        instance,
        expected_services=groups * users_per_group,
        expected_demands=groups * users_per_group * 2,
    )
