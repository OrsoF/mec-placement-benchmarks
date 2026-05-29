import networkx as nx
import numpy as np

from instances.builder import make_star_instance, make_tree_instance
from instances.utility import random_services, three_level_graph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service_package import ServicePackage


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

    instance = make_star_instance(
        number_of_groups=groups,
        users_per_group=users_per_group,
        neighbors_connection=1,
    )

    assert instance.problem_name == "star_2_3_1"
    assert_valid_problem_instance(
        instance,
        expected_services=groups * users_per_group,
        expected_demands=groups * users_per_group * 2,
    )


def test_large_builder_creates_valid_synthetic_instance():
    base_graph = nx.path_graph(3)
    n_services = 2

    network_graph, n_core, user_nodes = three_level_graph(
        base_graph,
        edge_per_core=2,
        users_per_core=2,
        user_degree=1,
        rng=np.random.default_rng(0),
    )
    services = random_services(
        network_graph,
        np.r_[np.arange(n_core), user_nodes],
        n_services,
        n_services,
        np.random.default_rng(0),
        latency_scale=25.0,
    )
    package = ServicePackage(services, "large_synthetic")
    instance = ProblemInstance(network_graph, package, "large_synthetic")

    assert_valid_problem_instance(
        instance,
        expected_services=n_services,
        expected_demands=n_services * n_services,
    )
