import pytest
import networkx as nx

from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph, is_graph_symmetric
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage
from core.problem_classes.solving_instance import SolvingInstance


def make_two_node_problem():
    graph = nx.Graph()
    graph.add_edge("user", "cloud")
    network = NetworkGraph(
        base_graph=graph,
        node_resources=[1, 10],
        edge_capacity=[10],
        latency_alpha_factor=[1],
        latency_beta_factor=[1],
    )
    demand = Demand(origin_node_index=0, flow_min=1, flow_max=3, max_latency=10)
    service = Service(task_list=[demand], network_graph=network, resources_needed=1)
    package = ServicePackage([service], "two_node")
    return ProblemInstance(network, package, "two_node")


def test_network_graph_relabels_and_makes_graph_symmetric():
    instance = make_two_node_problem()
    graph = instance.network_graph.nx_graph

    assert set(graph.nodes) == {0, 1}
    assert is_graph_symmetric(graph)
    assert graph.has_edge(0, 1)
    assert graph.has_edge(1, 0)
    assert instance.network_graph.available_resources == [1, 10]
    assert instance.network_graph.capacities == [10, 10]


def test_network_graph_duplicates_undirected_edge_attributes_by_reverse_pair():
    graph = nx.path_graph(["user", "edge", "cloud"])

    network = NetworkGraph(
        base_graph=graph,
        node_resources=[1, 5, 10],
        edge_capacity=[10, 20],
        latency_alpha_factor=[1, 2],
        latency_beta_factor=[3, 4],
    )

    assert network.nx_graph.edges[0, 1]["capacity"] == 10
    assert network.nx_graph.edges[1, 0]["capacity"] == 10
    assert network.nx_graph.edges[1, 2]["capacity"] == 20
    assert network.nx_graph.edges[2, 1]["capacity"] == 20
    assert network.capacities == [10, 10, 20, 20]
    assert network.alphas == [1, 1, 2, 2]
    assert network.betas == [3, 3, 4, 4]


def test_demand_and_service_collect_task_vectors():
    instance = make_two_node_problem()
    service = instance.service_package.services[0]

    assert service.task_number == 1
    assert service.origin_nodes == [0]
    assert service.min_flows == [1]
    assert service.max_flows == [3]
    assert service.max_latencies == [10]


def test_demand_rejects_invalid_values():
    with pytest.raises(ValueError):
        Demand(origin_node_index=-1, flow_min=1, flow_max=2, max_latency=10)
    with pytest.raises(ValueError):
        Demand(origin_node_index=0, flow_min=-1, flow_max=2, max_latency=10)
    with pytest.raises(ValueError):
        Demand(origin_node_index=0, flow_min=3, flow_max=2, max_latency=10)
    with pytest.raises(ValueError):
        Demand(origin_node_index=0, flow_min=1, flow_max=2, max_latency=-1)
    with pytest.raises(TypeError):
        Demand(origin_node_index=0.5, flow_min=1, flow_max=2, max_latency=10)


def test_service_rejects_demand_from_unknown_node():
    instance = make_two_node_problem()
    bad_demand = Demand(origin_node_index=99, flow_min=1, flow_max=2, max_latency=10)

    with pytest.raises(ValueError):
        Service(
            task_list=[bad_demand],
            network_graph=instance.network_graph,
            resources_needed=1,
        )


def test_service_rejects_invalid_inputs():
    instance = make_two_node_problem()
    demand = Demand(origin_node_index=0, flow_min=1, flow_max=2, max_latency=10)

    with pytest.raises(ValueError):
        Service(task_list=[demand], network_graph=instance.network_graph, resources_needed=-1)
    with pytest.raises(ValueError):
        Service(task_list=[], network_graph=instance.network_graph, resources_needed=1)
    with pytest.raises(TypeError):
        Service(task_list=[object()], network_graph=instance.network_graph, resources_needed=1)
    with pytest.raises(TypeError):
        Service(task_list=[demand], network_graph=object(), resources_needed=1)


def test_service_package_rejects_empty_services():
    with pytest.raises(ValueError):
        ServicePackage([], "empty")


def test_service_package_rejects_inconsistent_services():
    instance = make_two_node_problem()
    service = instance.service_package.services[0]
    another_service = Service(
        task_list=[Demand(origin_node_index=0, flow_min=1, flow_max=2, max_latency=10)],
        network_graph=instance.network_graph,
        resources_needed=1,
    )
    different_task_count_service = Service(
        task_list=[
            Demand(origin_node_index=0, flow_min=1, flow_max=2, max_latency=10),
            Demand(origin_node_index=1, flow_min=1, flow_max=2, max_latency=10),
        ],
        network_graph=instance.network_graph,
        resources_needed=1,
    )
    graph = nx.Graph()
    graph.add_edge("a", "b")
    different_network = NetworkGraph(
        base_graph=graph,
        node_resources=[1, 10],
        edge_capacity=[10],
        latency_alpha_factor=[1],
        latency_beta_factor=[1],
    )
    different_network_service = Service(
        task_list=[Demand(origin_node_index=0, flow_min=1, flow_max=2, max_latency=10)],
        network_graph=different_network,
        resources_needed=1,
    )

    with pytest.raises(ValueError):
        ServicePackage([service], "")
    with pytest.raises(TypeError):
        ServicePackage([object()], "bad")
    with pytest.raises(ValueError):
        ServicePackage([service, different_task_count_service], "bad")
    with pytest.raises(ValueError):
        ServicePackage([another_service, different_network_service], "bad")


def test_problem_and_solving_instance_basic_solution_is_feasible():
    instance = make_two_node_problem()
    solving_instance = SolvingInstance(instance)

    placement, routing, quantities = solving_instance.basic_placement_and_routing()

    assert placement == [1]
    assert routing == [0]
    assert quantities == [1]
    assert solving_instance.test_solution(placement, routing, quantities)
    instance.check_current_solution()


def test_check_current_solution_rejects_invalid_placement_assignment():
    instance = make_two_node_problem()
    instance.y_variable[0, :] = [1, 1]

    with pytest.raises(AssertionError):
        instance.check_current_solution()


def test_check_current_solution_rejects_invalid_capacity():
    instance = make_two_node_problem()
    instance.q_variable[0] = 11

    with pytest.raises(AssertionError):
        instance.check_current_solution()


def test_check_current_solution_rejects_quantity_below_minimum():
    instance = make_two_node_problem()
    instance.q_variable[0] = 0

    with pytest.raises(AssertionError):
        instance.check_current_solution()


def test_check_current_solution_rejects_quantity_above_maximum():
    instance = make_two_node_problem()
    instance.q_variable[0] = 4

    with pytest.raises(AssertionError):
        instance.check_current_solution()


def test_check_current_solution_rejects_invalid_latency():
    instance = make_two_node_problem()
    instance.network_graph.alphas = [10, 10]

    with pytest.raises(AssertionError):
        instance.check_current_solution()


def test_check_current_solution_rejects_invalid_routing():
    instance = make_two_node_problem()
    instance.x_variable[0, :] = [0, 1]

    with pytest.raises(AssertionError):
        instance.check_current_solution()
