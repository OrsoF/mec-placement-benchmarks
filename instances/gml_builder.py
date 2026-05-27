from pathlib import Path

import numpy as np

from core.graph_utils import make_graph_from_gml
from core.problem_classes.demand import Demand
from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.problem_instance import ProblemInstance
from core.problem_classes.service import Service
from core.problem_classes.service_package import ServicePackage
from data_io.experiment_io import pickle_save

RNG = np.random.default_rng(0)

ALL_GML_PATH = Path.cwd() / "data" / "topologies"
INSTANCE_PATH = Path.cwd() / "artifacts" / "instances"


def estimate_latency_budget(
    network_graph: NetworkGraph,
    number_of_services: int,
) -> float:
    """
    Estimate a feasible latency budget from average edge characteristics.
    """
    g = network_graph.nx_graph

    edge_attributes = list(g.edges.values())

    alpha_mean = np.mean([edge["alpha"] for edge in edge_attributes])
    beta_mean = np.mean([edge["beta"] for edge in edge_attributes])

    return 7.0 * (
        alpha_mean * 5.0 * number_of_services**2 / g.number_of_edges()
        + beta_mean
    )


def make_demands(
    network_graph: NetworkGraph,
    number_of_demands: int,
    max_latency: float,
    rng: np.random.Generator,
) -> list[Demand]:
    """
    Generate a list of stochastic demands.
    """
    g = network_graph.nx_graph
    nodes = list(g.nodes)

    return [
        Demand(
            origin_node_index=int(rng.choice(nodes)),
            flow_min=1,
            flow_max=100,
            max_latency=max_latency,
        )
        for _ in range(number_of_demands)
    ]


def make_services(
    network_graph: NetworkGraph,
    number_of_demands_per_service: int,
    number_of_services: int,
    instance_name: str,
    rng: np.random.Generator = RNG,
) -> ServicePackage:
    """
    Build a ServicePackage with independent demands for each service.
    """
    max_latency = estimate_latency_budget(
        network_graph,
        number_of_services,
    )

    resources_per_service = 5000 // number_of_services

    services: list[Service] = []

    for _ in range(number_of_services):
        demands = make_demands(
            network_graph=network_graph,
            number_of_demands=number_of_demands_per_service,
            max_latency=max_latency,
            rng=rng,
        )

        services.append(
            Service(
                demands=demands,
                network_graph=network_graph,
                resources_needed=resources_per_service,
            )
        )

    return ServicePackage(services, instance_name)


def build_instance_name(
    network_graph: NetworkGraph,
    number_of_services: int,
    number_of_demands_per_service: int,
    graph_name: str,
) -> str:
    g = network_graph.nx_graph

    return (
        f"{g.number_of_nodes()}_"
        f"{g.number_of_edges()}_"
        f"instance_"
        f"{number_of_services}_"
        f"{number_of_demands_per_service}_"
        f"{graph_name}"
    )


def make_problem_instance(
    network_graph: NetworkGraph,
    number_of_services: int,
    number_of_demands_per_service: int,
    graph_name: str,
    rng: np.random.Generator = RNG,
) -> ProblemInstance:
    """
    Build a complete problem instance.
    """
    instance_name = build_instance_name(
        network_graph=network_graph,
        number_of_services=number_of_services,
        number_of_demands_per_service=number_of_demands_per_service,
        graph_name=graph_name,
    )

    service_package = make_services(
        network_graph=network_graph,
        number_of_demands_per_service=number_of_demands_per_service,
        number_of_services=number_of_services,
        instance_name=instance_name,
        rng=rng,
    )

    return ProblemInstance(
        network_graph,
        service_package,
        instance_name,
    )


def save_problem_instance(
    problem_instance: ProblemInstance,
    instance_path: Path,
) -> None:
    """
    Persist a problem instance to disk.
    """
    instance_path.mkdir(parents=True, exist_ok=True)

    save_path = instance_path / f"{problem_instance.instance_name}.pkl"

    pickle_save(problem_instance, str(save_path))


def make_instances_from_gml(
    all_gml_path: Path = ALL_GML_PATH,
    instance_path: Path = INSTANCE_PATH,
    service_counts: list[int] | tuple[int, ...] = (2, 8),
    rng: np.random.Generator = RNG,
) -> None:
    """
    Generate and serialize problem instances from GML topologies.
    """
    gml_files = sorted(all_gml_path.glob("*.gml"))

    for gml_file in gml_files[:1]:
        graph_name = gml_file.stem

        network_graph = make_graph_from_gml(str(gml_file))

        for number_of_services in service_counts:
            number_of_demands_per_service = number_of_services

            problem_instance = make_problem_instance(
                network_graph=network_graph,
                number_of_services=number_of_services,
                number_of_demands_per_service=number_of_demands_per_service,
                graph_name=graph_name,
                rng=rng,
            )

            save_problem_instance(
                problem_instance=problem_instance,
                instance_path=instance_path,
            )
