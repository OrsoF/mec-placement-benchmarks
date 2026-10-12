"""Problem data; mutable historical solution access is isolated in compat."""

from mecbench.compat.problem_state import LegacyProblemState
from mecbench.domain.network import NetworkGraph as NetworkGraph
from mecbench.domain.network import check_network_graph as check_network_graph
from mecbench.domain.network import is_graph_symmetric as is_graph_symmetric
from mecbench.domain.services import Demand as Demand
from mecbench.domain.services import Service as Service
from mecbench.domain.services import ServicePackage as ServicePackage


class ProblemInstance(LegacyProblemState):
    def __init__(
        self,
        network_graph: NetworkGraph,
        service_package: ServicePackage,
        problem_name: str,
    ) -> None:
        self.network_graph = network_graph
        self.service_package = service_package
        self.problem_name = problem_name
        self.validate_components()

        self.number_of_service = self.service_package.number_of_services
        self.demand_per_service = [service.task_number for service in self.service_package.services]
        self.total_number_of_demands = sum(self.demand_per_service)

        self.origins = [
            origin for service in self.service_package.services for origin in service.origin_nodes
        ]
        self.min_flows = [
            flow for service in self.service_package.services for flow in service.min_flows
        ]
        self.max_flows = [
            flow for service in self.service_package.services for flow in service.max_flows
        ]
        self.latencies = [
            latency
            for service in self.service_package.services
            for latency in service.max_latencies
        ]
        self.needed_resources = [
            service.needed_resources for service in self.service_package.services
        ]
        self.demand_to_service = [
            k for k, demand_count in enumerate(self.demand_per_service) for _ in range(demand_count)
        ]

        self.seed: int = 0
        self._legacy_state = None

    def validate_components(self) -> None:
        assert isinstance(self.network_graph, NetworkGraph)
        assert isinstance(self.service_package, ServicePackage)
        assert self.problem_name

        graph = self.network_graph.nx_graph
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        assert n > 0
        assert m > 0
        assert len(self.network_graph.available_resources) == n
        assert len(self.network_graph.capacities) == m
        assert len(self.network_graph.alphas) == m
        assert len(self.network_graph.betas) == m
        assert all(resource >= 0 for resource in self.network_graph.available_resources)
        assert all(capacity > 0 for capacity in self.network_graph.capacities)
        assert all(alpha >= 0 for alpha in self.network_graph.alphas)
        assert all(beta >= 0 for beta in self.network_graph.betas)

        services = self.service_package.services
        assert services
        assert self.service_package.nx_graph is graph
        assert self.service_package.number_of_services == len(services)
        assert all(
            service.task_number == self.service_package.number_of_demand_per_service
            for service in services
        )
        for service in services:
            assert service.network_graph is self.network_graph
            assert service.task_number == len(service.task_list)
            assert service.needed_resources >= 0
            assert len(service.origin_nodes) == service.task_number
            assert len(service.min_flows) == service.task_number
            assert len(service.max_flows) == service.task_number
            assert len(service.max_latencies) == service.task_number
            assert all(0 <= origin < n for origin in service.origin_nodes)
            assert all(flow >= 0 for flow in service.min_flows)
            assert all(flow >= 0 for flow in service.max_flows)
            assert all(
                lo <= hi for lo, hi in zip(service.min_flows, service.max_flows, strict=False)
            )
            assert all(latency >= 0 for latency in service.max_latencies)
