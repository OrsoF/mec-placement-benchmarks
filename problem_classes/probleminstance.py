from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage


class ProblemInstance:
    def __init__(
        self,
        network_graph: NetworkGraph,
        service_package: ServicePackage,
        problem_name: str,
    ) -> None:
        self.network_graph = network_graph
        self.service_package = service_package
        self.problem_name = problem_name

        self.number_of_service: int = self.service_package.number_of_services
        self.demand_per_service: list[int] = [
            service.task_number for service in self.service_package.services
        ]
        self.total_number_of_demands: int = sum(self.demand_per_service)

        self.origins = [
            origin
            for service in self.service_package.services
            for origin in service.origin_nodes
        ]
        self.min_flows = [
            min_flow
            for service in self.service_package.services
            for min_flow in service.min_flows
        ]
        self.max_flows = [
            max_flow
            for service in self.service_package.services
            for max_flow in service.max_flows
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
            service_index
            for service_index in range(self.number_of_service)
            for _ in range(self.demand_per_service[service_index])
        ]

class ToyInstance(ProblemInstance):
    def __init__(self) -> None:
        """Toy Instance"""
        pass