from core.problem_classes.service import Service
import networkx as nx


class ServicePackage:
    """A list of Services."""

    def __init__(self, services: list[Service], instance_name: str) -> None:
        if not services:
            raise ValueError("services must not be empty.")
        if not instance_name:
            raise ValueError("instance_name must not be empty.")
        if not all(isinstance(service, Service) for service in services):
            raise TypeError("services must contain only Service instances.")

        first_network_graph = services[0].network_graph
        first_task_number = services[0].task_number
        for service in services:
            if service.network_graph is not first_network_graph:
                raise ValueError("All services must use the same NetworkGraph instance.")
            if service.task_number != first_task_number:
                raise ValueError(
                    "All services must have the same number of demands."
                )

        self.nx_graph: nx.Graph = services[0].network_graph.nx_graph

        self.services: list[Service] = []
        for service in services:
            self.services.append(service)

        self.instance_name = instance_name

        self.number_of_services: int = len(services)
        self.number_of_demand_per_service: int = services[0].task_number
        self.number_of_nodes = self.nx_graph.number_of_nodes()
        self.number_of_edges = self.nx_graph.number_of_edges()
