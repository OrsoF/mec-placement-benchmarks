from problem_classes.service import Service
import networkx as nx


class ServicePackage:
    def __init__(self, services: list[Service], instance_name: str) -> None:
        assert len(services), "services should not be empty."

        self.nx_graph: nx.Graph = services[0].network_graph.nx_graph

        self.services = []
        for service in services:
            self.services.append(service)

        self.instance_name = instance_name

        self.number_of_services: int = len(services)
        self.number_of_task_per_services: int = services[0].task_number
        self.number_of_nodes = self.nx_graph.number_of_nodes()
        self.number_of_edges = self.nx_graph.number_of_edges()
