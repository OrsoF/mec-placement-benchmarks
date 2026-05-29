from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.demand import Demand


class Service:
    def __init__(
        self,
        task_list: list[Demand],
        network_graph: NetworkGraph,
        resources_needed: float,
    ):
        if not isinstance(network_graph, NetworkGraph):
            raise TypeError("network_graph must be a NetworkGraph.")
        if resources_needed < 0:
            raise ValueError("resources_needed must be non-negative.")
        if not task_list:
            raise ValueError("task_list must not be empty.")
        if not all(isinstance(task, Demand) for task in task_list):
            raise TypeError("task_list must contain only Demand instances.")

        self.network_graph = network_graph
        self.node_number = self.network_graph.nx_graph.number_of_nodes()
        self.edge_number = self.network_graph.nx_graph.number_of_edges()
        self.needed_resources = resources_needed

        self.task_number = 0
        self.task_list = []

        self.origin_nodes = []
        self.min_flows = []
        self.max_flows = []
        self.max_latencies = []

        for task in task_list:
            self.add_task(task)

    def add_task(self, task_element: Demand):
        self._check_task(task_element)

        self.current_task_element = task_element
        self.task_list.append(self.current_task_element)
        self.task_number += 1

        self.origin_nodes.append(self.current_task_element.origin_node_index)
        self.min_flows.append(self.current_task_element.flow_min)
        self.max_flows.append(self.current_task_element.flow_max)
        self.max_latencies.append(self.current_task_element.max_latency)

    def _check_task(self, task: Demand):
        if not 0 <= task.origin_node_index < self.node_number:
            raise ValueError(
                f"Demand origin node {task.origin_node_index} is outside graph nodes."
            )
