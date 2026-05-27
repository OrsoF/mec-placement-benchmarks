from core.problem_classes.network_graph import NetworkGraph
from core.problem_classes.demand import Demand


class Service:
    def __init__(
        self,
        task_list: list[Demand],
        network_graph: NetworkGraph,
        resources_needed: float,
    ):
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
        assert task.origin_node_index < self.node_number
