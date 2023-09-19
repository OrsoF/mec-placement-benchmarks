from problem_classes.networkgraph import NetworkGraph
from problem_classes.demand import Demand

class Service:
    def __init__(self, 
                 task_list : list[Demand], 
                 network_graph : NetworkGraph):
        
        self.network_graph = network_graph
        self.node_number = self.network_graph.nx_graph.number_of_nodes()
        self.edge_number = self.network_graph.nx_graph.number_of_edges()

        self.task_number = 0
        self.task_list = []

        self.origin_nodes = []
        self.min_flows = []
        self.max_flows = []
        self.max_latencies = []
        self.needed_resources = []

        for task in task_list:
            self.add_task(task)

    def add_task(self, 
                 task_element : Demand):
        
        self._check_task(task_element)

        self.current_task_element = task_element
        self.task_list.append(self.current_task_element)
        self.task_number += 1

        self.origin_nodes.append(self.current_task_element.origin_node_index)
        self.min_flows.append(self.current_task_element.flow_min)
        self.max_flows.append(self.current_task_element.flow_max)
        self.max_latencies.append(self.current_task_element.max_latency)
        self.needed_resources.append(self.current_task_element.resources_needed)

    def _check_task(self, task : Demand):
        assert task.origin_node_index < self.node_number