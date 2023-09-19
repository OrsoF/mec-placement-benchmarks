from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage

class ProblemInstance:
    def __init__(self, 
                 network_graph : NetworkGraph, 
                 service_package : ServicePackage,
                 problem_name : str) -> None:
        self.network_graph = network_graph
        self.service_package = service_package
        self.problem_name = problem_name