from problem_classes.networkgraph import NetworkGraph
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
import numpy as np
import networkx as nx
from utils.pickle import pickle_save
import matplotlib.pyplot as plt
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node


class SolvingInstance:
    def __init__(self, problem_instance: ProblemInstance):
        self.problem_instance = problem_instance
        self.graph = problem_instance.network_graph.nx_graph

    def build_initial_solution(self):
        pass
