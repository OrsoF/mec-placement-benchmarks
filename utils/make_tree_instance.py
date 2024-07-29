from networkx.readwrite.gml import read_gml
import networkx as nx
from problem_classes.networkgraph import NetworkGraph
from problem_classes.demand import Demand
from problem_classes.service import Service
from problem_classes.servicepackage import ServicePackage
from problem_classes.probleminstance import ProblemInstance
import numpy as np
import os
from utils.pickle import pickle_save
from utils.graph_utils import make_graph_from_gml
import matplotlib.pyplot as plt

# Build graph

nx_graph = nx.Graph([(0, 1), (1, 2), (1, 3)])
network_graph = NetworkGraph(
    base_graph=nx_graph,
    node_resources=[1000, 100, 10, 10],
    edge_capacity=[1000, 1000, 100, 100, 20, 20],
    latency_alpha_factor=[1] * 6,
    latency_beta_factor=[1] * 6,
)

# Create services
# service_1

service_1 = Service(
    task_list=[Demand(0, 0, 20, 2), Demand(2, 0, 20, 2)],
    network_graph=network_graph,
    resources_needed=50,
)

service_2 = Service(
    task_list=[Demand(0, 0, 20, 2), Demand(3, 0, 20, 2)],
    network_graph=network_graph,
    resources_needed=50,
)

PROBLEM_NAME = "tree_0"
INSTANCE_PATH = os.path.join(os.getcwd(), "instances", PROBLEM_NAME + ".pkl")

service_package = ServicePackage([service_1, service_2], PROBLEM_NAME)

# Create instances

problem_instance = ProblemInstance(network_graph, service_package, PROBLEM_NAME)

pickle_save(
    problem_instance, INSTANCE_PATH
)
