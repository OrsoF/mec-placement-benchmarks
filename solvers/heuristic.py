import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from problem_classes.service import Service
from itertools import islice
from problem_classes.probleminstance import ProblemInstance
from utils.generic_solver import GenericSolver
from utils.graph_utils import path_node_to_path_tuple, path_tuple_to_path_node
from problem_classes.solvinginstance import SolvingInstance

rnd_gen = np.random.default_rng(0)


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, limit_shortest_path: int = 1):
        self.problem_instance = problem_instance
        self.limit_shortest_path = limit_shortest_path

        self.nx_graph: nx.DiGraph = self.problem_instance.network_graph.nx_graph
        self.solver_name = "heuristic"

        self.solving_instance = SolvingInstance(self.problem_instance)

    def run(
        self,
    ):
        start_time = time.time()

        self.solving_instance.basic_placement_and_routing()

        self.solving_instance._maximize_quantities_greedy()

        self.runtime = time.time() - start_time
        self.score = self.solving_instance.get_score()

        self.x_variable, self.y_variable, self.q_variable = (
            self.solving_instance.get_arc_variables()
        )

        self._end_solving()
