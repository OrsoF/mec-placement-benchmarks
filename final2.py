from graph_maker import graph_design
import os

graphs = ['atlanta', 'france', 'germany']
root = os.getcwd()

graph_design(graphs, root)

from heuristic import heuristic_solve
import numpy as np
from networkx.readwrite.gml import read_gml
graph_path, service_path = 'data/france/france.gml', 'data/france/f_03.pkl'
i = 3

G = read_gml(graph_path)
x, y, p, t, solved = heuristic_solve(graph_name = graph_path, 
                                     services_name = service_path, 
                                     weight_nodes = np.ones(len(G.nodes)), 
                                     weight_edges = np.ones(len(G.edges)), 
                                     limit_shortest_paths = 1000)
print('Heuristic objective : ', int(solved)*sum(p))

from heuristic_weighted import heuristic_weighted_solve

x, y, p, t, solved = heuristic_weighted_solve(graph_name = graph_path, 
                         services_name = service_path, 
                         loop_1 = 3, 
                         loop_2 = 3, 
                         limit = 10)

print('Weighted heuristic objective : ', int(solved)*sum(p))

from heuristic_ES import heuristic_ES_solve

x, y, p, t, solved = heuristic_ES_solve(graph_name = graph_path, 
                                        services_name = service_path, 
                                        numb = 35, 
                                        n_selection = 30, 
                                        max_iter = 40)

print('ES heuristic objective : ', int(solved)*sum(p))

from path_ILP import path_ILP_solve

model, shortest_paths, results = path_ILP_solve(graph_name = graph_path, 
                                                services_name = service_path, 
                                                solver = 'cplex', 
                                                limit = 1, 
                                                options = {'threads' : 1, 'timelimit' : 3600})

print('path_ILP')
print(results)

from path_MILP import path_MILP_solve

model, shortest_paths, results = path_MILP_solve(graph_name = graph_path, 
                                                services_name = service_path, 
                                                solver = 'cplex', 
                                                limit = 1, 
                                                options = {'threads' : 1, 'timelimit' : 3600})

print('path_MILP')
print(results)

from path_LP import path_LP_solve

model, shortest_paths, results = path_LP_solve(graph_name = graph_path, 
                                                services_name = service_path, 
                                                solver = 'cplex', 
                                                limit = 1, 
                                                options = {'threads' : 1, 'timelimit' : 3600})

print('path_LP')
print(results)

from arc_ILP import arc_ILP_solve

model, results = arc_ILP_solve(graph_name = graph_path, 
                  services_name = service_path, 
                  solver = 'cplex', 
                  options = {'threads' : 1, 'timelimit' : 3600})
                    
print('arc_ILP')
print(results)

from arc_MILP import arc_MILP_solve

model, results = arc_MILP_solve(graph_name = graph_path, 
                  services_name = service_path, 
                  solver = 'cplex', 
                  options = {'threads' : 1, 'timelimit' : 3600})

print('arc_MILP')
print(results)

from arc_LP import arc_LP_solve

model, results = arc_LP_solve(graph_name = graph_path, 
                  services_name = service_path, 
                  solver = 'cplex', 
                  options = {'threads' : 1, 'timelimit' : 3600})
print('arc_LP')
print(results)