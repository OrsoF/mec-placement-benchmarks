from graph_maker import graph_design
import os

graphs = ['atlanta', 'france', 'germany']
root = os.getcwd()

graph_design(graphs, root)

from heuristic import heuristic_solve
import numpy as np
from networkx.readwrite.gml import read_gml
graph_path, service_path = 'data/germany/germany.gml', 'data/germany/g_03.pkl'
i = 3

G = read_gml(graph_path)

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