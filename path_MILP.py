import pyomo.environ as pyo
import numpy as np
from itertools import islice
from networkx.readwrite.gml import read_gml
import networkx as nx
import pickle
import os

def services_to_features(services):
    return [demand['origin'] for service in services for demand in service],\
[demand['pmin'] for service in services for demand in service],\
[demand['pmax'] for service in services for demand in service],\
[demand['latency'] for service in services for demand in service],\
[sum(demand['resources'] for demand in service) for service in services]

def get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value

def path_MILP_solve(graph_name, services_name, solver, limit, options):
    G = read_gml(graph_name)
    G = nx.relabel.relabel_nodes(G, {node : int(node) for node in G.nodes()})
    with open(services_name, 'rb') as f:
        services = pickle.load(f)
    S, D, Sd = len(services), sum(len(elem) for elem in services), [len(elem) for elem in services]
    U, A, arc_dic, arc_list = len(G.nodes), len(G.edges), {elem : i for i, elem in enumerate(G.edges)}, list(G.edges)
    origins, Pdmin, Pdmax, latencies, resources = services_to_features(services)
    alpha, beta = [G.edges[edge]['alpha'] for edge in G.edges], [G.edges[edge]['beta'] for edge in G.edges]
    
    # ------------------------------
    # ---------- MODEL -------------
    # ------------------------------
    
    model = pyo.ConcreteModel()
    model.D = pyo.Set(initialize = range(D))
    model.U = pyo.Set(initialize = range(U))
    model.S = pyo.Set(initialize = range(S))
    model.A = pyo.Set(initialize = range(A))
    
    model.origins = pyo.Param(model.D, initialize = {d : origins[d] for d in range(D)}, mutable = False, within = pyo.NonNegativeIntegers)
    model.Pdmin = pyo.Param(model.D, initialize = {d : Pdmin[d] for d in range(D)}, mutable = False, within = pyo.NonNegativeIntegers)
    model.Pdmax = pyo.Param(model.D, initialize = {d : Pdmax[d] for d in range(D)}, mutable = False, within = pyo.NonNegativeIntegers)
    model.latencies = pyo.Param(model.D, initialize = {d : latencies[d] for d in range(D)}, mutable = False, within = pyo.NonNegativeReals)
    model.resources = pyo.Param(model.S, initialize = {s : resources[s] for s in range(S)}, mutable = False, within = pyo.NonNegativeIntegers)
    
    shortest_paths = [[elem for target in G.nodes for elem in islice(nx.algorithms.simple_paths.shortest_simple_paths(G, model.origins[d], target), limit)] for d in range(D)]
    all_shortest_paths = [elem for path_list in shortest_paths for elem in path_list]
    
    C, C_demand = sum(len(elem) for elem in shortest_paths), [len(elem) for elem in shortest_paths]
    model.C = pyo.Set(initialize = range(C))
    
    model.l = pyo.Var(model.C, within = pyo.Binary)
    model.y = pyo.Var(model.S, model.U, within = pyo.Binary)
    model.p = pyo.Var(model.D, within = pyo.NonNegativeReals)
    model.P = pyo.Var(model.C, within = pyo.NonNegativeReals)
    model.L = pyo.Var(model.C, model.A, within = pyo.NonNegativeReals)
    
    model.nul = pyo.Var(range(1), within = pyo.Binary)
    model.nul[0].fix(0)
    
    model.valeurs_p = pyo.Constraint(model.D, 
                                     rule = lambda model, d : pyo.inequality(model.Pdmin[d], model.p[d], model.Pdmax[d]))
    
    D_path_number = [i for i, path_list in enumerate(shortest_paths) for path in path_list]
    model.Pdmax_path = pyo.Param(model.C, 
                                 initialize = {c : Pdmax[D_path_number[c]] for c in range(C)}, 
                                 mutable = False, 
                                 within = pyo.NonNegativeIntegers)
    model.valeurs_P = pyo.Constraint(model.C, 
                                     rule = lambda model, c : model.P[c] <= model.Pdmax_path[c])
    
    model.latency_path = pyo.Param(model.C, 
                                   initialize = {c : latencies[D_path_number[c]] for c in range(C)}, 
                                   mutable = False, 
                                   within = pyo.NonNegativeReals)
    model.valeurs_L = pyo.Constraint(model.C, 
                                     model.A, rule = lambda model, c, a : model.L[c, a] <= model.latency_path[c])
    
    model.obj = pyo.Objective(sense = pyo.maximize, 
                              rule = lambda model : sum(model.p.values()))
    
    model.node_resources = pyo.Param(model.U, 
                                     initialize = {u : G.nodes[u]['resources'] for u in range(U)}, 
                                     mutable = False, 
                                     within = pyo.NonNegativeIntegers)
    model.resources_constraint = pyo.Constraint(model.U, 
                                                rule = lambda model, u : sum(model.resources[s]*model.y[s, u] for s in model.S) <= model.node_resources[u])
    
    model.alpha = pyo.Param(model.A, 
                                initialize = {edge : alpha[edge] for edge in model.A}, 
                                mutable = False, 
                                within = pyo.NonNegativeReals)
    
    model.beta = pyo.Param(model.A, 
                            initialize = {edge : beta[edge] for edge in model.A}, 
                            mutable = False, 
                            within = pyo.NonNegativeReals)
    
    model.delta = pyo.Param(model.C, 
                            model.A, 
                            initialize = {(c, a) : (arc_list[a] in [(all_shortest_paths[c][i], all_shortest_paths[c][i+1]) for i in range(len(all_shortest_paths[c])-1)]) for a in range(A) for c in range(C)}, 
                            mutable = False, 
                            within = pyo.Binary)
    model.arc_capacity = pyo.Param(model.A, 
                                   initialize = {a : G.edges[edge]['capacity'] for a, edge in enumerate(G.edges)}, 
                                   mutable = False, 
                                   within = pyo.NonNegativeIntegers)
    model.capacity = pyo.Constraint(model.A, 
                                    rule = lambda model, a : model.nul[0] + sum(model.delta[c, a] * model.P[c] for c in model.C) <= model.arc_capacity[a])
    
    model.all_shortest_paths_arc = pyo.Param(model.C, 
                                             initialize = {c : [arc_dic[(all_shortest_paths[c][i], all_shortest_paths[c][i+1])] for i in range(len(all_shortest_paths[c])-1)] for c in range(C)}, 
                                             mutable = False, 
                                             within = pyo.Any)
    model.path_demand = pyo.Param(range(D+1), 
                                  initialize = {d : sum(len(elem) for elem in shortest_paths[:d]) for d in range(D+1)}, 
                                  mutable = False, 
                                  within = pyo.NonNegativeIntegers)
    model.latency = pyo.Constraint(model.D, 
                                   rule = lambda model, d : sum(model.L[c, a] for c in range(model.path_demand[d], model.path_demand[d+1]) for a in model.all_shortest_paths_arc[c]) <= model.latencies[d])
    
    model.realization = pyo.Constraint(model.S, rule = lambda model, s : sum(model.y[s, u] for u in model.U) >= 1)
    
    model.serv_number = pyo.Param(model.D, 
                                  initialize = {sum(len(elem) for elem in services[:i])+j : i for i, service in enumerate(services) for j, task in enumerate(service)}, 
                                  mutable = False, 
                                  within = pyo.NonNegativeIntegers)
    model.ind_destination = pyo.Param(model.C, 
                                      model.U, 
                                      initialize = {(c, u) : all_shortest_paths[c][-1] == u for c in range(C) for u in range(U)})
    model.reach_server = pyo.Constraint(model.D, 
                                        model.U, 
                                        rule = lambda model, d, u : model.y[model.serv_number[d], u] <= sum(model.ind_destination[c, u]*model.l[c] for c in range(model.path_demand[d], model.path_demand[d+1])))
    
    D_path = [i for i, path_list in enumerate(shortest_paths) for path in path_list]
    model.D_path = pyo.Param(model.C, 
                             initialize = {c : D_path[c] for c in range(C)}, 
                             mutable = False, 
                             within = pyo.NonNegativeIntegers)
    m = max(Pdmax)
    model.flow_rule_1 = pyo.Constraint(model.C, 
                                       rule = lambda model, c : model.P[c] <= model.p[model.D_path[c]])
    model.flow_rule_2 = pyo.Constraint(model.C, 
                                       rule = lambda model, c : model.P[c] <= m*model.l[c])
    model.flow_rule_3 = pyo.Constraint(model.C, 
                                       rule = lambda model, c : model.P[c] >= model.p[model.D_path[c]]-m*(1-model.l[c]))
    
    n = max(latencies)
    model.latency_rule_1 = pyo.Constraint(model.C, 
                                          model.A, 
                                          rule = lambda model, c, a : model.L[c, a] <= model.alpha[a]*sum(model.delta[cp, a]*model.P[cp] for cp in model.C) + model.beta[a])
    model.latency_rule_2 = pyo.Constraint(model.C, 
                                          model.A, 
                                          rule = lambda model, c, a : model.L[c, a] <= n*model.l[c])
    model.latency_rule_3 = pyo.Constraint(model.C, 
                                          model.A, 
                                          rule = lambda model, c, a : model.L[c, a] >= model.alpha[a]*sum(model.delta[cp, a]*model.P[cp] for cp in model.C) + model.beta[a] - n*(1-model.l[c]))
    
    opt = pyo.SolverFactory(solver)
    for key in options.keys():
        opt.options[key] = options[key]
    resultat = opt.solve(model)
    res = ['Time: ', 'Lower bound: ', 'Upper bound: ', 'Termination condition: ']
    results = {}
    for elem in res:
        results[elem] = get_info_from_results(resultat, elem)
    return model, shortest_paths, results