### This code solve the linearly relaxed linear arc formulation of the problem using Pyomo

import pyomo.environ as pyo
import time
from networkx.readwrite.gml import read_gml
import networkx as nx
import pickle
import os

### Useful Functions

def services_to_features(services):
    return [demand['origin'] for service in services for demand in service],\
[demand['pmin'] for service in services for demand in service],\
[demand['pmax'] for service in services for demand in service],\
[demand['latency'] for service in services for demand in service],\
[sum(demand['resources'] for demand in service) for service in services]

def get_encoded_outgoing_arc(G, arc_dic, n):
    return [arc_dic[elem] for elem in get_outgoing_arc(G, n)]

def get_outgoing_arc(G, n):
    return [(n, elem) for elem in G.neighbors(n)]

def get_encoded_incoming_arc(G, arc_dic, n):
    return [arc_dic[elem] for elem in get_incoming_arc(G, n)]

def get_incoming_arc(G, n):
    return [(elem, n) for elem in G.neighbors(n)]

def get_info_from_results(results, info_string):
    i = str(results).lower().find(info_string.lower()) + len(info_string)
    value = ''
    while str(results)[i] != '\n':
        value = value + str(results)[i]
        i += 1
    return value
    
# The solve function

# graph_name = the Digraph with capacity, alpha, beta, resources ; 
# services_name = the services with origin, latency, pmin, pmax, resources ; 
# solver = str of the name of the solver ; 
# options = the options we want to pass in the solver

# return the pyomo model and a dict with some values that returns the linear solver

def arc_LP_solve(graph_name, services_name, solver, options, alpha = 1, beta = 1, warmstart = False):

    if warmstart:
        x, y, p, t, solved = heuristic_solve(graph_name, services_name, numb = 10, n_selection = 5, max_iter = 15, limit = 10)
        print(int(solved)*sum(p))

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
    model.S = pyo.Set(initialize = range(S))
    model.U = pyo.Set(initialize = range(U))
    model.A = pyo.Set(initialize = range(A))
    
    if warmstart:
        model.x = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals, 
                          bounds = (0, 1), 
                          initialize = {(d, a) : x[d, a] for d in model.D for a in model.A})

        model.y = pyo.Var(model.S, 
                          model.U, 
                          within = pyo.NonNegativeReals, 
                          bounds = (0, 1),
                          initialize = {(s, e) : y[s, e] for s in model.S for e in model.U})
        model.p = pyo.Var(model.D, 
                          within = pyo.NonNegativeReals, 
                          initialize = {d : p[d] for d in model.D})
        model.P = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals, 
                          initialize = {(d, a) : p[d]*x[d, a] for d in model.D for a in model.A})
        model.L = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals, 
                          initialize = {(d, a) : alpha[a]*sum(p[dp]*x[dp, a] for dp in model.D)+beta[a] for d in model.D for a in model.A})
    else:
        model.x = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals, 
                          bounds = (0, 1))

        model.y = pyo.Var(model.S, 
                          model.U, 
                          within = pyo.NonNegativeReals, 
                          bounds = (0, 1))
        model.p = pyo.Var(model.D, 
                          within = pyo.NonNegativeReals)
        model.P = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals)
        model.L = pyo.Var(model.D, 
                          model.A, 
                          within = pyo.NonNegativeReals)
    
    model.origins = pyo.Param(model.D, 
                              initialize = {d : origins[d] for d in range(D)}, 
                              mutable = False, 
                              within = pyo.NonNegativeIntegers)
    model.Pdmin = pyo.Param(model.D, 
                            initialize = {d : Pdmin[d] for d in range(D)}, 
                            mutable = False, 
                            within = pyo.NonNegativeIntegers)
    model.Pdmax = pyo.Param(model.D, 
                            initialize = {d : Pdmax[d] for d in range(D)}, 
                            mutable = False, 
                            within = pyo.NonNegativeIntegers)
    model.latencies = pyo.Param(model.D, 
                                initialize = {d : latencies[d] for d in range(D)}, 
                                mutable = False, 
                                within = pyo.NonNegativeReals)
    model.resources = pyo.Param(model.S, 
                                initialize = {s : resources[s] for s in range(S)}, 
                                mutable = False, 
                                within = pyo.NonNegativeIntegers)
    
    model.alpha = pyo.Param(model.A, 
                                initialize = {edge : alpha[edge] for edge in model.A}, 
                                mutable = False, 
                                within = pyo.NonNegativeReals)
    
    model.beta = pyo.Param(model.A, 
                            initialize = {edge : beta[edge] for edge in model.A}, 
                            mutable = False, 
                            within = pyo.NonNegativeReals)
    
    model.valeurs_p = pyo.Constraint(model.D, 
                                     rule = lambda model, d : pyo.inequality(model.Pdmin[d], model.p[d], model.Pdmax[d]))
    model.valeurs_P = pyo.Constraint(model.D, 
                                     model.A, 
                                     rule = lambda model, d, a : model.P[d, a] <= model.Pdmax[d])
    
    model.obj = pyo.Objective(sense = pyo.maximize, 
                              rule = lambda model : sum(model.p.values()))
    
    model.serv_number = pyo.Param(model.D, 
                                  initialize = {sum(len(elem) for elem in services[:i])+j : i for i, service in enumerate(services) for j, task in enumerate(service)}, 
                                  mutable = False, 
                                  within = pyo.NonNegativeIntegers)
    model.out_arcs = pyo.Param(model.U, 
                               initialize = {u : get_encoded_outgoing_arc(G, arc_dic, u) for u in range(U)}, 
                               mutable = False, 
                               within = pyo.Any)
    model.inc_arcs = pyo.Param(model.U, 
                               initialize = {u : get_encoded_incoming_arc(G, arc_dic, u) for u in range(U)}, 
                               mutable = False, 
                               within = pyo.Any)
    model.ind_u_origin = pyo.Param(model.U, 
                                   model.D, initialize = {(u, d) : int(u == model.origins[d]) for u in range(U) for d in range(D)}, 
                                   mutable = False, 
                                   within = pyo.Binary)
    model.conservation_constraint = pyo.Constraint(model.D, 
                                                   model.U, 
                                                   rule = lambda model, d, u : sum(model.x[d, arc] for arc in model.out_arcs[u])-sum(model.x[d, arc] for arc in model.inc_arcs[u]) == model.ind_u_origin[u, d] - model.y[model.serv_number[d], u])
    
    model.node_resources = pyo.Param(model.U, initialize = {u : G.nodes[u]['resources'] for u in range(U)}, mutable = False, within = pyo.NonNegativeIntegers)
    model.resources_constraint = pyo.Constraint(model.U, rule = lambda model, u : sum([model.y[s, u]*model.resources[s] for s in model.S]) <= model.node_resources[u])
    
    model.arc_capacity = pyo.Param(model.A, initialize = {a : G.edges[edge]['capacity'] for a, edge in enumerate(G.edges)}, mutable = False, within = pyo.NonNegativeIntegers)
    model.capacity_constraint = pyo.Constraint(model.A, rule = lambda model, a : sum([model.P[d, a] for d in model.D]) <= model.arc_capacity[a])
    
    
    model.latency_arc_1 = pyo.Constraint(model.D, 
                                         rule = lambda model, d : sum(model.L[d, ap] for ap in model.A) <= model.latencies[d])
    model.latency_arc_2 = pyo.Constraint(model.D, 
                                         model.A, 
                                         rule = lambda model, d, a : model.L[d, a] <= model.alpha[a]*sum(model.P[dp, a] for dp in model.D)+model.beta[a])
    
    model.N = pyo.Param(initialize = max(latencies), 
                        mutable = False, 
                        within = pyo.NonNegativeReals)
    model.latency_arc_3 = pyo.Constraint(model.D, 
                                         model.A, 
                                         rule = lambda model, d, a : model.L[d, a] >= model.alpha[a]*sum(model.P[dp, a] for dp in model.D)+model.beta[a]-model.N*(1-model.x[d, a]))
    model.latency_arc_4 = pyo.Constraint(model.D, 
                                         model.A, 
                                         rule = lambda model, d, a : model.L[d, a] <= model.N*model.x[d, a])
    
    model.power_1 = pyo.Constraint(model.D, model.A, rule = lambda model, d, a : model.P[d, a] <= model.p[d])

    model.M = pyo.Param(initialize = max(Pdmax), 
                        mutable = False, 
                        within = pyo.NonNegativeIntegers)
    model.power_2 = pyo.Constraint(model.D, 
                                   model.A, 
                                   rule = lambda model, d, a : model.P[d, a] >= model.p[d]-model.M*(1-model.x[d, a]))
    model.power_3 = pyo.Constraint(model.D, 
                                   model.A, 
                                   rule = lambda model, d, a : model.P[d, a] <= model.M*model.x[d, a])
    
    opt = pyo.SolverFactory(solver)
    for key in options.keys():
        opt.options[key] = options[key]
    if warmstart:
        resultat = opt.solve(model, warmstart=True)
    else:
        resultat = opt.solve(model)
    res = ['Time: ', 'Lower bound: ', 'Upper bound: ', 'Termination condition: ']
    results = {}
    for elem in res:
        results[elem] = get_info_from_results(resultat, elem)
    return model, results