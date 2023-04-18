import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from itertools import islice

# If we concatenate the demands of the different services, 
# from a given demand d we can retrieve the service s and 
# the demand d0 of this service using the serv_number function
def serv_number(services, d):
    services_len = np.array([len(elem) for elem in services])
    task_number = np.array([sum(services_len[:s]) <= d < sum(services_len[:s+1]) for s in range(len(services)-1)]+[1])
    s = np.argmax(task_number)
    return s, d - sum(services_len[:s])

# For a given path and an additional quantity, we test if 
# the remaining_capacity of crossed edges allows it
def capacity_check(G, path, quantity):
    if len(path) == 1:
        return True
    else:
        return min([G.edges[(path[i], path[i+1])]['remaining_capacity'] >= quantity for i in range(len(path)-1)])

# For a given path and an additional quantity, 
# we test if the latency of the demand is not exceeded
def latency_check(G, path, latency, quantity, alpha, beta, arc_dic):
    if len(path) == 1:
        return True
    else:
        used_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        current_latency = sum(G.edges[arc]['latency'] for arc in used_edges)
        additional_latency = sum(alpha[arc_dic[edge]]*quantity + beta[arc_dic[edge]]*int(G.edges[edge]['latency'] == 0) for edge in used_edges)
        return current_latency + additional_latency <= latency

def services_to_features(services):
    origins, Pdmin, Pdmax, resources, latencies = [], [], [], [], []
    for service in services:
        resources.append(sum(service[i]['resources'] for i in range(len(service))))
        for demande in service:
            Pdmin.append(demande['pmin'])
            Pdmax.append(demande['pmax'])
            latencies.append(demande['latency'])
            origins.append(demande['origin'])
    return np.array(origins), np.array(Pdmin), np.array(Pdmax), np.array(resources), np.array(latencies)

# Hypothesis : available_resources can support all services
# The principle here is simple :
# In the case where the weights are 1 :
# We choose the service that use the most resources
# We choose the node that have the most available resources
# We associate the two
# We reduce available_resources by the amount of usedresources
# In the case where available_resources is negative, it will be not chosen
# If a weight are more than 1: the choice of the used node is driven by resources/weight_node
# (we reduce the priority of some nodes to ensure that a maximum number of nodes are used)
def services_nodes_association(G, resources, y, Sd, weight_nodes):
    available_resources = [G.nodes[node]['resources'] for node in G.nodes]
    needed_resources = np.copy(resources)
    S, U = len(resources), len(G.nodes)
    y = np.zeros((S, U))
    possible_nodes = [res/weight_nodes[i] for i, res in enumerate(available_resources)]
    chosen_nodes = [-1]*S
    while max(needed_resources) > 0:
        s = np.argmax(needed_resources)
        resources = needed_resources[s]
        node = np.argmax(possible_nodes)
        if available_resources[node] - resources >= 0:
            available_resources[node] -= resources
            possible_nodes[node] = available_resources[node]/weight_nodes[node]
            needed_resources[s] = 0
            
            y[s, node] = 1
            chosen_nodes[s] = node
        else:
            weight_nodes[node] += 1
            possible_nodes = [res/weight_nodes[i] for i, res in enumerate(available_resources)]
    return y, chosen_nodes

# The weight here are used to compute the sortest paths and prioritize some arcs in the choice of the sortest paths
def shortest_paths_choice(G, services, origins, chosen_nodes, Pdmin, latencies, alpha, beta, arc_dic, x, y, p, weight_edges, limit_shortest_paths):
    for i, edge in enumerate(G.edges):
        G.edges[edge]['weight'] = weight_edges[i]
        
    S, D, A, U = len(services), sum(len(service) for service in services), len(G.edges), len(G.nodes)
    service_number = np.array([serv_number(services, d) for d in range(D)])
    
    used_paths = [[]]*D
    for d in range(D):
        s, d0 = service_number[d]
        origin, destination = origins[d], chosen_nodes[s]
        shortest_paths = islice(nx.algorithms.simple_paths.shortest_simple_paths(G, origin, destination, weight = 'weight'), limit_shortest_paths)
        quantity, latency = Pdmin[d], latencies[d]
        for i, path in enumerate(shortest_paths):
            if capacity_check(G, path, quantity) and latency_check(G, path, latency, quantity, alpha, beta, arc_dic):
                used_paths[d] = path
                for i in range(len(path)-1):
                    arc = (path[i], path[i+1])
                    x[d, arc_dic[arc]] = 1
                    G.edges[arc]['remaining_capacity'] -= quantity
                    G.edges[arc]['latency'] += alpha[arc_dic[arc]]*quantity + beta[arc_dic[arc]]
                flag = True
                break
            else:
                flag = False
        if not flag:
            return G, x, used_paths, False
    return G, x, used_paths, True
    
def increase_quantities(G, used_paths, weight_edges, Pdmin, Pdmax, latencies, alpha, beta, x, y, p, arc_dic):
    paths_len = [sum(weight_edges[arc_dic[(path[i], path[i+1])]] for i in range(len(path)-1)) for path in used_paths]
    considered_demand = np.argmin(paths_len)
    quantity, path, latency = Pdmin[considered_demand], used_paths[considered_demand], latencies[considered_demand]
    while True:
        if capacity_check(G, path, quantity+1) and latency_check(G, path, latency, quantity+1, alpha, beta, arc_dic) and quantity + 1 <= Pdmax[considered_demand]:
            p[considered_demand] += 1
            quantity += 1
            path = used_paths[considered_demand]
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                G.edges[edge]['latency'] += alpha[arc_dic[edge]]
                G.edges[edge]['remaining_capacity'] -= 1
        else:
            paths_len[considered_demand] = np.inf
            considered_demand = np.argmin(paths_len)
            quantity, path, latency = Pdmin[considered_demand], used_paths[considered_demand], latencies[considered_demand]
        if np.min(paths_len) == np.inf:
            break
    return G, p
    
# graph_name = graph path, services_name = service path, weight_nodes = a weight vector of len len(G.nodes), weight_edges = a weight vector of len len(G.edges), limit_shortest_paths = the number of considered shortest paths

# return x, y, p variables, the computing time, and a bool equal to True if the problem is solved, False if not

def heuristic_solve(graph_name, services_name, weight_nodes, weight_edges, limit_shortest_paths):
    # Loading the graph and the services
    G = read_gml(graph_name)
    G = nx.relabel.relabel_nodes(G, {node : int(node) for node in G.nodes()})
    with open(services_name, 'rb') as f:
        services = pickle.load(f)
    
    print('pmin', sum(demand['pmin'] for service in services for demand in service))
    print('pmax', sum(demand['pmax'] for service in services for demand in service))
    
    origins, Pdmin, Pdmax, resources, latencies = services_to_features(services)
    alpha, beta = [G.edges[edge]['alpha'] for edge in G.edges], [G.edges[edge]['beta'] for edge in G.edges]
    
    S, Sd, D = len(services), np.array([len(elem) for elem in services]), sum(len(elem) for elem in services)
    A, U = len(G.edges), len(G.nodes)
    arc_list, arc_dic = np.array(G.edges), {edge : a for a, edge in enumerate(G.edges)}
    
    x, y, p = np.zeros((D, A)), np.zeros((S, U)), np.zeros(D)
    
    for arc in G.edges:
        G.edges[arc]['remaining_capacity'] = G.edges[arc]['capacity']
        G.edges[arc]['latency'] = 0
        
    p = Pdmin.copy()
    
    t = time.time()
    
    # Association noeuds/services
    
    y, chosen_nodes = services_nodes_association(G, resources, y, Sd, weight_nodes)
                
    # For a given demand, we list the paths (shortest to longest) and take the first possible
    
    G, x, used_paths, solved = shortest_paths_choice(G, services, origins, chosen_nodes, Pdmin, latencies, alpha, beta, arc_dic, x, y, p, weight_edges, limit_shortest_paths)
    if not solved:
        return x, y, p, time.time()-t, False
    
    # We now increase the quantity of data
    
    else:
        G, p = increase_quantities(G, used_paths, weight_edges, Pdmin, Pdmax, latencies, alpha, beta, x, y, p, arc_dic)
        return x, y, p, time.time()-t, True