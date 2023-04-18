import numpy as np
import networkx as nx
from networkx.readwrite.gml import read_gml
import pickle
import time
import os
from itertools import islice

def serv_number(services, d):
    services_len = np.array([len(elem) for elem in services])
    task_number = np.array([sum(services_len[:s]) <= d < sum(services_len[:s+1]) for s in range(len(services)-1)]+[1])
    s = np.argmax(task_number)
    return s, d - sum(services_len[:s])

def capacity_check(G, path, quantity):
    if len(path) == 1:
        return True
    else:
        return min(G.edges[(path[i], path[i+1])]['remaining_capacity'] >= quantity for i in range(len(path)-1))

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

def services_nodes_association(G, resources, y, Sd, weight_nodes):
    available_resources = [G.nodes[u]['resources'] for u in G.nodes]
    needed_resources = np.copy(resources)
    S, U = len(needed_resources), len(available_resources)
    y = np.zeros((S, U))
    possible_nodes = [elem/weight_nodes[i] for i, elem in enumerate(available_resources)]
    chosen_nodes = [-1]*S
#     for i in range(S):
#         s = np.argmax(needed_resources)
#         resources = needed_resources[s]
#         needed_resources[s] = 0
#         node = np.argmax(possible_nodes)
#         y[s, node] = 1
#         chosen_nodes[s] = node
#         available_resources[node] -= resources
#         possible_nodes[node] = available_resources[node]/weight_nodes[node]
    
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

def shortest_paths_choice(G, services, origins, chosen_nodes, Pdmin, latencies, alpha, beta, arc_dic, x, y, p, weight_edges, limit):
    for i, edge in enumerate(G.edges):
        G.edges[edge]['weight'] = weight_edges[i]

    S, D, A, U = len(services), sum(len(service) for service in services), len(G.edges), len(G.nodes)
    service_number = np.array([serv_number(services, d) for d in range(D)])
    
    used_paths = [[]]*D
    for d in range(D):
        s, d0 = service_number[d]
        origin, destination = origins[d], chosen_nodes[s]
        shortest_paths = islice(nx.algorithms.simple_paths.shortest_simple_paths(G, origin, destination, weight = 'weight'), limit)
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

def heuristic(G, services, weight_nodes, weight_edges, origins, Pdmin, Pdmax, resources, latencies, S, Sd, D, A, U, arc_list, arc_dic, alpha, beta, limit):
    
    x, y, p = np.zeros((D, A)), np.zeros((S, U)), np.zeros(D)
    
    for arc in G.edges:
        G.edges[arc]['remaining_capacity'] = G.edges[arc]['capacity']
        G.edges[arc]['latency'] = 0
        
    p = Pdmin.copy()
    
    # Association noeuds/services
    
    y, chosen_nodes = services_nodes_association(G, resources, y, Sd, weight_nodes)
                
    # For a given demand, we list the paths (shortest to longest) and take the first possible
    
    G, x, used_paths, solved = shortest_paths_choice(G, services, origins, chosen_nodes, Pdmin, latencies, alpha, beta, arc_dic, x, y, p, weight_edges, limit)
    if not solved:
        return x, y, p, False
    
    # We now increase the quantity of data
    
    else:
        G, p = increase_quantities(G, used_paths, weight_edges, Pdmin, Pdmax, latencies, alpha, beta, x, y, p, arc_dic)
        return x, y, p, True

def heuristic_ES_solve(graph_name, services_name, numb = 30, n_selection = 25, max_iter = 60, limit = 1000):
    G = read_gml(graph_name)
    G = nx.relabel.relabel_nodes(G, {node : int(node) for node in G.nodes()})

    with open(services_name, 'rb') as f:
        services = pickle.load(f)
        
    origins, Pdmin, Pdmax, resources, latencies = services_to_features(services)
    S, Sd, D = len(services), np.array([len(elem) for elem in services]), sum(len(elem) for elem in services)
    A, U = len(G.edges), len(G.nodes)
    arc_list, arc_dic = np.array(G.edges), {edge : a for a, edge in enumerate(G.edges)}
    alpha, beta = [G.edges[edge]['alpha'] for edge in G.edges], [G.edges[edge]['beta'] for edge in G.edges]
    l = U+A

    t = time.time()

    def f(vec):
        vec[vec < 1] = 1
        x, y, p, solved = heuristic(G, services, vec[:U], vec[U:], origins, Pdmin, Pdmax, resources, latencies, S, Sd, D, A, U, arc_list, arc_dic, alpha, beta, limit)
        return int(solved)*sum(p)
    
    obj = f(np.ones(l))

    def croiser(A, B):
        return np.array([A[i] if np.random.random() < 0.5 else B[i] for i in range(l)]).reshape((1, l))

    def mutation(A, effet, distance):
        for i in range(effet):
            ind = np.random.randint(l)
            A[ind] = A[ind] + np.random.randint(low = -distance, high = distance+1)
        return A.reshape((1, l))

    def evaluation(pop):
        return [f(elem) for elem in pop]

    def selection(pop, objectives, nombre):
        pop = sorted([(pop[i], objectives[i]) for i in range(len(pop))], key = lambda elem : elem[1], reverse=True)
        return np.array([elem[0] for elem in pop[:nombre]])

    def evolve(pop, n_croisement, n_mutation, effet, distance, n_selection, introduction):
        best = pop[0]

        pop = np.concatenate((pop, np.random.randint(low = 1, high = 3, size = (introduction, l))))

        for i in range(n_croisement):
            a, b = np.random.randint(len(pop)), np.random.randint(len(pop))
            pop = np.concatenate((pop, croiser(pop[a], pop[b])))

        for i in range(n_mutation):
            pop = np.concatenate((pop, mutation(pop[np.random.randint(len(pop))], effet, distance)))

        objectives = evaluation(pop)

        pop = selection(pop, objectives, n_selection)

        return np.concatenate((pop, best.reshape(1, l)))

    pop = np.random.randint(low = 1, high = 3, size = (numb, l))
    score = f(pop[0])
    best = pop[0][:]
    results = np.zeros((max_iter, 4))
    best = [max(1, elem) for elem in best]

    for i in range(max_iter):
        pop = evolve(pop, n_croisement = 2, n_mutation = 2, effet = 5, distance = 1, n_selection = n_selection, introduction = 2)
        meilleur = pop[0]
        if f(meilleur) > score:
            best = meilleur[:]
            score = f(best)
    x, y, p, solved = heuristic(G, services, best[:U], best[U:], origins, Pdmin, Pdmax, resources, latencies, S, Sd, D, A, U, arc_list, arc_dic, limit = 1000, alpha = alpha, beta = beta)
    
    return x, y, p, time.time()-t, solved