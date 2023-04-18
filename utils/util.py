from os import listdir, mkdir
from os.path import join
import numpy as np
import pickle

import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

data_folder = 'data'
original_graph_folder = join(data_folder, 'base_graphs')
base_graphs_name = [elem[:-4] for elem in listdir(original_graph_folder)]
base_graphs = [read_gml(join(original_graph_folder, base_graph_name+'.gml')) for base_graph_name in base_graphs_name]
processed_graph_folder = join(data_folder, 'graphs_with_services')

try:
    mkdir(processed_graph_folder)
except FileExistsError:
    pass

def get_graphs_gml(original_graph_path):
    original_graph_list = listdir(original_graph_path)
    return [read_gml(elem) for elem in original_graph_list]

def get_graph_names(original_graph_path):
    original_graph_list = listdir(original_graph_path)
    return [elem[:-4] for elem in original_graph_list]

def make_graph_symmetric(nx_graph):
    for arc in nx_graph.edges:
        a, b = arc
        nx_graph.add_edge(b, a)
    return nx_graph

def remove_all_attributes(nx_graph):
    for node in nx_graph.nodes:
        keys = list(nx_graph.nodes[node].keys())
        for key in keys:
            del nx_graph.nodes[node][key]
    for edge in nx_graph.edges:
        keys = list(nx_graph.edges[edge].keys())
        for key in keys:
            del nx_graph.edges[edge][key]
    return nx_graph

def generate_service(nx_graph, 
                     number_of_services, 
                     latency, 
                     resources, 
                     flow):
    service = []
    for _ in range(number_of_services):
        dic = {}
        dic['origin'] = np.random.choice(nx_graph.nodes)
        dic['pmin'] = 1
        dic['pmax'] = flow
        dic['resources'] = resources
        dic['latency'] = latency
        service.append(dic)
    return service

def process_base_graph(nx_graph,
                       resource_by_node,
                       capacity_function,
                       alpha_constant,
                       beta_constant):
    nx_graph = nx.DiGraph(nx_graph)
    nx_graph = nx.relabel.relabel_nodes(nx_graph, {node : i for i, node in enumerate(nx_graph.nodes())})
    nx_graph = make_graph_symmetric(nx_graph)
    nx_graph = remove_all_attributes(nx_graph)

    # 10**6 resources, 1000 capacity, 1/5 alpha, 5 beta

    nx.classes.function.set_node_attributes(nx_graph, {node : resource_by_node for node in nx_graph.nodes}, 'resources')
    nx.classes.function.set_edge_attributes(nx_graph, {edge : {'capacity' : capacity_function(len(nx_graph.nodes)), 
                                                               'alpha' : alpha_constant, 
                                                               'beta' : beta_constant} for edge in nx_graph.edges})

    return nx_graph

def network_graph_design(graph_name, 
                         nx_graph,
                         resource_by_node=1e4,
                         capacity_function=lambda graph_node : 200+200*graph_node//15,
                         alpha_constant=.2,
                         beta_constant=5):
    
    try:
        mkdir(join('data', 'graphs_with_services', graph_name))
    except FileExistsError:
        pass

    nx_graph = process_base_graph(nx_graph,
                                  resource_by_node,
                                  capacity_function,
                                  alpha_constant,
                                  beta_constant)

    write_gml(nx_graph, join('data', 'graphs_with_services', graph_name, graph_name+'.gml'))

    alpha_moyen = sum(nx_graph.edges[edge]['alpha'] for edge in nx_graph.edges)/len(nx_graph.edges)
    beta_moyen = sum(nx_graph.edges[edge]['beta'] for edge in nx_graph.edges)/len(nx_graph.edges)

    for i in range(1, 11):
        services_i = [generate_service(nx_graph, 
                                       number_of_services = i, 
                                       latency = 7.*(alpha_moyen*5.*i**2/len(nx_graph.edges)+beta_moyen), 
                                       resources = 5000//i, 
                                       flow = 100) for _ in range(i)]
        if i < 10:
            file_name = join(processed_graph_folder, graph_name, graph_name+'_0'+str(i)+'.pkl')
        else:
            file_name = join(processed_graph_folder, graph_name, graph_name+'_'+str(i)+'.pkl')
        f = open(file_name, "w+")
        f.close()
        f = open(file_name, 'wb')
        pickle.dump(services_i, f, protocol=2)
        f.close()
