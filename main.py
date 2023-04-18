from os import listdir, mkdir
from os.path import join

import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

original_graph_path = join('data', 'base_graphs')

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

def network_graph_design(graph_name, 
                         nx_graph):
    try:
        mkdir(join('data', 'graphs_with_services'))
    except FileExistsError:
        pass

    mkdir(join('data', 'graphs_with_services', graph_name))
    nx_graph = nx.DiGraph(nx_graph)
    nx_graph = nx.relabel.relabel_nodes(nx_graph, {node : i for i, node in enumerate(nx_graph.nodes())})
    nx_graph = make_graph_symmetric(nx_graph)
