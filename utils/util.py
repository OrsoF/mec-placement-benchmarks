from os import listdir, mkdir
from os.path import join
import numpy as np
import pickle
from typing import Callable

import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

data_folder = 'data'
original_graph_folder = join(data_folder, 'base_graphs')
base_graphs_name = [elem[:-4] for elem in listdir(original_graph_folder)]
base_graphs = [read_gml(join(original_graph_folder, base_graph_name+'.gml')) for base_graph_name in base_graphs_name]
processed_graph_folder = join(data_folder, 'graphs_with_services')
service_folder_name = 'services'

base_graphs_directory = 'base_graphs'
graphs_with_services_directory = 'graphs_with_services'

base_graphs_files_list = listdir(base_graphs_directory)
base_graphs_names_list = [graph_file[:-4] for graph_file in base_graphs_files_list]

def make_graph_with_services_directory():
    try:
        mkdir(graphs_with_services_directory)
    except FileExistsError:
        pass

# try:
#     mkdir(processed_graph_folder)
# except FileExistsError:
#     pass

def get_base_graphs_gml(original_graph_path):
    original_graph_list = listdir(original_graph_path)
    return [read_gml(elem) for elem in original_graph_list]

def get_base_graph_names(original_graph_path):
    original_graph_list = listdir(original_graph_path)
    return [elem[:-4] for elem in original_graph_list]




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

class GraphWithServices:
    def __init__(self, 
                 base_graph : nx.Graph,
                 node_resources : float,
                 edge_capacity : float,
                 latency_alpha_factor : float,
                 latency_beta_factor : float):

        self.nx_graph = nx.DiGraph(base_graph)
        self.nx_graph = nx.relabel.relabel_nodes(self.nx_graph, {node : i for i, node in enumerate(self.nx_graph.nodes())})
        self.make_graph_symmetric()
        self.remove_all_attributes()

        self.set_resources(node_resources)
        self.set_capacities(edge_capacity)
        self.set_latency_alpha_factor(latency_alpha_factor)
        self.set_latency_beta_factor(latency_beta_factor)

    def set_resources(self, 
                      node_resources : float):
        resources_map = {node : node_resources for node in self.nx_graph.nodes}
        nx.classes.function.set_node_attributes(self.nx_graph, 
                                                resources_map, 
                                                'resources')
        
    def set_capacities(self,
                       edge_capacity : float):
        capacity_map = {edge : {'capacity' : edge_capacity} for edge in self.nx_graph.edges}
        nx.classes.function.set_edge_attributes(self.nx_graph, capacity_map)

    def set_latency_alpha_factor(self,
                                 latency_alpha_factor):
        latency_alpha_map = {edge : {'alpha' : latency_alpha_factor} for edge in self.nx_graph.edges}
        nx.classes.function.set_edge_attributes(self.nx_graph, latency_alpha_map)

    def set_latency_beta_factor(self,
                                 latency_beta_factor):
        latency_beta_map = {edge : {'alpha' : latency_beta_factor} for edge in self.nx_graph.edges}
        nx.classes.function.set_edge_attributes(self.nx_graph, latency_beta_map)

    def make_graph_symmetric(self):
        for arc in self.nx_graph.edges:
            self.nx_graph.add_edge(arc[1], arc[0])

    def remove_all_attributes(self):
        for node in self.nx_graph.nodes:
            keys = list(self.nx_graph.nodes[node].keys())
            for key in keys:
                del self.nx_graph.nodes[node][key]
        for edge in self.nx_graph.edges:
            keys = list(self.nx_graph.edges[edge].keys())
            for key in keys:
                del self.nx_graph.edges[edge][key]
        return self.nx_graph

def process_base_graph(nx_graph : nx.Graph,
                       resource_by_node : float,
                       capacity_function : Callable,
                       alpha_constant : float,
                       beta_constant : float):
    
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

def create_processed_graph_folder(graph_name):
    try:
        mkdir(join('graphs_with_services', graph_name))
    except FileExistsError:
        pass

def network_graph_design(graph_name, 
                         nx_graph,
                         resource_by_node=1e4,
                         capacity_function=lambda graph_node : 200+200*graph_node//15,
                         alpha_constant=.2,
                         beta_constant=5):
    
    create_processed_graph_folder(graph_name)

    nx_graph = process_base_graph(nx_graph,
                                  resource_by_node,
                                  capacity_function,
                                  alpha_constant,
                                  beta_constant)

    write_gml(nx_graph, join('data', 'graphs_with_services', graph_name, 'graph_'+graph_name+'.gml'))

    alpha_moyen = np.mean([nx_graph.edges[edge]['alpha'] for edge in nx_graph.edges])
    beta_moyen = np.mean([nx_graph.edges[edge]['beta'] for edge in nx_graph.edges])

    for i in range(1, 11):
        # We generate i services
        services_i = [generate_service(nx_graph, 
                                       number_of_services = i, 
                                       latency = 7.*(alpha_moyen*5.*i**2/len(nx_graph.edges)+beta_moyen), 
                                       resources = 5000//i, 
                                       flow = 100) for _ in range(i)]
        
        # We save each service instance
        if i < 10:
            file_name = join(processed_graph_folder, graph_name, 'servi_'+graph_name+'_0'+str(i)+'.pkl')
        else:
            file_name = join(processed_graph_folder, graph_name, 'servi_'+graph_name+'_'+str(i)+'.pkl')
        f = open(file_name, "w+")
        f.close()
        f = open(file_name, 'wb')
        pickle.dump(services_i, f, protocol=2)
        f.close()

class SingleTask:
    def __init__(self,
                 nx_graph,
                 origin_node_index,
                 flow_min,
                 flow_max,
                 max_latency,
                 resources_needed):
        self.origin_node_index = origin_node_index
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.max_latency = max_latency
        self.resources_needed = resources_needed

class Services:
    def __init__(self, 
                 task_list, 
                 nx_graph):
        
        self.nx_graph = nx_graph

        self.task_number = 0
        self.task_list = []

        self.origin_nodes = []
        self.min_flows = []
        self.max_flows = []
        self.max_latencies = []
        self.needed_resources = []

        for task in task_list:
            self.add_task(task)

    def add_task(self, 
                 task_element):
        self.current_task_element = task_element
        self.task_list.append(self.current_task_element)
        self.task_number += 1

        self.origin_nodes.append(self.current_task_element.origin_node_index)
        self.min_flows.append(self.current_task_element.flow_min)
        self.max_flows.append(self.current_task_element.flow_max)
        self.max_latencies.append(self.current_task_element.max_latency)
        self.needed_resources.append(self.current_task_element.resources_needed)
