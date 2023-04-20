import os
import numpy as np

from utils.util import base_graphs_directory, graphs_with_services_directory, base_graphs_files_list, base_graphs_names_list, service_folder_name
from utils.util import make_graph_with_services_directory, pickle_save
from utils.util import GraphWithServices, SingleTask, Services
import networkx as nx
from networkx.readwrite.gml import read_gml, write_gml

assert base_graphs_directory in os.listdir(), 'No folder with base graphs.'
assert len(os.listdir(base_graphs_directory)), 'No graph in the base graphs folder.'
make_graph_with_services_directory()
assert graphs_with_services_directory in os.listdir(), 'Making graph with services directory failed.'

node_resources = 1e4
edge_capacity_function = lambda graph_node : 200+200*graph_node//15
latency_alpha_factor = .2
latency_beta_factor = 5

required_service_latency_function = lambda edge_number, number_of_tasks : 7.*(latency_alpha_factor*5.*number_of_tasks**2/edge_number+latency_beta_factor)
required_resources_function = lambda number_of_tasks : 5000//number_of_tasks
flow_min = 1
flow_max = 100

for graph_number in range(len(base_graphs_names_list)):
    graph_name = base_graphs_names_list[graph_number]
    graph_file_name = base_graphs_files_list[graph_number]
    graph_file_path = os.path.join(base_graphs_directory, graph_file_name)
    
    nx_graph = read_gml(graph_file_path)

    edge_capacity = edge_capacity_function(len(nx_graph.nodes))

    graph_with_services = GraphWithServices(base_graph=nx_graph,
                                            node_resources=node_resources,
                                            edge_capacity=edge_capacity,
                                            latency_alpha_factor=latency_alpha_factor,
                                            latency_beta_factor=latency_beta_factor)

    current_processed_graph_directory = os.path.join(graphs_with_services_directory, graph_name)
    try:
        os.mkdir(current_processed_graph_directory)
    except FileExistsError:
        pass

    current_processed_graph_file_path = os.path.join(current_processed_graph_directory, graph_file_name)
    write_gml(graph_with_services.nx_graph, current_processed_graph_file_path)

    services_directory = os.path.join(current_processed_graph_directory, service_folder_name)

    try:
        os.mkdir(services_directory)
    except FileExistsError:
        pass

    base_services_name = graph_name+'_'

    

    for number_of_tasks in range(1, 11):
        tasks = []

        required_max_latency = required_service_latency_function(len(graph_with_services.nx_graph.edges), number_of_tasks)
        required_resources = required_resources_function(number_of_tasks)

        for _ in range(number_of_tasks):
            task = SingleTask(nx_graph=graph_with_services.nx_graph,
                               origin_node_index=np.random.choice(graph_with_services.nx_graph.nodes),
                               flow_min=flow_min,
                               flow_max=flow_max,
                               max_latency=required_max_latency,
                               resources_needed=required_resources)
            tasks.append(task)
        
        services_problem = Services(task_list=tasks,
                                    nx_graph=graph_with_services.nx_graph)
        services_problem_name = base_services_name+str(number_of_tasks)
        services_problem_file_name = services_problem_name+'.pkl'

        services_problem_file_directory = os.path.join(services_directory, services_problem_file_name)

        pickle_save(services_problem,
                    services_problem_file_directory)
