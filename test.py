import os

base_graphs_directory = 'base_graphs'
graphs_with_services_directory = 'graphs_with_services'

assert base_graphs_directory in os.listdir(), 'No folder with base graphs.'
assert len(os.listdir(base_graphs_directory)), 'No graph in the base graphs folder.'

try:
    os.mkdir(graphs_with_services_directory)
except FileExistsError:
    pass

    try:
        mkdir(join('graphs_with_services', graph_name))
    except FileExistsError:
        pass