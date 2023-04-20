from os import listdir, mkdir
from os.path import join
import numpy as np

from networkx.readwrite.gml import read_gml

from utils.util import network_graph_design, base_graphs_name, base_graphs

for graph_index in range(len(base_graphs_name)):
    network_graph_design(base_graphs_name[graph_index], base_graphs[graph_index])
