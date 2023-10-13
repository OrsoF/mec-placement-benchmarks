import networkx as nx
from utils.check_network_graph import check_network_graph


class NetworkGraph:
    def __init__(
        self,
        base_graph: nx.Graph,
        node_resources: list[float],
        edge_capacity: list[float],
        latency_alpha_factor: list[float],
        latency_beta_factor: list[float],
    ):
        # The graph needs to be symmetric. We label nodes with integers.
        if not isinstance(base_graph, nx.DiGraph):
            self.nx_graph = nx.DiGraph(base_graph)
            self.nx_graph = nx.relabel.relabel_nodes(
                self.nx_graph, {node: i for i, node in enumerate(self.nx_graph.nodes())}
            )
            self.make_graph_symmetric()
            self.remove_all_attributes()
        else:
            self.nx_graph = base_graph
            self.make_graph_symmetric()

        check_network_graph(self.nx_graph)

        self.set_node_attribute(node_resources, "available_resources")
        self.set_edge_attribute(edge_capacity, "capacity")
        self.set_edge_attribute(latency_alpha_factor, "alpha")
        self.set_edge_attribute(latency_beta_factor, "beta")

        self.alphas: list[float] = [
            self.nx_graph.edges[edge]["alpha"] for edge in self.nx_graph.edges
        ]
        self.betas: list[float] = [
            self.nx_graph.edges[edge]["beta"] for edge in self.nx_graph.edges
        ]
        self.capacities = [
            self.nx_graph.edges[edge]["capacity"] for edge in self.nx_graph.edges
        ]
        self.available_resources = [
            self.nx_graph.nodes[node]["available_resources"]
            for node in self.nx_graph.nodes
        ]
        self.arc_index_dictionary = {
            arc: i for i, arc in enumerate(self.nx_graph.edges)
        }
        self.arc_list = list(self.nx_graph.edges)

    def set_node_attribute(self, attributes: list[float], name: str):
        assert (
            len(attributes) == self.nx_graph.number_of_nodes()
        ), "There not as many node parameters as the number of nodes in the graph."
        attribute_map = {
            node_index: attributes[node_index]
            for node_index in range(self.nx_graph.number_of_nodes())
        }
        nx.classes.function.set_node_attributes(self.nx_graph, attribute_map, name)

    def set_edge_attribute(self, attributes: list[float], name: str):
        assert (
            len(attributes) == self.nx_graph.number_of_edges()
        ), "{} parameters not equal to {} edges in graph.".format(
            len(attributes), self.nx_graph.number_of_edges()
        )
        attribute_map = {
            edge: attributes[edge_index]
            for edge_index, edge in enumerate(self.nx_graph.edges)
        }
        nx.classes.function.set_edge_attributes(self.nx_graph, attribute_map, name)

    def make_graph_symmetric(self):
        for arc in self.nx_graph.edges:
            if (arc[1], arc[0]) not in self.nx_graph.edges:
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
