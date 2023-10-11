import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from problem_classes.probleminstance import ProblemInstance

red_color = "tab:red"
gray_color = "tab:gray"


def plot_large_instance(
    nx_graph: nx.Graph,
    x: np.ndarray = None,
    p: np.ndarray = None,
    y: np.ndarray = None,
    problem_instance: ProblemInstance = None,
):
    services = problem_instance.service_package.services
    weights = 0.5 * np.log(3 * np.dot(x.transpose(), p) + 1) + 0.01
    size = 200 * np.sum(y, axis=0) + 100
    origins = [demand["origin"] for service in services for demand in service]
    colors = [gray_color] * len(nx_graph.nodes)
    for origin in origins:
        colors[origin] = red_color

    used_servers = np.sum(y, axis=0)

    for index, server in enumerate(used_servers):
        if server and colors[index] == red_color:
            colors[index] = "tab:purple"
        elif server and colors[index] == gray_color:
            colors[index] = "tab:blue"

    n = len(nx_graph.nodes) // 31
    for index in range(len(nx_graph.edges)):
        if index < n and colors[index] == gray_color:
            colors[index] = "black"
        elif n <= index < 11 * n and colors[index] == gray_color:
            colors[index] = "dimgray"
        elif 11 * n <= index < 31 * n and colors[index] == gray_color:
            colors[index] = "lightgray"

    pos = nx.shell_layout(nx_graph, [range(n), range(n, 11 * n), range(11 * n, 31 * n)])

    plt.axis("off")
    plt.figure(3, figsize=(12, 12))
    nx.draw_networkx_nodes(nx_graph, pos, node_color=colors, node_size=size)
    nx.draw_networkx_edges(
        nx_graph, pos, connectionstyle="arc3,rad=0.1", width=weights, arrowstyle="->"
    )

    plt.show()


def plot_small_instance(
    nx_graph: nx.Graph,
    x: np.ndarray = None,
    p: np.ndarray = None,
    y: np.ndarray = None,
    problem_instance: ProblemInstance = None,
):
    services = problem_instance.service_package.services
    pos = nx.drawing.layout.spring_layout(nx_graph)
    weights = np.log(3 * np.dot(x.transpose(), p) + 1) + 0.5
    size = 200 * np.sum(y, axis=0) + 100
    origins = [demand["origin"] for service in services for demand in service]
    colors = [gray_color] * len(nx_graph.nodes)
    for origin in origins:
        colors[origin] = red_color

    used_servers = np.sum(y, axis=0)
    # print(used_servers)
    for index, server in enumerate(used_servers):
        if server and colors[index] == red_color:
            colors[index] = "tab:purple"
        elif server and colors[index] == gray_color:
            colors[index] = "tab:blue"

    plt.axis("off")
    plt.figure(3, figsize=(12, 12))
    nx.draw_networkx_nodes(nx_graph, pos, node_color=colors, node_size=size)
    nx.draw_networkx_edges(nx_graph, pos, connectionstyle="arc3,rad=0.1", width=weights)
    nx.draw_networkx_labels(
        nx_graph, pos, labels={i: i for i in range(len(nx_graph.nodes))}
    )
    plt.show()


def plot_instance(
    nx_graph: nx.Graph,
    x: np.ndarray = None,
    p: np.ndarray = None,
    y: np.ndarray = None,
    problem_instance: ProblemInstance = None,
):
    if nx_graph.number_of_nodes() > 1000:
        plot_large_instance(nx_graph, x, p, y, problem_instance)
    else:
        plot_small_instance(nx_graph, x, p, y, problem_instance)
