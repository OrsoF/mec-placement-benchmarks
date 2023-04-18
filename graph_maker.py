from networkx.readwrite.gml import read_gml, write_gml
import networkx as nx
import os
import numpy as np
import pickle

def graph_design(graphs, root):
    for graph in graphs:

        ###### STANDARD GRAPH

        # Create a directory 'data' and one for each instance
        try:
            os.mkdir('data')
        except:
            pass

        try:
            os.chdir('data')
            os.mkdir(graph)
            os.mkdir(graph+'_completed')
        except:
            pass

        os.chdir(root)

        graph_name = graph+'.gml'
        G = nx.DiGraph(read_gml(graph_name))
        G = nx.relabel.relabel_nodes(G, {node : i for i, node in enumerate(G.nodes())})

        # We add the symmetric arc to make it symmetric

        def make_graph_symmetric(G):
            for arc in G.edges:
                a, b = arc
                G.add_edge(b, a)

        make_graph_symmetric(G)

        # We remove attributes to write personal attributes

        def remove_all_attributes(G):
            for node in G.nodes:
                keys = list(G.nodes[node].keys())
                for key in keys:
                    del G.nodes[node][key]
            for edge in G.edges:
                keys = list(G.edges[edge].keys())
                for key in keys:
                    del G.edges[edge][key]
            return G

        G = remove_all_attributes(G)

        # 10**6 resources, 1000 capacity, 1/5 alpha, 5 beta

        nx.classes.function.set_node_attributes(G, {node : 10**4 for node in G.nodes}, 'resources')
        nx.classes.function.set_edge_attributes(G, {edge : {'capacity' : 200+200*len(G.nodes)//15, 'alpha' : 1./5, 'beta' : 5} for edge in G.edges})

        write_gml(G, 'data/'+graph+'/'+graph_name)

        def generate_service(G, D, latency, resources, flow):
            service = []
            for j in range(D):
                dic = {}
                dic['origin'] = np.random.choice(G.nodes)
                dic['pmin'] = 1
                dic['pmax'] = flow
                dic['resources'] = resources
                dic['latency'] = latency
                service.append(dic)
            return service
        alpha_moyen, beta_moyen = sum(G.edges[edge]['alpha'] for edge in G.edges)/len(G.edges), sum(G.edges[edge]['beta'] for edge in G.edges)/len(G.edges)

        for i in range(1, 11):
            services_i = [generate_service(G, D = i, latency = 7.*(alpha_moyen*5.*i**2/len(G.edges)+beta_moyen), resources = 5000//i, flow = 100) for j in range(i)]
            if i < 10:
                file_name = 'data/'+graph+'/'+graph_name[0]+'_0'+str(i)+'.pkl'
            else:
                file_name = 'data/'+graph+'/'+graph_name[0]+'_'+str(i)+'.pkl'
            f = open(file_name, "w+")
            f.close()
            f = open(file_name, 'wb')
            pickle.dump(services_i, f, protocol=2)
            f.close()

        ##### COMPLETED GRAPH

        def complete_graph(G):
            n = len(G.nodes)

            # Create 10*n edge servers
            labels = list(range(n, 11*n))
            G.add_nodes_from(labels)

            # Connect each edge server to a given server

            server_edge_server_connection = [(labels[i], np.random.randint(n)) for i in range(10*n)]
            G.add_edges_from(server_edge_server_connection)
            make_graph_symmetric(G)

            # Create 20*n users
            second_labels = list(range(11*n, 31*n))
            G.add_nodes_from(second_labels)

            # Connect each user with four random edges servers
            G.add_edges_from([(second_labels[i], np.random.randint(n, 11*n)) for i in range(20*n) for j in range(4)])
            make_graph_symmetric(G)

            resources = [10**6]*n+[10**4]*10*n+[10**2]*20*n 
            nx.classes.function.set_node_attributes(G, {node : resources[i] for i, node in enumerate(G.nodes)}, 'resources')
            
            capacities = [0]*len(G.edges)
            arc_dic = {edge : i for i, edge in enumerate(G.edges)}

            for edge in G.edges:
                node1, node2 = edge
                if node1 < n and node2 < n:
                    capacities[arc_dic[(node1, node2)]] = 5000
                elif node1 < n or node2 < n:
                    capacities[arc_dic[(node1, node2)]] = 3000
                else:
                    capacities[arc_dic[(node1, node2)]] = 100

            capacities = {edge : capacities[i] for i, edge in enumerate(G.edges)}
            nx.classes.function.set_edge_attributes(G, capacities, 'capacity')

            beta = [0]*len(G.edges)

            for edge in G.edges:
                node1, node2 = edge
                if node1 < n and node2 < n:
                    beta[arc_dic[(node1, node2)]] = 5
                elif node1 < n or node2 < n:
                    beta[arc_dic[(node1, node2)]] = 7
                else:
                    beta[arc_dic[(node1, node2)]] = 15

            beta_ = {edge : beta[i] for i, edge in enumerate(G.edges)}
            nx.classes.function.set_edge_attributes(G, beta_, 'beta')

            alpha = [0]*len(G.edges)

            for edge in G.edges:
                node1, node2 = edge
                if node1 < n and node2 < n:
                    alpha[arc_dic[(node1, node2)]] = 5*10**-1
                elif node1 < n or node2 < n:
                    alpha[arc_dic[(node1, node2)]] = 7*10**-1
                else:
                    alpha[arc_dic[(node1, node2)]] = 15*10**-1

            alpha_ = {edge : alpha[i] for i, edge in enumerate(G.edges)}
            nx.classes.function.set_edge_attributes(G, alpha_, 'alpha')

            return G

        G = complete_graph(G)
        
        write_gml(G, 'data/'+graph+'_completed/'+graph+'_completed.gml')

        def generate_service_completed(G, D, latency, resources, flow):
            n = len(G.nodes)//31
            possible_origins = list(range(n))+list(range(21*n, 31*n))
            service = []
            for j in range(D):
                dic = {}
                dic['origin'] = np.random.choice(possible_origins)
                dic['pmin'] = 1
                dic['pmax'] = flow
                dic['resources'] = resources
                dic['latency'] = latency
                service.append(dic)
            return service

        alpha_moyen, beta_moyen = sum(G.edges[edge]['alpha'] for edge in G.edges)/len(G.edges), sum(G.edges[edge]['beta'] for edge in G.edges)/len(G.edges)

        for i in range(1, 11):
            services_i = [generate_service_completed(G, D = i, latency = 5.*(alpha_moyen*5.*i**2/len(G.edges)+beta_moyen), resources = 5000//i, flow = 100) for j in range(i)]
            if i < 10:
                file_name = 'data/'+graph+'_completed/'+graph_name[0]+'_0'+str(i)+'.pkl'
            else:
                file_name = 'data/'+graph+'_completed/'+graph_name[0]+'_'+str(i)+'.pkl'
            f = open(file_name, "w+")
            f.close()
            f = open(file_name, 'wb')
            pickle.dump(services_i, f, protocol=2)
            f.close()