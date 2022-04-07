import networkx as nx
import numpy as np

def springen(city, iterations): # NEW TECHNIQUES FOR GEOGRAPHIC ROUTING PHD PAPER

    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')

    new_coordinates = dict()

    for node in graph.nodes():
        new_coordinates[node] = np.array([node['x'], node['y']])

    # for each node we will now add the percentage of common neighbors this way we don't have to calculate this every time we will just keep this in a seperate dictionary

    percentage_dict = dict() # will be a dict of a dict with percentage of common neighbors as values of this list dict

    for node in graph.nodes():
        si = set(nx.neighbors(graph, node))
        percentage_dict[node] = dict()
        
        for neighbor in si:
            sj = set(nx.neighbors(graph, neighbor))
            sij = si.intersection(sj)
            
            percentage_dict[node][neighbor] = len(sij) / (len(sij) + len(si) + len(sj))

    print(percentage_dict)

springen('Brugge',1)
    

    