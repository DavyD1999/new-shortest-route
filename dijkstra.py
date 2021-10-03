import osmnx as ox
import numpy as np
import time
import node_functions as nf
"""
does dijkstra on a certain node to all nodes
"""

graph_basic = ox.io.load_graphml('manhattan_5km_(40.754932, -73.984016).graphml')
inf = np.inf

def dijkstra(id1, graph):
    """
    function that calculates the distance from node_id1 first argument to all other nodes given graph graph
    """
    
    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"

    distances = dict()
    for node_id in unvisited_nodes:
        distances[node_id] = inf # every element infinite except for the starnode
    distances[id1] = 0 # overwrite distance for our start node

    visited_nodes = set()
    current_node = id1
    
    for _ in range(len(graph.nodes())): # we will try every node for the distance
        for neighbor_node in graph.neighbors(current_node): # calculate from every neighbour

          new_edge_length = nf.get_edge_length(current_node, neighbor_node, graph) + distances[current_node]
          if distances[neighbor_node] > new_edge_length:
            distances[neighbor_node] = new_edge_length
            
        
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        minimum = inf
        
        for key, value in distances.items(): # find value with lowest distance to walk through now
            if value < minimum and key in unvisited_nodes:
                minimum = value
                minimum_key = key

        if minimum == inf:
            #print('some roads have no connection')
            break
        current_node = minimum_key

    return distances

print(dijkstra(9121386338,graph_basic)[1692433918])
