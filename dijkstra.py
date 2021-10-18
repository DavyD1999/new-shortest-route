import osmnx as ox
import numpy as np
import time
import node_functions as nf
"""
does dijkstra on a certain node to all nodes
"""

# graph_basic = ox.io.load_graphml('manhattan_5km_(40.754932, -73.984016).graphml')
inf = np.inf

def dijkstra(id1, graph):
    """
    function that calculates the time from node_id1 first argument to all other nodes given graph graph
    """
    
    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"

    times = dict()
    for node_id in unvisited_nodes:
        times[node_id] = inf # every element infinite except for the starnode
    times[id1] = 0 # overwrite time for our start node

    visited_nodes = set()
    current_node = id1
    
    for _ in range(len(graph.nodes())): # we will try every node for the time
        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every neighbour
          
          new_weight = edge_weight +  times[current_node]
          if times[neighbor_node] > new_weight:
            times[neighbor_node] = new_weight
            
        
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        minimum = inf
        
        for key, value in times.items(): # find value with lowest time to walk through now
            if value < minimum and key in unvisited_nodes:
                minimum = value
                minimum_key = key

        if minimum == inf:
            #print('some roads have no connection')
            break
        current_node = minimum_key

    return times

# print(dijkstra(9121386338,graph_basic)[1692433918])

def dijkstra_to_node(id1, id2, graph):
    """
    function that calculates the time from node_id1 first argument to id2 given graph graph
    """
    
    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"

    times = dict()
    for node_id in unvisited_nodes:
        times[node_id] = inf # every element infinite except for the starnode
    times[id1] = 0 # overwrite time for our start node

    visited_nodes = set()
    current_node = id1
    
    while id2 not in visited_nodes: # we will try every node for the time
        
      for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every neighbour
          
          new_weight = edge_weight +  times[current_node]
          if times[neighbor_node] > new_weight:
            times[neighbor_node] = new_weight
              
          
      visited_nodes.add(current_node)

      unvisited_nodes.remove(current_node)
      minimum = inf
      
      for key, value in times.items(): # find value with lowest time to walk through now
          if value < minimum and key in unvisited_nodes:
              minimum = value
              minimum_key = key

      if minimum == inf:
          #print('some roads have no connection')
          break
      current_node = minimum_key

    return times[id2]

# print(dijkstra_to_node(9121386338,1692433918, graph_basic))