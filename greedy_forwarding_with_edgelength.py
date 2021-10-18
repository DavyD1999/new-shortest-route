import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import networkx as nx

"""
does greedy forwarding including edge length it stops when a cycle is discovered or no neighbors are found (which is only possible in a directed graph)
"""

def greedy_forwarding_with_edge_length(id1, id2, graph, ratio_travelled=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  distance_travelled = 0
  min_distance = inf
  min_edge_length = 0
  while (current_node != id2):

    for _ , neighbor_node, edge_length in graph.edges(current_node, data = 'length'): # calculate from every neighbour

      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.distance(id2, neighbor_node, graph) + edge_length # end node is id2 so try to get closer to this end node
        
        if new_distance < min_distance: # needs to keep decreasing
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_length = edge_length
    
    if node_with_min_distance in visited or min_distance == inf: # can't be together with the above if else could be referenced before assignment
      if ratio_travelled:
        return inf, cf.distance(id2, current_node, graph) / cf.distance(id2, id1, graph)
      
      return inf

    distance_travelled += min_edge_length
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  if ratio_travelled:
    return distance_travelled, 1 # reached the end
  
  return distance_travelled
