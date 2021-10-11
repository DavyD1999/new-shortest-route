import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import networkx as nx
import a_star

"""
does normal greedy forwarding till a node is already in visited then do a_star
"""

def greedy_forwarding_then_a_star(id1, id2, graph): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  distance_travelled = 0

  while (current_node != id2):
    
    min_distance = inf
    for _, neighbor_node in graph.edges(current_node):
      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.distance(id2, neighbor_node, graph) # end node is id2 so try to get closer to this end node
        if new_distance < min_distance:
          node_with_min_distance = neighbor_node
          min_distance = new_distance

    if min_distance == inf:
      return inf 
    
    if node_with_min_distance in visited: # if empty next sequence then already stop here else a star won't even find a path
      """
      if the node was already visited or the node that will be visited does not seem to have any neighbors
      """
      distance_travelled += a_star.A_star(current_node, id2, graph)
      return distance_travelled

    edge_length = nf.get_edge_length(current_node, node_with_min_distance, graph)

    distance_travelled += edge_length
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  return distance_travelled