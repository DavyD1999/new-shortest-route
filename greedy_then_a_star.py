import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import networkx as nx
import a_star

"""
does normal greedy forwarding till not getting closer then do a star
"""

def greedy_forwarding_then_a_star(id1, id2, graph, velocity, ratio_travelled=False, plot_stuck=False): # id1 is start node id2 is go to node, plot stuck never used cause never stuck
  inf = np.inf
  total_nodes = graph.nodes()
  
  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  sec_travelled = 0
  min_distance = inf
  min_edge_weight = 0
  while (current_node != id2):
     
    for _, neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'):
      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.distance(id2, neighbor_node, graph) # end node is id2 so try to get closer to this end node
        if new_distance < min_distance: # needs to keep decreasing
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = edge_weight

    if node_with_min_distance in visited or min_distance == inf: # if empty next sequence then already stop here else a star won't even find a path
      """
      if the node was already visited or the node that will be visited does not seem to have any neighbors
      """
      sec_travelled += a_star.A_star(current_node, id2, graph, velocity)
      if ratio_travelled:
        return sec_travelled, 1 # reached the end like always for A*
  
      return sec_travelled

    sec_travelled += min_edge_weight
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled