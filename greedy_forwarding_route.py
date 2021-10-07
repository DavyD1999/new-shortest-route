import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import networkx as nx

"""
does normal greedy forwarding it stops when a cycle is discovered or no neighbors are found (which is only possible in a directed graph)
"""

def greedy_forwarding(id1, id2, graph): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  distance_travelled = 0

  while (current_node != id2):
    
    min_distance = inf
    for neighbor_node in graph.neighbors(current_node):
      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.euclid_distance(id2, neighbor_node, graph) # end node is id2 so try to get closer to this end node
        if new_distance < min_distance:
          node_with_min_distance = neighbor_node
          min_distance = new_distance

    if min_distance == inf:
      return inf 
    
    if node_with_min_distance in visited: # can't be together with the above if else could be referenced before assignment
      return inf

    edge_length = nf.get_edge_length(current_node, node_with_min_distance, graph)

    distance_travelled += edge_length
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  return distance_travelled
