import numpy as np
import coordinate_functions as cf
import node_functions as nf

"""
does greedy forwarding including edge weight it stops when not getting closer
"""

def greedy_forwarding_with_edge_weight(id1, id2, graph, ratio_travelled=False, plot_stuck=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  sec_travelled = 0
  min_distance = inf
  min_edge_weight = 0
  while (current_node != id2):

    for _ , neighbor_node, data in graph.edges(current_node, data=True): # calculate from every neighbour

      if neighbor_node != current_node: # can be omitted since cycles were eliminated
        new_distance = cf.distance(id2, neighbor_node, graph) +  data['length']
        if new_distance < min_distance: # needs to keep decreasing
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = data['travel_time']
    
    if node_with_min_distance in visited or min_distance == inf: # can't be together with the above if else could be referenced before assignment
      if plot_stuck is True: # if we want to plot explicitely where we are greedy_forwarding
          nf.plot_stuck(id1, id2, current_node, graph, given_depth=1)

      
      if ratio_travelled:
        return inf, cf.distance(id1, current_node, graph) / cf.distance(id2, id1, graph)
      
      return inf

    sec_travelled += min_edge_weight
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled
