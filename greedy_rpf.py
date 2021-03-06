import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import networkx as nx

"""
does normal greedy forwarding with rpf metric, see Routing Metric Designs for Greedy, Face and
Combined-Greedy-Face Routing

rpf tries to find the best projection of u,v on the straight u,d with u being the node now v the neighbor and d the destination see coordinate function for exact metric
"""

def greedy_forwarding_rpf(id1, id2, graph, ratio_travelled=False, plot_stuck=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  
  sec_travelled = 0
  min_distance = inf
  min_edge_weight = 0
  visited = set()
  while (current_node != id2):
    min_distance = inf # in rpf distance is not really an actual distance and it can be negative too but it's a weight function we want to minimilize
    # min_distance will keep decreasing like we want if it doesn't decrease it will keep the same node with min distance and thus greedy forwarding will fail
    for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every out
     
      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.rpf_distance(current_node, neighbor_node, id2, graph) # end node is id2 so try to get closer to this end node
        if new_distance < min_distance:
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = edge_weight

    if min_distance == inf or node_with_min_distance in visited:

      if plot_stuck is True: # if we want to plot explicitely where we are greedy_forwarding
          nf.plot_stuck(id1, id2, current_node, graph, given_depth=1)      
      if ratio_travelled:
        return inf, cf.euclid_distance(id1, current_node, graph) / cf.euclid_distance(id2, id1, graph)
      
      return inf

    visited.add(node_with_min_distance)
    sec_travelled += min_edge_weight
    current_node = node_with_min_distance

  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled
