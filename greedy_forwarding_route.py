import numpy as np
import coordinate_functions as cf
from collections import deque
from fix_graph_data import load_graph
import osmnx as ox
import networkx as nx
import matplotlib
"""
does normal greedy forwarding it stops when a cycle is discovered or no neighbors are found (which is only possible in a directed graph)
"""

def greedy_forwarding(id1, id2, graph, ratio_travelled=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()
  route = list() # list of nodes which brings us to an end point
  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  route.append(current_node)
  sec_travelled = 0
  min_distance = inf
  min_edge_weight = 0

  while (current_node != id2):
    
    # min_distance will keep decreasing like we want if it doesn't decrease it will keep the same node with min distance and thus greedy forwarding will fail
    for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every out
      if neighbor_node != current_node: # eliminate cycles
        new_distance = cf.distance(id2, neighbor_node, graph) # end node is id2 so try to get closer to this end node
        if new_distance < min_distance:
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = edge_weight

    if min_distance == inf or current_node == node_with_min_distance:
      """
      matplotlib.use('Agg')
      routenx = nx.shortest_path(graph, start, end, 'travel_time')
      
      print(route)

      fig, _ = ox.plot.plot_graph_routes(graph, [list(route), routenx], route_colors = ['r', 'g'], route_linewidth=1, node_size=0, orig_dest_size=3)

      fig.savefig('testgreedy.png', dpi=500)
      fig.clf()
      """
      if ratio_travelled:
        return inf, cf.distance(id2, current_node, graph) / cf.distance(id2, id1, graph)
      
      return inf
        

    sec_travelled += min_edge_weight
    current_node = node_with_min_distance
    route.append(current_node) 
  
  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled

