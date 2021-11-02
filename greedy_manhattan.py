import numpy as np
import coordinate_functions as cf

"""
does greedy forwarding with manhattan distance it stops when not getting closer
"""

def manhattan_greedy_forwarding(id1, id2, graph, ratio_travelled=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()
  route = list() # list of nodes which brings us to an end point
  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  route.append(current_node)
  sec_travelled = 0
  min_distance = inf # will need to keep decreasing 
  min_edge_weight = 0

  while (current_node != id2):
    
    for neighbor_node in graph.neighbors(current_node): # calculate from every out
		
       new_distance = cf.distance_manhattan(neighbor_node, id2, graph) # end node is id2 so try to get closer to this end node
       #print(f'{new_distance} dit is een voorstel afstand' )
       if new_distance < min_distance:
        
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = graph[neighbor_node][current_node]['travel_time']

    #print(f'{min_distance} dit is de gekozen afstand')
    if min_distance == inf or current_node == node_with_min_distance:

      if ratio_travelled:
        return inf, cf.distance(id1, current_node, graph) / cf.distance(id2, id1, graph)
      
      return inf

    sec_travelled += min_edge_weight
    current_node = node_with_min_distance
    route.append(current_node) 
  
  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled



