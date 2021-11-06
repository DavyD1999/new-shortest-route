import numpy as np
import coordinate_functions as cf
import networkx as nx
"""
does normal greedy forwarding it stops when a cycle is discovered or no neighbors are found (which is only possible in a directed min_tree)
"""

def hyperbolic_greedy_forwarding(id1, id2, min_tree, node_dict, ratio_travelled=False): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = min_tree.nodes()
  route = list() # list of nodes which brings us to an end point
  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the min_tree"

  current_node = id1
  route.append(current_node)
  sec_travelled = 0
  min_distance = inf # will need to keep decreasing 
  min_edge_weight = 0

  while (current_node != id2):
    
    for neighbor_node in min_tree.neighbors(current_node): # calculate from every out
		
       new_distance = cf.distance_hyperbolic(node_dict[neighbor_node], node_dict[id2]) # end node is id2 so try to get closer to this end node, in contrary to other functions here give coordinates
       #print(f'{new_distance} dit is een voorstel afstand' )
       if new_distance < min_distance:
        
          node_with_min_distance = neighbor_node
          min_distance = new_distance
          min_edge_weight = min_tree[neighbor_node][current_node]['travel_time']

    #print(f'{min_distance} dit is de gekozen afstand')
    if min_distance == inf or current_node == node_with_min_distance:
      print(nx.shortest_path_length(min_tree, current_node, id2, 'travel_time'))
      if ratio_travelled:
        print('not reached')
        return inf, cf.distance(id1, current_node, min_tree) / cf.distance(id2, id1, min_tree) 
      
      return inf

    sec_travelled += min_edge_weight
    current_node = node_with_min_distance
    route.append(current_node) 
  
  if ratio_travelled:
    print('reached')
    return sec_travelled, 1 # reached the end
  
  return sec_travelled
