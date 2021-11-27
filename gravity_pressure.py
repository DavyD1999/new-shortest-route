import numpy as np
import coordinate_functions as cf
import collections

"""
does the gravity pressure algorithm as found in Hyperbolic Embedding and Routing for
Dynamic Graphs
"""

def gravity_pressure(id1, id2, graph, distance_function=cf.distance, ratio_travelled=False, plot_stuck=False): # id1 is start node id2 is go to node
  
  inf = np.inf
  total_nodes = graph.nodes()
  
  route = list() # list of nodes which brings us to an end point
  visits = collections.Counter() # every element not yet in dict has count zero

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  route.append(current_node)

  sec_travelled = 0

  gravity_mode = True
  min_distance = inf

  while (current_node != id2):
    
     if gravity_mode is True:

        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'):
      
            new_distance = distance_function(id2, neighbor_node, graph) + edge_weight
            
            if new_distance < min_distance:
                node_with_min_distance = neighbor_node
                min_distance = new_distance
                min_edge_weight = edge_weight

        if current_node == node_with_min_distance:
      # enter the pressure mode
                gravity_mode = False
                visits[current_node] += 1 
                d_v = min_distance # distance that needs to get beaten in the pressure mode
     else: # pressure mode

        canidates = set()
        current_min_visits = np.inf
        
        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'):
            # find the canidates
            if visits[neighbor_node] < current_min_visits:
                canidates = set() 
                canidates.add(neighbor_node)
                current_min_visits = visits[neighbor_node]
            
            elif visits[neighbor_node] == current_min_visits:
                canidates.add(neighbor_node)
        

        min_distance = inf
        for node in canidates: # try to minimize the loss so get as close as possible
            new_distance = distance_function(id2, node, graph) + graph[node][current_node]['travel_time']
            if new_distance < min_distance:
                node_with_min_distance = node
                min_distance = new_distance
                min_edge_weight = edge_weight

        visits[current_node] += 1

        if new_distance < d_v: # if progress compared to the gravity mode was booked then switch back
            gravity_mode = True
        
     sec_travelled += min_edge_weight
     current_node = node_with_min_distance
     route.append(current_node) 

  if ratio_travelled:
    return sec_travelled, 1 # reached the end
  
  return sec_travelled

