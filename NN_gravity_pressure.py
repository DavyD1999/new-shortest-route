import numpy as np
import coordinate_functions as cf
import collections
from gravity_pressure import remove_doubles2


def gravity_pressure_embedding(id1, id2, graph,  embedding, distance_function=cf.l2, ratio_travelled=False): # id1 is start node id2 is go to node 
  """
  does the gravity pressure algorithm as found in Hyperbolic Embedding and Routing for
  Dynamic Graphs 10.1109/infcom.2009.5062083
  """

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
      
            new_distance = distance_function(id2, neighbor_node, embedding)
            
            if new_distance < min_distance:
                node_with_min_distance = neighbor_node
                min_distance = new_distance
                min_edge_weight = edge_weight

        if current_node == node_with_min_distance:
      # enter the pressure mode
            gravity_mode = False
            visits[current_node] += 1 
            d_v = min_distance # distance that needs to get beaten in the pressure mode
            continue

     
     else: # pressure mode

        canidates = dict()
        current_min_visits = np.inf
        
        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'):
            # find the canidates
            if visits[neighbor_node] < current_min_visits:
                canidates = dict() 
                canidates[neighbor_node] = edge_weight
                current_min_visits = visits[neighbor_node]
            
            elif visits[neighbor_node] == current_min_visits:
                canidates[neighbor_node] = edge_weight
        
        min_distance = inf
        for node, edge_weight in canidates.items(): # try to minimize the loss so get as close as possible
            new_distance = distance_function(id2, node, embedding)
            if new_distance < min_distance:
                node_with_min_distance = node
                min_distance = new_distance
                min_edge_weight = edge_weight
                

        visits[current_node] += 1

        if min_distance < d_v: # if progress compared to the gravity mode was booked then switch back
            gravity_mode = True
            continue

     sec_travelled += min_edge_weight
     current_node = node_with_min_distance
     route.append(current_node) 
  

  # now we can substract the stretch we did too much aka the double or triple visited nodes
  route_removed_extra_steps = remove_doubles2(route)

  assert id1 == route_removed_extra_steps[0] and id2 == route_removed_extra_steps[-1], 'something went wrong in the remove doubles'
  
  i = 0
  sec_travelled_removed = 0
  while i < len(route_removed_extra_steps) - 1: # recount the travel time with this new route
      sec_travelled_removed += graph[route_removed_extra_steps[i]][route_removed_extra_steps[i+1]]['travel_time']
      i += 1

  assert sec_travelled >= sec_travelled_removed, 'wow this is strange removing nodes adds length'

  if ratio_travelled:
    return sec_travelled_removed, 1 # reached the end
  
  return sec_travelled_removed

