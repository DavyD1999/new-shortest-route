import numpy as np
import coordinate_functions as cf
import collections
import networkx as nx
import random
import sklearn
import sklearn.model_selection
import sklearn.linear_model
from queue import PriorityQueue
import time
import linear

random.seed(42)
np.random.seed(42)
def remove_doubles2(route_list):
    
    indices = dict()

    for i, val in enumerate(route_list):
        if val in indices.keys():
            indices[val] = [indices[val][0], i]
        else:
            indices[val] = [i]

    to_return = list()

    i = 0
    j = 0

    while i < len(route_list):
        if len(indices[route_list[i]]) > 1:
            to_return += route_list[j:i]

            j = indices[route_list[i]][1]
            i = indices[route_list[i]][1]

        i += 1
        
    to_return += route_list[j:len(route_list)]

    return to_return

def gravity_pressure(id1, id2, graph, distance_function=cf.euclid_distance, ratio_travelled=False, plot_stuck=False): # id1 is start node id2 is go to node 
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
      
            new_distance = distance_function(id2, neighbor_node, graph)
            
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
            new_distance = distance_function(id2, node, graph)
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


    
#graph = nx.read_gpickle(f'./graph_pickle/Brugge.gpickle')
#add_amount_of_visited_weights(graph, 7)


                

def weighted_gravity_pressure(id1, id2, graph, weight_function, ratio_travelled=False, plot_stuck=False):
  inf = np.inf
  total_nodes = graph.nodes()
  
  route = list() # list of nodes which brings us to an end point
  visits = collections.Counter() # every element not yet in dict has count zero

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  route.append(current_node)

  min_function = inf
  sec_travelled = 0 # arbitrary initialisation
    
  while (current_node != id2):

    current_min_visits = inf
    previous = cf.euclid_distance(current_node, id2, graph)
    for _ , neighbor_node, amount_travelled in graph.edges(current_node, data = 'amount_travelled'):
        # find the canidates
        if visits[neighbor_node] < current_min_visits:
            canidates = dict() 
            canidates[neighbor_node] = amount_travelled
            current_min_visits = visits[neighbor_node]
        
        elif visits[neighbor_node] == current_min_visits:
            canidates[neighbor_node] = amount_travelled
    
    min_function = inf
    for neighbor_node, amount_travelled in canidates.items(): # try to minimize the loss so get as close as possible
        euclid_distance = cf.euclid_distance(id2, neighbor_node, graph)
        
        function_result = - output_function(weight_function.coef_, np.array([np.log(amount_travelled), np.log(1 + euclid_distance), graph[current_node][neighbor_node]['travel_time'], euclid_distance-previous])) # minus since this predicts prob of going to this node so higher -> better
        if function_result < min_function:
            node_with_min_distance = neighbor_node
            min_function = function_result
            min_travel_time = graph[current_node][neighbor_node]['travel_time']
            
    visits[current_node] += 1


    sec_travelled += min_travel_time
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

  assert sec_travelled >= sec_travelled_removed - 10 ** 5, 'wow this is strange removing nodes adds length' # extra correction term because pc's can't count properly

  if ratio_travelled:
    return sec_travelled_removed, 1 # reached the end
  
  return sec_travelled_removed

