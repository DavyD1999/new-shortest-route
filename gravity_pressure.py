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
random.seed(42)

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

def add_amount_of_visited_weights(graph, number_of_landmarks):

    landmarks_tot = random.sample(list(graph.nodes()), number_of_landmarks) # without repetition

    landmarks1 = landmarks_tot[:len(landmarks_tot)//2]
    landmarks2 = landmarks_tot[len(landmarks_tot)//2:]
    path_list = list()
    
    for u, v in graph.edges():
        graph[u][v]['amount_travelled'] = 1
    for landmark in landmarks1:
        distances, paths = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        for value in paths.values():
            for x in range(len(value)-1): # -1 cause we can't count too for out of the list range
                graph[value[x]][value[x+1]]['amount_travelled'] += 0.1

        path_list.append(paths)

    x_list, y_list = list(), list()

    landmarks = random.choices(list(graph.nodes()), k= 2 * number_of_landmarks//4) # new landmarks else amount of times visited makes no sense
    print(landmarks)
    
    for landmark in landmarks2: 
        distances, paths = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        for value in paths.values():

            for x in range(len(value)-1): # -1 cause we can't count too for out of the list range
                previous = cf.euclid_distance(value[x], value[-1], graph)
                
                euclid_distance = cf.euclid_distance(value[x+1], value[-1], graph)

                amount_travelled = graph[value[x]][value[x+1]]['amount_travelled']

                x_list.append(np.array([amount_travelled, euclid_distance, graph[value[x]][value[x+1]]['travel_time'], euclid_distance-previous, 1/amount_travelled, 1/(0.1 + euclid_distance), 1 / graph[value[x]][value[x+1]]['travel_time']]))
                # x_list.append([amount_travelled, euclid_distance, graph[value[x]][value[x+1]]['travel_time'], euclid_distance-previous])
                # eucldian distance between the destination and the neighbor node we are actually going to
                y_list.append(1 + 5 * distances[value[x+1]]/distances[value[-1]]) # used to be just one
                
                neighbors = list(graph.neighbors(value[x]))
                neighbors.remove(value[x+1])

                if neighbors != []: # if other neighbors are available
                    chosen_neighbor = neighbors[0]
                    amount_travelled = graph[value[x]][chosen_neighbor]['amount_travelled']
                    euclid_distance = cf.euclid_distance(chosen_neighbor, value[-1], graph)

                    x_list.append(np.array([amount_travelled,  euclid_distance, graph[value[x]][chosen_neighbor]['travel_time'], euclid_distance-previous, 1/amount_travelled, 1/(0.1+euclid_distance), 1 / graph[value[x]][chosen_neighbor]['travel_time']]))
                    #x_list.append([amount_travelled,  euclid_distance, graph[value[x]][chosen_neighbor]['travel_time'], euclid_distance-previous])
                    y_list.append(0)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_list, y_list, test_size=0.25, random_state=42, shuffle=True)

    scaler = sklearn.preprocessing.StandardScaler(copy=False)
    #x_train = scaler.fit_transform(x_train)
    #x_test = scaler.transform(x_test)
    
    linear_regression = sklearn.linear_model.LinearRegression()

    clf = linear_regression.fit(x_train, y_train)

    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    print(clf.coef_)

    return clf, scaler
    
#graph = nx.read_gpickle(f'./graph_pickle/Brugge.gpickle')
#add_amount_of_visited_weights(graph, 7)

def output_function(weights, vals):

    return np.sum(weights * vals)
                
    
def weighted_function(weight_difference_distance, distance, amount_travelled):
    if distance < 10 ** -5: # reached destination
        return - np.inf 

    return weight_difference_distance * np.log(distance) - amount_travelled # minus since we want to decrease our distance

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
 
"""
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
        function_result = - output_function(weight_function.coef_, np.array([amount_travelled, euclid_distance, graph[current_node][neighbor_node]['travel_time']])) # minus since this predicts prob of going to this node so higher -> better
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
"""

"""
 
  uses a weighted function to decide which step is better

  inf = np.inf
  total_nodes = graph.nodes()
  
  route = list() # list of nodes which brings us to an end point
  visits = collections.Counter() # every element not yet in dict has count zero

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  current_node = id1
  route.append(current_node)

  sec_travelled = 0

  gravity_mode = True
  min_function = inf

  while (current_node != id2):

     if gravity_mode is True:

        for _ , neighbor_node, amount_travelled in graph.edges(current_node, data = 'amount_travelled'):
      
            euclid_distance = cf.euclid_distance(id2, neighbor_node, graph)
            function_result = - weight_function.predict(np.array([amount_travelled, euclid_distance, graph[current_node][neighbor_node]['travel_time']]).reshape(1, -1))
            print(function_result)
            if function_result < min_function:
                node_with_min_distance = neighbor_node
                min_function = function_result
                min_travel_time = graph[current_node][neighbor_node]['travel_time']

        if current_node == node_with_min_distance:  
      # enter the pressure mode
            gravity_mode = False
            visits[current_node] += 1 
            d_v = min_function # distance that needs to get beaten in the pressure mode
            continue
     
     else: # pressure mode

        canidates = dict()
        current_min_visits = np.inf
        
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
            function_result = - weight_function.predict(np.array([amount_travelled, euclid_distance, graph[current_node][neighbor_node]['travel_time']]).reshape(1, -1)) # minus since this predicts prob of going to this node so higher -> better
            
            if function_result < min_function:
                node_with_min_distance = neighbor_node
                min_function = function_result
                min_travel_time = graph[current_node][neighbor_node]['travel_time']
                
        visits[current_node] += 1

        if min_function < d_v: # if progress compared to the gravity mode was booked then switch back
            gravity_mode = True
            continue

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

"""

def priority_queue_new_evaluation_function(id1, id2, graph, weight_function, scaler,ratio_travelled=False, return_counter=False): # id1 is start node id2 is go to node
    inf = np.inf
    # heuristic function 
    total_nodes = graph.nodes()

    assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

    came_from = dict()

    visited = set()
    priority_queue = PriorityQueue()

    f_score = dict()
    g_score = dict()

    for node in total_nodes:
        f_score[node] = inf
        g_score[node] = inf
    
    g_score[id1] = 0
    f_score[id1] = 0
    priority_queue.put((f_score[id1], id1))

    teller = 1
    
    while priority_queue.empty() is False:
        _ , current_node = priority_queue.get() # first attribute is the weight
    
        if current_node in visited: # don't want to visit same node twice
            continue
        
        if current_node == id2:
            
            if ratio_travelled is False:
                
                return g_score[id2]
                
            if return_counter is True:
                return g_score[id2], 1,teller
            return g_score[id2], 1 
        
        teller += 1
        previous = cf.euclid_distance(current_node, id2, graph)
        for _ ,neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): #first one is the current node the last argument makes sure we get the length
           
            tentative_g_score = g_score[current_node] + edge_weight
            

            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score 
                
                amount_travelled = graph[current_node][neighbor_node]['amount_travelled']
                # start_time = time.time()
                euclid_distance = cf.euclid_distance(neighbor_node, id2, graph)
                # print(f'eulid distance time {time.time()-start_time}')
                # amount_travelled,  euclid_distance, graph[value[x]][chosen_neighbor]['travel_time'], euclid_distance-previous, 1/amount_travelled, 1/(0.1+euclid_distance), 1 / graph[value[x]][chosen_neighbor]['travel_time']

                # start_time = time.time()
                for_function = np.array([[amount_travelled,  euclid_distance, graph[current_node][neighbor_node]['travel_time'], euclid_distance-previous, 1/amount_travelled, 1/(0.1+euclid_distance), 1 / graph[current_node][neighbor_node]['travel_time']]])
                #print(f'array making time {time.time()-start_time}')
                # for_function = np.array([[amount_travelled,  euclid_distance, graph[current_node][neighbor_node]['travel_time'], euclid_distance-previous]])
                #start_time = time.time()
                #for_function = scaler.transform(for_function)
                #print(f'scaler doing time {time.time()-start_time}')
                #start_time = time.time()
                f_score[neighbor_node] = - output_function(weight_function.coef_, for_function) # expand lowest first
                #print(f'output function doing time {time.time()-start_time}')
                priority_queue.put((f_score[neighbor_node], neighbor_node))

        visited.add(current_node)

    return inf