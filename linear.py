import numpy as np
import coordinate_functions as cf
import networkx as nx
import random
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.svm
from queue import PriorityQueue
import time

def list_maker(euclid_distance, previous_euclid, amount_travelled, travel_time):
    difference = euclid_distance - previous_euclid
    #sqep = difference * difference

    # new = 0.001 * (amount_travelled ** 2 + 10) / (amount_travelled ** 4 + 1) 

    # new2 =  (euclid_distance ** 4 + 1) / (euclid_distance ** 2 + euclid_distance ** 4 + 10)
    
    return np.array([amount_travelled, euclid_distance, travel_time, difference,  1/amount_travelled, 1/difference,1/(euclid_distance+0.1), 1/travel_time])

def add_amount_of_visited_weights(graph, number_of_landmarks, cutoff=False):

    landmarks_tot = random.sample(list(graph.nodes()), number_of_landmarks) # without repetition

    landmarks1 = landmarks_tot[:len(landmarks_tot)//4]
    landmarks2 = landmarks_tot[len(landmarks_tot)//4:]
    path_list = list()
    
    for u, v in graph.edges():
        graph[u][v]['amount_travelled'] = 1
    for landmark in landmarks1:
        distances, paths = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        for i, value in enumerate(paths.values()):
            for x in range(len(value)-1): # -1 cause we can't count too for out of the list range
                graph[value[x]][value[x+1]]['amount_travelled'] += 0.1

        path_list.append(paths)

    x_list, y_list = list(), list()
    
    for landmark in landmarks2: 
        print(landmark)
        distances, paths = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        for i, value in enumerate(paths.values()):
            
            if  np.random.uniform(0.0, 1.0) < 0.001 and cutoff is True:
                print(i)
                break
            
            
            for x in range(len(value)-1): # -1 cause we can't count too for out of the list range
                previous = cf.euclid_distance(value[x], value[-1], graph)
                
                euclid_distance = cf.euclid_distance(value[x+1], value[-1], graph)

                amount_travelled = graph[value[x]][value[x+1]]['amount_travelled']
                travel_time = graph[value[x]][value[x+1]]['travel_time']


                x_list.append(list_maker(euclid_distance, previous, amount_travelled, travel_time))
             
                # eucldian distance between the destination and the neighbor node we are actually going to
                y_list.append(1 + 5 * distances[value[x+1]]/distances[value[-1]]) # used to be just one
                
                neighbors = list(graph.neighbors(value[x]))
                neighbors.remove(value[x+1])

                if neighbors != []: # if other neighbors are available
                    chosen_neighbor = neighbors[0]
                    amount_travelled = graph[value[x]][chosen_neighbor]['amount_travelled']
                    euclid_distance = cf.euclid_distance(chosen_neighbor, value[-1], graph)
                    travel_time = graph[value[x]][chosen_neighbor]['travel_time']

                    x_list.append(list_maker(euclid_distance, previous, amount_travelled, travel_time))
                    
                    y_list.append(0)

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_list, y_list, test_size=0.25, random_state=42, shuffle=True)
    
    #linear_regression = sklearn.svm.SVR(kernel='linear', max_iter = 50000)
    linear_regression = sklearn.linear_model.LinearRegression()
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print(x_train.shape)

    clf = linear_regression.fit(x_train, y_train)

    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    print(clf.coef_)

    return clf


def output_function(weights, vals):
    return np.sum(weights * vals)

def priority_queue_new_evaluation_function(id1, id2, graph, weight_function,ratio_travelled=False, return_counter=False): # id1 is start node id2 is go to node
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

                euclid_distance = cf.euclid_distance(neighbor_node, id2, graph)
                # amount_travelled,  euclid_distance, graph[value[x]][chosen_neighbor]['travel_time'], euclid_distance-previous, 1/amount_travelled, 1/(0.1+euclid_distance), 1 / graph[value[x]][chosen_neighbor]['travel_time']

                travel_time = graph[current_node][neighbor_node]['travel_time']
                for_function = list_maker(euclid_distance, previous, amount_travelled, travel_time)
                
                f_score[neighbor_node] = - output_function(weight_function.coef_, for_function) # expand lowest first
                
                priority_queue.put((f_score[neighbor_node], neighbor_node))

        visited.add(current_node)

    return inf