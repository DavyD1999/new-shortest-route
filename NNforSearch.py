import numpy as np
import coordinate_functions as cf
from queue import PriorityQueue

"""
this file is able to calculate the shortest time using a star with the euclidian distance with and without priority queue
"""

def A_star(id1, id2, graph, velocity): # id1 is start node id2 is go to node
    inf = np.inf
    # heuristic function 
    total_nodes = graph.nodes()

    assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

    came_from = dict()
    open_set = set()
    open_set.add(id1)

    f_score = dict()
    g_score = dict()

    for node in total_nodes:
        f_score[node] = inf
        g_score[node] = inf
    
    g_score[id1] = 0

    f_score[id1] = cf.euclid_distance(id1, id2, graph) / velocity * 3.6 # f score is a lower bound
    empty_set = set()

    while open_set != empty_set:
        
        minimum_f = inf
        
        for node in open_set:
            # id2 is our go to node
            if f_score[node] < minimum_f:
                minimum_f = f_score[node]
                current_node = node
        
        if current_node == id2:
            return g_score[id2]

        open_set.remove(current_node)

        for _ ,neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): #first one is the current node the last argument makes sure we get the length
           
            tentative_g_score = g_score[current_node] + edge_weight 

            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score # CHANGE TO MAX VEL
                f_score[neighbor_node] = g_score[neighbor_node] + cf.euclid_distance(neighbor_node, id2, graph) / velocity * 3.6 # this will make it a good lower bound

                open_set.add(neighbor_node) # this will check if already in it so just add it
    print("hier")
    return inf

# print(A_star(9121386338,1692433918,graph_basic))
#precompute_map_haversine_vs_real_distance(graph_basic)    



def A_star_priority_queue(id1, id2, graph, velocity): # id1 is start node id2 is go to node
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
    f_score[id1] = cf.euclid_distance(id1, id2, graph) / velocity * 3.6 # f score is a lower bound
    priority_queue.put((f_score[id1], id1))
    
    while priority_queue.empty() is False:

        _ , current_node = priority_queue.get() # first attribute is the weight
        
        if current_node in visited: # don't want to visit same node twice
            continue
        
        if current_node == id2:
            return g_score[id2]

        for _ ,neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): #first one is the current node the last argument makes sure we get the length
           
            tentative_g_score = g_score[current_node] + edge_weight 

            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score # CHANGE TO MAX VEL
                f_score[neighbor_node] = g_score[neighbor_node] + cf.euclid_distance(neighbor_node, id2, graph) / velocity * 3.6 # this will make it a good lower bound
                priority_queue.put((f_score[neighbor_node], neighbor_node))

        visited.add(current_node)

    return inf