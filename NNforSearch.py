import numpy as np
from queue import PriorityQueue  
import coordinate_functions as cf
import time
import tensorflow as tf

def NN_predict_travel_time(id1, id2, graph, scaler, embedding, model):
    vec_1 = embedding[str(id1)][0:8]
    vec_2 = embedding[str(id2)][0:8]

    dist = cf.euclid_distance(id1, id2, graph)

    result = np.array(list(vec_1) + list(vec_2), dtype=np.float16)
    np.append(result,dist) # same type of input data needs to be given
    
    tot_rescaled = scaler.transform(result.reshape(1,-1)) # rescale it with the original scaler

    input = tf.convert_to_tensor(tot_rescaled, dtype=tf.float16)

    #return (np.min((vec_1-vec_2)))

    return model.predict_on_batch(input)[0][0]

def A_star_priority_queue_NN(id1, id2, graph, scaler, embedding, model): # id1 is start node id2 is go to node
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
    f_score[id1] = NN_predict_travel_time(id1, id2, graph, scaler, embedding, model)# f score is a lower bound
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
                f_score[neighbor_node] = g_score[neighbor_node] + NN_predict_travel_time(neighbor_node, id2, graph, scaler, embedding, model)
                priority_queue.put((f_score[neighbor_node], neighbor_node))

        visited.add(current_node)

    return inf