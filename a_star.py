"""
DO NOT FORGET TO ACTIVATE THE ENV
this file is able to calculate the shortest distance using a star with the haversine distance
"""
import osmnx as ox
import numpy as np
import coordinate_functions as cf
import node_functions as nf

graph_basic = ox.io.load_graphml('manhattan_5km_(40.754932, -73.984016).graphml') # change file name here

def precompute_map_haversine_vs_real_distance(graph): # not such a useful function
  actual_distance = dict()
  haversine_distance = dict()
  for current_node in graph.nodes(): # we will try every node for the distance
    for neighbor_node in graph.neighbors(current_node):
      if neighbor_node != current_node:
        
        try: # it might be a one way street
            edge_val = graph[neighbor_node][current_node].values()
        except:
            edge_val = graph[current_node][neighbor_node].values()
        key = (current_node, neighbor_node) # needs to be immutable
        for edge in edge_val:
          if key in actual_distance.keys():
            if edge.get('length') < actual_distance[key]:
              actual_distance[key] = edge.get('length')
          elif key[::-1] in actual_distance:
            key = key[::-1]
            
            if edge.get('length') < actual_distance[key]:
              actual_distance[key] = edge.get('length')
          else:
            actual_distance[key] = edge.get('length')
            haversine_distance[key] = cf.distance(current_node, neighbor_node, graph)
  horizontal_list = list()
  vertical_list = list()
  
  for key, value in actual_distance.items():
    horizontal_list.append(value)
    vertical_list.append(value/haversine_distance[key])
  
  print(np.mean(vertical_list))
  print(np.std(vertical_list))


def A_star(id1, id2, graph): # id1 is start node id2 is go to node
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
    f_score[id1] = cf.euclid_distance(id1, id2, graph)
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

        for neighbor_node in graph.neighbors(current_node): 
            
            distance = nf.get_edge_length(neighbor_node, current_node, graph)
        
            tentative_g_score = g_score[current_node] + distance

            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score
                f_score[neighbor_node] = g_score[neighbor_node] + cf.euclid_distance(id1, id2, graph)

                open_set.add(neighbor_node) # this will check if already in it so just add it

    return inf

print(A_star(9121386338,1692433918,graph_basic))
#precompute_map_haversine_vs_real_distance(graph_basic)    