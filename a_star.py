"""
DO NOT FORGET TO ACTIVATE THE ENV
"""


import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt


graph_basic = ox.io.load_graphml('zwijnlandstraat_6_10km.graphml') # change file name here


def precompute_map_haversine_vs_real_distance(graph):
  actual_distance = dict()
  haversine_distance = dict()
  for current_node in graph.nodes(): # we will try every node for the distance
    for neighbor_node in graph.neighbors(current_node):
      if neighbor_node != current_node:
        
        try: # it might be a one way street
            edge_val = graph_basic[neighbor_node][current_node].values()
        except:
            edge_val = graph_basic[current_node][neighbor_node].values()
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
            haversine_distance[key] = haversine(get_coordinates(current_node, graph), get_coordinates(neighbor_node, graph))
  horizontal_list = list()
  vertical_list = list()
  
  for key, value in actual_distance.items():
    horizontal_list.append(value)
    vertical_list.append(value/haversine_distance[key])
  
  print(np.mean(vertical_list))
  print(np.std(vertical_list))



def get_coordinates(node_id, graph):
    coordinate = graph.nodes[node_id]
    return coordinate['y'], coordinate['x']

def degree_to_radian(angle):
    return angle / 180 * np.pi

def haversine(couple_1, couple_2): # gets two couples of latitude longitude couples

    y1, x1, y2, x2 = degree_to_radian(np.array([couple_1[0],couple_1[1],couple_2[0],couple_2[1]]))
    
    radius = 6371000 # in m
    argument = np.sin((y1 - y2) / 2) ** 2 + np.sin((x1 - x2) / 2) ** 2 * np.cos(y1) * np.cos(y2)
    return 2 * radius * np.arcsin(np.sqrt(argument))

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
    f_score[id1] = haversine(get_coordinates(id1, graph), get_coordinates(id2, graph))
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
            
            try: # it might be a one way street
                edge_val = graph_basic[neighbor_node][current_node].values()
            except:
                edge_val = graph_basic[current_node][neighbor_node].values()


            distance = inf 
            for edge in edge_val: # if two roads or more roads do connect one chose the shortest one of both
                if distance > edge.get('length'):
                    distance = edge.get('length')
        
            tentative_g_score = g_score[current_node] + distance

            if tentative_g_score < g_score[neighbor_node]:
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score
                f_score[neighbor_node] = g_score[neighbor_node] + haversine(get_coordinates(neighbor_node, graph), get_coordinates(id2, graph))

                open_set.add(neighbor_node) # this will check if already in it so just add it

    return inf
print(A_star(658924891,3214467066,graph_basic))
#precompute_map_haversine_vs_real_distance(graph_basic)    