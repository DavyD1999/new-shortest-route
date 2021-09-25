"""
this file has all kinds of functions mainly to help calculate the haversine distance between two nodes in a graph
"""

import numpy as np

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

def distance(node_id1, node_id2, graph): # calculates the haversine distance 
  y1, x1 = get_coordinates(node_id1, graph)
  y2,x2 = get_coordinates(node_id2, graph)
  return haversine((y1,x1),(y2,x2))
