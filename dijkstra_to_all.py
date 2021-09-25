import osmnx as ox
import numpy as np
import dijkstra
import pickle
"""
does dijkstra on all nodes and returns a dictionary of all starting nodes with keys as nodes the values are dictionaries( so dict within dict) with the destination node as keys and the values of the distance as values
notice this does not necessarily contain double info since there might be one way streets
"""

graph_basic = ox.io.load_graphml('zw6_5km.graphml') # change file name here
inf = np.inf

all_distance_dict = dict()
for node in graph_basic.nodes():
  print(node)
  all_distance_dict[node] = dijkstra.dijkstra(node, graph_basic)

filename = 'all_distance_dict.pk' # write the data to a file so the expensive procedure does not to be done all over again

with open(filename, 'wb') as file:
  pickle.dump(all_distance_dict, file)





