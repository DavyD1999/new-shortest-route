import osmnx as ox
import numpy as np
import coordinate_functions as cf
import dijkstra
import matplotlib.pyplot as plt
"""
does dijkstra on all nodes and returns a dictionary of all starting nodes with keys as nodes the values are dictionaries( so dict within dict) with the destination node as keys and the values of the distance as values
notice this does not necessarily contain double info since there might be one way streets
"""

graph_basic = ox.io.load_graphml('kort.graphml') # change file name here
inf = np.inf

all_distance_dict = dict()
for node in graph_basic.nodes():
  
  all_distance_dict[node] = dijkstra.dijkstra(node, graph_basic)
  previous_node = node

"""
now make a graph horizontal axis says how far the haversine distance is and vertical the route distance/haversine distance
"""

xvalues = np.zeros(len(all_distance_dict)*len(all_distance_dict))
yvalues = np.zeros(len(all_distance_dict)*len(all_distance_dict))

i = 0
for start_key, value in all_distance_dict.items(): # key is start node value is a dict
  for destination_key, route_distance in value.items():
    haversine_distance = cf.distance(start_key, destination_key, graph_basic) # make sure not to calculate twice
    xvalues[i] = haversine_distance
    yvalues[i] = route_distance / haversine_distance

    i += 1

print(xvalues)
print(yvalues)

plt.plot(xvalues, yvalues, "ro")
plt.hlines(1.0, xmin=0, xmax=10000)
plt.xlim([0,10000])
plt.ylim([0.7,1.5])
plt.xlabel('haversine distance (m)')
plt.ylabel('route devided by true distance')
plt.title('longer route-->less direct route?')
plt.savefig('test1.png')

