import osmnx as ox
import networkx as nx
import time
import numpy as np
"""
this file will load the graph, make it undirected and select the largest connected component
"""

def load_graph(name):
  start_time = time.time()
  graph = ox.io.load_graphml(f'{name}.graphml')
  
  print(f'{time.time() - start_time} geladen')
  graph = ox.speed.add_edge_speeds(graph)
  graph = ox.speed.add_edge_travel_times(graph, precision=2)
  graph = ox.utils_graph.get_undirected(graph) # makes it undirected but still a multigraph

  # now take the largest component
  biggest_component = list(graph.subgraph(c) for c in nx.connected_components(graph))[0]

  to_return = biggest_component.copy() # will unfreeze the graph
  
  # go over every edge and delte the ones with higher travel times than edges with the same start and destination
  for u, v in biggest_component.edges():
    if len(biggest_component[u][v]) > 1:
      min_travel_time = np.inf
      saved_key = np.inf # will be out of bounds
      for key in biggest_component[u][v]:
        if biggest_component[u][v][key]['travel_time'] < min_travel_time:
          if min_travel_time != np.inf:
            try:
              to_return.remove_edge(u, v, key=saved_key)
            except:
              pass
          min_travel_time = biggest_component[u][v][key]['travel_time']
          saved_key = key
        else:
          try:
            to_return.remove_edge(u, v, key=key)
          except: # might have been deleted already since u->v and v->u are both 
            pass

  print(f'{time.time() - start_time} gedaan')
  
  return to_return 

load_graph('brugge_5km_(51.209348, 3.224700)')