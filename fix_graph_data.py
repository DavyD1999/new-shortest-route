import osmnx as ox
import networkx as nx
import time
"""
this file will load the graph, make it undirected and select the largest connected component
"""

def load_graph(name):
  start_time = time.time()
  graph = ox.io.load_graphml(f'{name}.graphml')
  print(f'{time.time() - start_time}geladen')
  graph = nx.MultiDiGraph.to_undirected(graph) # makes it undirected

  # now take the largest component
  biggest_component = list(graph.subgraph(c) for c in nx.connected_components(graph))[0]
  print(len(biggest_component))
  print(f'{time.time() - start_time} gedaan')
  return biggest_component  # first one is the biggest