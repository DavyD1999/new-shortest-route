import numpy as np
import coordinate_functions as cf
import osmnx as ox
import a_star

graph_basic = ox.io.load_graphml('kort.graphml')

def greedy_forwarding(id1, id2, graph): # id1 is start node id2 is go to node
  inf = np.inf
  start_distance = cf.distance(id1, id2, graph)
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  distance_travelled = 0

  while (current_node != id2):
    min_distance = inf
    
    for neighbor_node in graph.neighbors(current_node):
      # id2 is our go to node
      try: # it might be a one way street
        edge_val = graph[neighbor_node][current_node].values()
      except KeyError:
        try:
          edge_val = graph[current_node][neighbor_node].values()
        except KeyError:
          pass # no actual path seems to exist 
      edge_length = inf
      for edge in edge_val: # if two roads or more roads do connect one chose the shortest one of both
        if edge_length > edge.get('length'):
          edge_length = edge.get('length')
        
      if cf.distance(current_node, neighbor_node, graph) + edge_length < min_distance:
        node_with_min_distance = neighbor_node
        min_distance = cf.distance(current_node, neighbor_node, graph) + edge_length
        
        min_edge_length = edge_length # also save the length of the path
      distance_travelled += min_edge_length
      current_node = node_with_min_distance
      if distance_travelled > 20*start_distance or current_node in visited:
        
        return inf;
      visited.add(current_node) 
  
  return distance_travelled
i = 0
for start_node in graph_basic.nodes():
  for node in graph_basic.nodes():
    if start_node != node:
      if greedy_forwarding(node,start_node,graph_basic) != np.inf:
        print("verhouding:")
        print(greedy_forwarding(node,start_node,graph_basic)/a_star.A_star(node, start_node, graph_basic))
        print("a_star distance:")
        print(a_star.A_star(node, start_node, graph_basic))
        i += 1 
print(i) # 48 for kort graph with edge lengths
