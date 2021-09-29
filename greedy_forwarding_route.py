import numpy as np
import coordinate_functions as cf
import osmnx as ox
import a_star

graph_basic = ox.io.load_graphml('rio_de_janeiro_5km_(-22.908333, -43.196388).graphml')

def greedy_forwarding(id1, id2, graph): # id1 is start node id2 is go to node
  inf = np.inf
  total_nodes = graph.nodes()

  assert id1 in total_nodes and id2 in total_nodes , "node_id is not in the graph"

  visited = set()
  current_node = id1
  distance_travelled = 0

  while (current_node != id2):
    min_distance = inf
    
    for neighbor_node in graph.neighbors(current_node):
        
      if cf.euclid_distance(current_node, neighbor_node, graph) < min_distance:
        node_with_min_distance = neighbor_node
        min_distance = cf.euclid_distance(current_node, neighbor_node, graph)

    if min_distance == inf:
      return inf 
    if node_with_min_distance in visited: # can't be together with the above if else could be referenced before assignment
      return inf
    try: # it might be a one way street
      edge_val = graph[node_with_min_distance][current_node].values()
    except KeyError:
      try:
        edge_val = graph[current_node][node_with_min_distance].values()
      except KeyError:
        pass # no actual path seems to exist 
    edge_length = inf
    for edge in edge_val: # if two roads or more roads do connect one chose the shortest one of both
      if edge_length > edge.get('length'):
        edge_length = edge.get('length')
      
    distance_travelled += edge_length
    current_node = node_with_min_distance

    visited.add(current_node) 
  
  return distance_travelled


node_list = list(graph_basic.nodes())
list_indices_start = np.random.randint(0, len(node_list), size=100000) # first generate random numbers this is quicker
list_indices_end = np.random.randint(0, len(node_list), size=100000)

result_stretch = np.zeros_like(list_indices_start)
reached_end_node =  np.zeros_like(list_indices_start)

succes_stories = 0

for i in range(len(list_indices_start)):
  if list_indices_start[i] != list_indices_end[i]:
    result = greedy_forwarding(node_list[list_indices_start[i]], node_list[list_indices_end[i]],graph_basic)
    if result != np.inf:
      succes_stories += 1
      print(result/a_star.A_star(node_list[list_indices_start[i]], node_list[list_indices_end[i]],graph_basic))

        
print(succes_stories)