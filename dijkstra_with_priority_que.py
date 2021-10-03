import osmnx as ox
import numpy as np
from queue import PriorityQueue
import time
import dijkstra # import normal dijkstra function to compare
import node_functions as nf
"""
does dijkstra on a certain node to all nodes but with priority queue
"""

graph_basic = ox.io.load_graphml('zw6_5km.graphml') # change file name here
inf = np.inf

def dijkstra_with_priority_queue(id1, graph):
    """
    function that calculates the distance from node_id1 first argument to all other nodes given graph graph
    """

    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"
    
    distance_queue = PriorityQueue()
    distances = dict()
    
    for node_id in unvisited_nodes:
      if node_id != id1:
        distance_queue.put((inf, node_id))# every element infinite except for the starnode
        distances[node_id] = inf
      else:
        distance_queue.put((0, id1))
        distances[id1] = 0 # overwrite distance for our start node


    visited_nodes = set()
    current_node = id1
    
    for _ in range(len(graph.nodes())): # we will try every node for the distance
        for neighbor_node in graph.neighbors(current_node): # calculate from every neighbour
          new_length = (nf.get_edge_length(current_node, neighbor_node, graph) +  distances[current_node])
          if distances[neighbor_node] > new_length:
            distances[neighbor_node] = new_length
            distance_queue.put((new_length,neighbor_node))        
      
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        
        minimum_distance, current_node = distance_queue.get() # auto removes element 
        while current_node in visited_nodes: # very possible since we add element every time
          minimum_distance, current_node = distance_queue.get() # get new element every time


        if minimum_distance  == inf:
            #print('some roads have no connection')
            break

    return distances

# do timing to compare with dijkstra
start_time = time.time()

for _ in range(100):
  dijkstra_with_priority_queue(8790497014, graph_basic) # 8,2

print(f'{time.time()-start_time} with priority queue')

start_time = time.time()
for _ in range(100):
  dijkstra.dijkstra(8790497014, graph_basic) #15,1s for execution time

print(f'{time.time()-start_time} without priority queue')

# check if the right result is there

assert dijkstra.dijkstra(8790497014, graph_basic) == dijkstra_with_priority_queue(8790497014, graph_basic), "solutions don't match"