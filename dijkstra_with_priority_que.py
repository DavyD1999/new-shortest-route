import osmnx as ox
import numpy as np
from queue import PriorityQueue
import time
import node_functions as nf

# graph_basic = ox.io.load_graphml('manhattan_5km_(40.754932, -73.984016).graphml')
inf = np.inf

def dijkstra_with_priority_queue(id1, graph):
    """
    function that calculates the distance from node_id1 first argument to all other nodes given graph graph with a priority queue
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
        
        for _ , neighbor_node, edge_length in graph.edges(current_node, data = 'length'): # calculate from every neighbour
          
          new_length = edge_length +  distances[current_node]
          if distances[neighbor_node] > new_length:
            distances[neighbor_node] = new_length
            distance_queue.put((new_length, neighbor_node))        
      
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        
        minimum_distance, current_node = distance_queue.get() # auto removes element 
        while current_node in visited_nodes: # very possible since we add element every time
          minimum_distance, current_node = distance_queue.get() # get new element every time

        if minimum_distance  == inf:
            #print('some roads have no connection')
            break

    return distances

#print(dijkstra_with_priority_queue(9121386338,graph_basic)[1692433918])
# do timing to compare with dijkstra


def dijkstra_with_priority_queue_to_node(id1, id2, graph):
    """
    function that calculates the distance from node_id1 first argument to noide_id2 other nodes given graph
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
    
    while id2 not in visited_nodes: # we will try every node for the distance
        for _ , neighbor_node, edge_length in graph.edges(current_node, data = 'length'): # calculate from every neighbour
          
          new_length = edge_length +  distances[current_node]
          if distances[neighbor_node] > new_length:
            distances[neighbor_node] = new_length
            distance_queue.put((new_length, neighbor_node))           
      
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        
        minimum_distance, current_node = distance_queue.get() # auto removes element 
        while current_node in visited_nodes: # very possible since we add element every time
          minimum_distance, current_node = distance_queue.get() # get new element every time

        if minimum_distance  == inf:
            #print('some roads have no connection')
            break

    return distances[id2]

#print(dijkstra_with_priority_queue_to_node(9121386338,1692433918, graph_basic))