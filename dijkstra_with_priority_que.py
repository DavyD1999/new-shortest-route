import osmnx as ox
import numpy as np
from queue import PriorityQueue
import time
import node_functions as nf

# graph_basic = ox.io.load_graphml('manhattan_5km_(40.754932, -73.984016).graphml')
inf = np.inf

def dijkstra_with_priority_queue(id1, graph):
    """
    function that calculates the time from node_id1 first argument to all other nodes given graph graph with a priority queue
    """

    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"
    
    time_queue = PriorityQueue()
    times = dict()
    
    for node_id in unvisited_nodes:
      if node_id != id1:
        time_queue.put((inf, node_id))# every element infinite except for the starnode
        times[node_id] = inf
      else:
        time_queue.put((0, id1))
        times[id1] = 0 # overwrite time for our start node


    visited_nodes = set()
    current_node = id1
    
    for _ in range(len(graph.nodes())): # we will try every node for the time
        
        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every neighbour
          
          new_weight = edge_weight +  times[current_node]
          if times[neighbor_node] > new_weight:
            times[neighbor_node] = new_weight
            time_queue.put((new_weight, neighbor_node))        
      
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        
        minimum_time, current_node = time_queue.get() # auto removes element 
        while current_node in visited_nodes: # very possible since we add element every time
          minimum_time, current_node = time_queue.get() # get new element every time

        if minimum_time  == inf:
            #print('some roads have no connection')
            break

    return times

#print(dijkstra_with_priority_queue(9121386338,graph_basic)[1692433918])
# do timing to compare with dijkstra


def dijkstra_with_priority_queue_to_node(id1, id2, graph, return_visited=False):
    """
    function that calculates the time from node_id1 first argument to noide_id2 other nodes given graph
    """

    unvisited_nodes = set(graph.nodes())
    assert id1 in unvisited_nodes, "node_id is not in the graph"
    
    time_queue = PriorityQueue()
    times = dict()
    came_from = dict()
    
    for node_id in unvisited_nodes:
      if node_id != id1:
        time_queue.put((inf, node_id))# every element infinite except for the starnode
        times[node_id] = inf
      else:
        time_queue.put((0, id1))
        times[id1] = 0 # overwrite time for our start node


    visited_nodes = set()
    current_node = id1
    
    while id2 not in visited_nodes: # we will try every node for the time
        for _ , neighbor_node, edge_weight in graph.edges(current_node, data = 'travel_time'): # calculate from every neighbour
          
          new_weight = edge_weight +  times[current_node]
          if times[neighbor_node] > new_weight:
            times[neighbor_node] = new_weight
            came_from[neighbor_node] = current_node
            time_queue.put((new_weight, neighbor_node))           
      
        visited_nodes.add(current_node)
        unvisited_nodes.remove(current_node)
        
        minimum_time, current_node = time_queue.get() # auto removes element 
        while current_node in visited_nodes: # very possible since we add element every time
          minimum_time, current_node = time_queue.get() # get new element every time

        if minimum_time  == inf:
            #print('some roads have no connection')
            break

    if return_visited is True:
        return visited_nodes, came_from
    return times[id2]

#print(dijkstra_with_priority_queue_to_node(9121386338,1692433918, graph_basic))