import dijkstra
import dijkstra_with_priority_que as dwpq
import networkx as nx
import numpy as np
import a_star
import stratified_sampling as ss

import time
import matplotlib.pyplot as plt
from fix_graph_data import get_max_velocity
import matplotlib as mpl

mpl.style.use('tableau-colorblind10')
np.random.seed(42)

"""
compares the speed of dwpq dijkstra dijkstra of networkx and a star and asserts that they give back the right distance
"""

def speed_comparator(name):
  graph_basic = nx.read_gpickle(f'./graph_pickle/{name}.gpickle')
  
  max_velocity = get_max_velocity(graph_basic) # gets the max velocity of all edges, useful for A*
  node_list = list(graph_basic.nodes())
  
  _ , list_indices_start, list_indices_end = ss.stratified_sampling(50, 50, 150, node_list, graph_basic)

  dijkstra_networkx_speed = 0
  dijkstra_speed = 0
  dwpq_speed =  0
  a_star_speed = 0
  a_star_wpq_speed = 0
  
  for i in range(len(list_indices_start)):
  
    start = time.time()
    a = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], weight='travel_time', method='dijkstra') # happens with dijkstra
    dijkstra_networkx_speed += time.time() - start
    
    start = time.time()
    b = dijkstra.dijkstra_to_node(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)
    dijkstra_speed += time.time() - start
    
    start = time.time()
    c = dwpq.dijkstra_with_priority_queue_to_node(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)
    dwpq_speed += time.time() - start
    
    start = time.time()
    d = a_star.A_star(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic, max_velocity)
    a_star_speed += time.time() - start
    
    start = time.time()
    e = a_star.A_star_priority_queue(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic, max_velocity)
    a_star_wpq_speed += time.time() - start

    assert abs(b-c) < 10**-2 and abs(d-c) < 10**-2 and abs(a-c) < 10**-2 and abs(a-e) < 10**-2, f'{c} dwpq and {b} normal dijkstra {d} astar {a} networkx {e} astar wpq'
    

  plt.bar(['Networkx dijkstra', 'Dijkstra', 'Dijkstra w/ pq', 'A*', 'A* w/ pq'],np.array([dijkstra_networkx_speed, dijkstra_speed, dwpq_speed, a_star_speed, a_star_wpq_speed])/number_of_routes)
  plt.title(f'{name} execution time per path')
  plt.xlabel('method')
  plt.ylabel('execution time per path (s)')
  plt.savefig(f'./speed_comparison/{name}_execution_time_per_path.png')

speed_comparator('Brugge')