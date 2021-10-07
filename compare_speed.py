import dijkstra
import dijkstra_with_priority_que as dwpq
import networkx as nx
import numpy as np
import a_star
import osmnx as ox
import time
import matplotlib.pyplot as plt

def speed_comparator(name, number_of_routes): # generates the data for greedy forwarding
  graph_basic = ox.io.load_graphml(f'{name}.graphml')
  
  node_list = list(graph_basic.nodes())
  
  list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
  list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

  dijkstra_networkx_speed = 0
  dijkstra_speed = 0
  dwpq_speed =  0
  a_star_speed = 0 
  
  for i in range(number_of_routes): 
    test_length = 0. 
    while test_length == 0:
      try:
        if (list_indices_start[i] != list_indices_end[i]):
          # just confirm there is a path
          test_length = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'length')
        
        else:
          list_indices_end[i] = np.random.randint(0, len(node_list))
          list_indices_start[i] = np.random.randint(0, len(node_list)) # change both since dead end might be caused by one of both
      
      except nx.exception.NetworkXNoPath: # geen pad gevonden
        list_indices_end[i] = np.random.randint(0, len(node_list))
        list_indices_start[i] = np.random.randint(0, len(node_list))

  
    start = time.time()
    a = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], weight='length', method='dijkstra') # happens with dijkstra
    dijkstra_networkx_speed += time.time() - start
    
    start = time.time()
    b = dijkstra.dijkstra_to_node(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)
    dijkstra_speed += time.time() - start
    
    start = time.time()
    c = dwpq.dijkstra_with_priority_queue_to_node(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)
    dwpq_speed += time.time() - start
    
    start = time.time()
    d = a_star.A_star(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)
    a_star_speed += time.time() - start

    assert abs(c - b) < 10**-2 and abs(c - d) < 10**-2, 'path lengths are not the same' # networkx treats one way streets as one way
  

  
  plt.bar(['networkx dijkstra', 'dijkstra', 'dwpq', 'A*'],np.array([dijkstra_networkx_speed, dijkstra_speed, dwpq_speed, a_star_speed])/number_of_routes)
  plt.title(f'{name} execution time per path')
  plt.xlabel('method')
  plt.ylabel('execution time per path')
  plt.savefig(f'{name}_execution_time_per_path.png')
speed_comparator('brugge_5km_(51.209348, 3.224700)', 10**2)