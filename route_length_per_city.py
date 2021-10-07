import numpy as np
import coordinate_functions as cf
import node_functions as nf
import osmnx as ox
import a_star
import matplotlib.pyplot as plt
import networkx as nx
import gc

name_list = ['new_dehli_5km_(28.644800, 77.216721)', 'nairobi_5km_(-1.28333, 36.81667)',  'manhattan_5km_(40.754932, -73.984016)', 'rio_de_janeiro_5km_(-22.908333, -43.196388)', 'brugge_5km_(51.209348, 3.224700)']


def get_route_length(name, number_of_routes): # generates the data for the desired function
  graph_basic = ox.io.load_graphml(f'{name}.graphml')
  
  node_list = list(graph_basic.nodes())
  list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
  list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

 
  length_path = np.zeros(len(list_indices_start))
  
  for i in range(number_of_routes):
    shortest_distance = 0
    while shortest_distance == 0:
      try:
        if (list_indices_start[i] != list_indices_end[i]):
          shortest_distance = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'length')
        else:
          list_indices_end[i] = np.random.randint(0, len(node_list))
          list_indices_start[i] = np.random.randint(0, len(node_list)) # change both since dead end might be caused by one of both
      except nx.exception.NetworkXNoPath: # geen pad gevonden
        list_indices_end[i] = np.random.randint(0, len(node_list))
        list_indices_start[i] = np.random.randint(0, len(node_list))


    length_path[i] = shortest_distance

  gc.collect() # repl hates it if you use "too much memory"

  values, base = np.histogram(length_path, bins=7) # base gives bin edges and values number in each bin
  base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last 

  # how many routes are in each bin
  plt.hist(length_path, base)
  plt.xlabel('shortest path length (m)')
  plt.ylabel('number of shortest paths')
  plt.title(f'{name} distribution of route length')
  plt.savefig(f'./route_length/{name}_route_length_distribution.png')
  plt.clf() 

for name in name_list:
  get_route_length(name, 10**3) # 1000 paths for every map

  


