import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import gc
import greedy_forwarding_route as gf
import greedy_forwarding_with_edgelength as gfwe
import greedy_then_a_star as gtas
import fix_graph_data as fgd
"""
generates stretch and arrival percentage histograms for the desired function
"""

def data_generator(name, number_of_routes, functions, foldername): # generates the data for the desired function
  
  graph_basic = fgd.load_graph(name)
  for x, function in enumerate(functions):
      node_list = list(graph_basic.nodes())
      list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
      list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

      result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
      reached_end_node =  np.zeros(len(list_indices_start))
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

        result = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic)

        length_path[i] = shortest_distance
        if result != np.inf:
          reached_end_node[i] = 1
          result_stretch[i] = result/shortest_distance


      # gc.collect() # repl hates it if you use "too much memory"

      values, base = np.histogram(length_path, bins=7) # base gives bin edges and values number in each bin
      base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last value since this one concludes the bin
      indices = np.digitize(length_path, base) - 1 # gives the bin index of all paths so says in which bin path i lays decrease with one to make it work
      
      arrived = np.zeros(len(values))
      average_stretch = np.zeros(len(values))

      for k, element in enumerate(reached_end_node):
        if element: # element is one or zero did it arrive or not
          arrived[indices[k]] += 1
      
      for k, element in enumerate(result_stretch):
        if element: # if not a zero (from numpy zeros like)
          average_stretch[indices[k]] += element 
      
      for j in range(len(average_stretch)):
        if arrived[j] != 0:
          average_stretch[j] /= arrived[j]   # to actually be an average we still need to devide by the number in each bin that arrived  
          
      arrived_percentage = arrived / values

      # the below plot will tell us which percentage of a route of a certain binned length will arrive
      plt.hist(base[:-1], base, weights=arrived_percentage)
      plt.xlabel('shortest path length (m)')
      plt.ylabel('percentage arrived')
      plt.title(f'{name} percentage arrived')
      plt.savefig(f'./{foldername[x]}/{name}_percentage_arrived.png')
      plt.clf() 

      # average stretch per bin 
      plt.hist(base[:-1], base, weights=average_stretch)
      plt.xlabel('shortest path length (m)')
      plt.ylabel('average stretch')
      plt.title(f'{name} average stretch')
      plt.savefig(f'./{foldername[x]}/{name}_average_stretch.png')
      plt.clf() 

name_list = ['new_dehli_5km_(28.644800, 77.216721)', 'nairobi_5km_(-1.28333, 36.81667)',  'manhattan_5km_(40.754932, -73.984016)', 'rio_de_janeiro_5km_(-22.908333, -43.196388)', 'brugge_5km_(51.209348, 3.224700)']

functions = [gfwe.greedy_forwarding_with_edge_length, gtas.greedy_forwarding_then_a_star, gf.greedy_forwarding]

foldernames = ['greedy_with_edge_length', 'greedy_then_a_star', 'normal_greedy']
for name in name_list:
  # 1000 paths for every map
  
  data_generator(name, 1000, functions, foldernames)
  print(name)