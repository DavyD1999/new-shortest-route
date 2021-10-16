import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
import gc
import greedy_forwarding_route as gf
import greedy_forwarding_with_edgelength as gfwe
import greedy_then_a_star as gtas
import greedy_rpf as grpf
import fix_graph_data as fgd

"""
generates stretch and arrival percentage histograms for the desired function
"""

def data_generator(name, number_of_routes, functions, foldername): # generates the data for the desired function
  
  graph_basic = fgd.load_graph(name)
  
  node_list = list(graph_basic.nodes())
  list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
  list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

  result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
  reached_end_node =  np.zeros(len(list_indices_start))
  length_path = np.zeros(len(list_indices_start))

  for i in range(number_of_routes):
    
    while list_indices_start[i] == list_indices_end[i]: 
      list_indices_end[i] = np.random.randint(0, len(node_list)) # only change one now since it's a connected graph
      
    # calculate the shortest distance once
    shortest_distance = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'length')
    length_path[i] = shortest_distance
  
  step_size = 1500
  
  values, base = np.histogram(length_path, bins=np.arange(start=0,stop=max(length_path) + step_size, step=step_size)) # + stepsize makes sure we actually get a last bin too

    # values, base = np.histogram(length_path, bins=7) # base gives bin edges and values number in each bin
  base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last value since this one concludes the bin
  indices = np.digitize(length_path, base) - 1

  timing_array = np.zeros(len(functions))

  for x, function in enumerate(functions):
    print(foldername[x])
    result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
    reached_end_node =  np.zeros(len(list_indices_start))
    total_time = 0
    for i in range(number_of_routes): # do the greedy functions
      start_time = time.time()
      result = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_basic) # result like route length of the desired path
      
      if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
        total_time += time.time() - start_time # only track time if succesful
        reached_end_node[i] = 1
        result_stretch[i] = result/length_path[i]


    arrived = np.zeros(len(values))
    average_stretch = np.zeros(len(values))

    for k, element in enumerate(reached_end_node):
      if element: # element is one or zero did it arrive or not
        arrived[indices[k]] += 1 # indices[k] denotes to which bin index this element belongs
    
    for k, element in enumerate(result_stretch):
      if element: # if not a zero (from numpy zeros like)
        average_stretch[indices[k]] += element 

    for j in range(len(average_stretch)):
      if arrived[j] != 0:
        average_stretch[j] /= arrived[j]   # to actually be an average we still need to devide by the number in each bin that arrived  
    # calculate the average time for a succesfull route to arrive

    timing_array[x] = total_time / sum(arrived)

    # let's now calculate the standard deviation on the mean stretch....
    squared_sum = np.zeros(len(average_stretch))
    n = np.zeros(len(average_stretch)) # to keep track how many are in each bin
    
    for k, element in enumerate(result_stretch):
      if element: # if not a zero (from numpy zeros like)
        n[indices[k]] += 1
        squared_sum[indices[k]] += (element - average_stretch[indices[k]]) ** 2 
    
    standard_dev_on_mean = (squared_sum/(n * (n-1))) ** (0.5) # is likely to create invalid values but ignore them, they will not be plotted
    
    average_stretch[average_stretch == 0] = 'nan' # replace exact zeros with nan so they won't get plotted

    arrived_percentage = arrived / values

    # the below plot will tell us which percentage of a route of a certain binned length will arrive
    plt.hist(base[:-1], base + step_size/2, weights=arrived_percentage) # step size/2 makes sure it is centered around the center of the bin
    plt.xlabel('shortest path length (m)')
    plt.ylabel('percentage arrived')
    plt.title(f'{name} arrival probability')
    plt.savefig(f'./{foldername[x]}/{name}_percentage_arrived.png')
    plt.clf() 

    # average stretch per bin 
    plt.errorbar(base[:-1] + step_size/2, average_stretch, yerr=standard_dev_on_mean) # the :-1 because we only plot the middle and end value + half is outside our plotting region
    plt.xlabel('shortest path length (m)')
    plt.ylabel('average stretch')
    plt.ylim(bottom=0)
    plt.title(f'{name} average stretch')
    plt.savefig(f'./{foldername[x]}/{name}_average_stretch.png')
    plt.clf() 
  
  # generate timing plot
  plt.bar(foldernames, timing_array)
  plt.title(f'{name} execution time per succesful path')
  plt.xlabel('method')
  plt.ylabel('execution time per path (s)')
  plt.savefig(f'./speed_comparison/{name}greedy_execution_time_per_path.png')
  
name_list = ['new_dehli_5km_(28.644800, 77.216721)', 'nairobi_5km_(-1.28333, 36.81667)',  'manhattan_5km_(40.754932, -73.984016)', 'rio_de_janeiro_5km_(-22.908333, -43.196388)', 'brugge_5km_(51.209348, 3.224700)']

#functions = [gfwe.greedy_forwarding_with_edge_length, gtas.greedy_forwarding_then_a_star, gf.greedy_forwarding]

#foldernames = ['greedy_with_edge_length','greedy_then_a_star', 'normal_greedy']

functions = [grpf.greedy_forwarding_rpf]
foldernames = ['greedy_rpf']
for name in name_list:
  # 1000 paths for every map
  
  data_generator(name, 500, functions, foldernames)
  print(name)