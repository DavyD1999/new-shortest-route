import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import greedy_forwarding_route as gf
import greedy_forwarding_with_edgeweight as gfwe
import greedy_then_a_star as gtas
import greedy_rpf as grpf
import fix_graph_data as fgd
import hyperbolic_routing as hr
import hyperbolic_embedder as he
import greedy_manhattan as gm
import node_functions as nf

import matplotlib as mpl

mpl.style.use('tableau-colorblind10')
np.random.seed(42)
font = {'size'   : 16}
mpl.rc('font', **font)

"""
generates stretch and arrival percentage histograms for the desired function
"""

def data_generator(name, functions, foldername, number_of_routes=100, step_size=150): # generates the data for the desired function

  graph = nf.make_d_regular_tree(given_depth=10, degree=3, random_travel_time=True)

  node_list = list(graph.nodes())

  list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
  list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

  result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
  reached_end_node =  np.zeros(len(list_indices_start))
  weight_path = np.zeros(len(list_indices_start))

  for i in range(number_of_routes):

    while list_indices_start[i] == list_indices_end[i]: 
      list_indices_end[i] = np.random.randint(0, len(node_list)) # only change one now since it's a connected graph

    # calculate the shortest distance once
    shortest_distance = nx.shortest_path_length(graph, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'travel_time')
    weight_path[i] = shortest_distance

  values, base = np.histogram(weight_path, bins=np.arange(start=0,stop=max(weight_path) + step_size, step=step_size)) # + stepsize makes sure we actually get a last bin too


  indices = np.digitize(weight_path, base) - 1

  timing_array = np.zeros(len(functions))

  number_of_routes = len(weight_path)
  
  for x, function in enumerate(functions):
    result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
    reached_end_node =  np.zeros(len(list_indices_start))
    ratio_travelled_list = np.zeros(len(list_indices_start))
    total_time = 0

    if function == hr.hyperbolic_greedy_forwarding: # since the a star function needs an extra argument, the min velocity
      
      min_tree = nx.algorithms.tree.mst.minimum_spanning_tree(graph, weight='travel_time')
      node_dict = he.hyperbolic_embed(min_tree)

      for i in range(number_of_routes):  # do the greedy functions
        start_time = time.time()

        result = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]],
                                                min_tree,node_dict,
                                                ratio_travelled=False)  # result like route weight of the desired path
        

        if result != np.inf:  # only calculate how long the path was once one was found with greedy forwarding  else takes so long
            total_time += time.time() - start_time  # only track time if succesful
            reached_end_node[i] = 1
            result_stretch[i] = result / weight_path[i]

    elif function == gtas.greedy_forwarding_then_a_star:
      max_velocity = fgd.get_max_velocity(graph)

      for i in range(number_of_routes): # do the greedy functions
        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph, max_velocity=max_velocity, ratio_travelled=True) # result like route weight of the desired path
        
        ratio_travelled_list[i] = ratio_travelled

        if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
          total_time += time.time() - start_time # only track time if succesful
          reached_end_node[i] = 1
          result_stretch[i] = result/weight_path[i]
    
    else: # for the a star function
      for i in range(number_of_routes): # do the greedy functions
        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph ,ratio_travelled=True) # result like route weight of the desired path
        
        ratio_travelled_list[i] = ratio_travelled

        if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
          total_time += time.time() - start_time # only track time if succesful
          reached_end_node[i] = 1
          result_stretch[i] = result/weight_path[i]

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
                                           # this is quite a weighty way of doing it 

    timing_array[x] = total_time / sum(arrived)

    # let's now calculate the standard deviation on the mean stretch....
    squared_sum = np.zeros(len(average_stretch))
    n = np.zeros(len(average_stretch)) # to keep track how many are in each bin
    
    for k, element in enumerate(result_stretch):
      if element: # if not a zero (from numpy zeros like)
        n[indices[k]] += 1
        squared_sum[indices[k]] += (element - average_stretch[indices[k]]) ** 2 
    
    standard_dev_on_mean_stretch = (squared_sum/(n * (n-1))) ** (0.5) # is likely to create invalid values but ignore them, they will not be plotted

    binned_ratio_travelled = np.zeros(len(values))
    n = np.zeros(len(binned_ratio_travelled))  # to keep track how many are in each bin
    
    for s, rat_trav in enumerate(ratio_travelled_list):
      n[indices[s]] += 1
      binned_ratio_travelled[indices[s]] += rat_trav
    
    average_ratio_travelled = binned_ratio_travelled / n
    squared_sum = np.zeros(len(binned_ratio_travelled))
    for s, rat_trav in enumerate(ratio_travelled_list):
      squared_sum[indices[s]] += (rat_trav - average_ratio_travelled[indices[s]]) ** 2

    standard_dev_on_mean_ratio_travelled = (squared_sum/ (n * (n-1))) ** (0.5)

    average_stretch[average_stretch == 0] = np.nan

    arrived_percentage = arrived / values
    np.nan_to_num(arrived_percentage, copy=False, nan=1.) # sometimes we get 0/0
    # the below plot will tell us which percentage of a route of a certain binned weight will arrive
    plt.hist(base[:-1], base, weights=arrived_percentage) 
    plt.xlabel('snelste reistijd (s)')
    plt.ylabel('aankomst ratio')

    plt.savefig(f'./{foldername[x]}/{name}_percentage_arrived.png', bbox_inches='tight')
    plt.clf() 

    # average stretch per bin 
    print(base[:-1]+step_size/2)
    plt.plot(base[:-1] + step_size/2, average_stretch, linewidth=3) # the :-1 because we only plot the middle and end value + half is outside our plotting region
    # +/2 because we want centered at center of bin
    plt.xlabel('snelste reistijd (s)')
    plt.ylabel('gemiddelde rek')
    plt.ylim(bottom=0)


    plt.savefig(f'./{foldername[x]}/{name}_average_stretch.png', bbox_inches='tight')
    plt.clf() 
    
  
name_list = ['regular tree']

functions = [hr.hyperbolic_greedy_forwarding]

foldernames = ['greedy_hyperbolic']

for name in name_list:
  data_generator(name, functions, foldernames,number_of_routes=300)
  print(name)
