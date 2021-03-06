import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import greedy_forwarding_route as gf
import greedy_forwarding_with_edgeweight as gfwe # not really used in thesis
import greedy_then_a_star as gtas
import greedy_rpf as grpf
import fix_graph_data as fgd
import hyperbolic_routing as hr
import hyperbolic_embedder as he
import greedy_manhattan as gm
import node_functions as nf
import coordinate_functions as cf
import gravity_pressure as gp
import a_star
import NN_gravity_pressure as NNgp
from gensim.models import KeyedVectors
import pickle


import stratified_sampling as ss
import matplotlib as mpl



mpl.style.use('tableau-colorblind10')
np.random.seed(42)
font = {'size'   : 16}
mpl.rc('font', **font)

"""
generates stretch and arrival percentage histograms for the desired function
"""
def add_coordinates(graph, dictionary):
    fig, ax = plt.subplots()

    nx.draw_networkx_nodes(graph, dictionary, node_size=1, ax=ax)  # new layout specifies positions

        # labels = nx.get_edge_attributes(graph, 'travel_time')
    nx.draw_networkx_edges(graph, pos=dictionary, width=0.5, node_size=0.1)
    #nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.savefig('./semester2/convex.png', dpi=500)
    ax.clear()
    plt.clf()
    for node in graph.nodes():
        graph.add_node(node, coordinates=np.array(dictionary[node]))

    return graph

def data_generator(name, functions, foldername, number_of_routes_pre_compute=80, step_size=150, amount_of_samples_per_bin=50): # generates the data for the desired function

  graph = nx.read_gpickle(f'./graph_pickle/{name}.gpickle')

  node_list = list(graph.nodes())

  weight_path, list_indices_start, list_indices_end = ss.stratified_sampling(amount_of_samples_per_bin, number_of_routes_pre_compute, step_size, node_list, graph)

  result_stretch = np.zeros(len(list_indices_start)) # DO NOT USE LIKE CAUSE IT WOULD CONVERT THEM TO INTS
  reached_end_node =  np.zeros(len(list_indices_start))  
  
  values, base = np.histogram(weight_path, bins=np.arange(start=0,stop=max(weight_path) + step_size, step=step_size)) # + stepsize makes sure we actually get a last bin too
  
    # values, base = np.histogram(weight_path, bins=7) # base gives bin edges and values number in each bin
  base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last value since this one concludes the bin
  plt.hist(weight_path, bins=base)
  plt.xlabel('travel time (s)')
  plt.ylabel('number of paths')
  plt.title(f'{name} weight of paths')
  plt.savefig('./stratified_sampling_histogram.png')
 
  plt.clf()	

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
      
      #he.plot_hyperbolic_graph(graph, node_dict, name) # plot the actual graph in hyperbolic space
      
      for i in range(number_of_routes):  # do the greedy functions
        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]],
                                                graph,node_dict,
                                                ratio_travelled=True)  # result like route weight of the desired path
        ratio_travelled_list[i] = ratio_travelled

        if result != np.inf:  # only calculate how long the path was once one was found with greedy forwarding  else takes so long
            total_time += time.time() - start_time  # only track time if succesful
            reached_end_node[i] = 1
            result_stretch[i] = result / weight_path[i]

                
    elif function == gtas.greedy_forwarding_then_a_star: #or function == a_star.A_star_priority_queue:

      average_velocity = fgd.get_weigted_average_velocity(graph)

      for i in range(number_of_routes): # do the greedy functions
        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph, velocity=average_velocity, ratio_travelled=True) # result like route weight of the desired path
        #result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph, velocity=average_velocity),1 # pure a star always finds a way

        ratio_travelled_list[i] = ratio_travelled

        if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
          total_time += time.time() - start_time # only track time if succesful
          reached_end_node[i] = 1
          result_stretch[i] = result/weight_path[i]

    elif function == a_star.A_star_priority_queue:
        average_velocity = fgd.get_weigted_average_velocity(graph)
        tot_teller = 0
        for i in range(number_of_routes): # do the greedy functions
            start_time = time.time()
            
            
            result, teller = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph, velocity=average_velocity,  return_counter=True) # pure a star always finds a way
            tot_teller += teller
            ratio_travelled_list[i] = 1

            if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
                total_time += time.time() - start_time # only track time if succesful
                reached_end_node[i] = 1
                result_stretch[i] = result/weight_path[i]
        print(tot_teller)        
        
          
        
    elif foldername[x] == 'semester1/greedy_spring':
      print(graph.number_of_edges())
      print(graph.number_of_nodes())
      
      with open(f'./semester2/convex/nxspring.pickle', 'rb') as handle:
        coor_dict = pickle.load(handle)
      graph_spring = add_coordinates(graph, coor_dict)


      for i in range(number_of_routes): # do the greedy functions
        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph_spring, distance_function=cf.euclidian_n_dimensions ,ratio_travelled=True) # result like route weight of the desired path
        
        ratio_travelled_list[i] = ratio_travelled

        if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
          total_time += time.time() - start_time # only track time if succesful
          reached_end_node[i] = 1
          result_stretch[i] = result/weight_path[i]
        
    elif function == NNgp.gravity_pressure_embedding:
        embedding = KeyedVectors.load(f'./node2vec_models/{name}.wordvectors', mmap='r')
        
        for i in range(number_of_routes): # do the greedy functions

            start_time = time.time()
    
            result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph, embedding,ratio_travelled=True) # result like route weight of the desired path
            
            ratio_travelled_list[i] = ratio_travelled

            if result != np.inf: # only calculate how long the path was once one was found with greedy forwarding  else takes so long   
              total_time += time.time() - start_time # only track time if succesful
              reached_end_node[i] = 1
              result_stretch[i] = result/weight_path[i]

    else: 
   
      for i in range(number_of_routes): # do the greedy functions

        start_time = time.time()

        result, ratio_travelled = function(node_list[list_indices_start[i]], node_list[list_indices_end[i]], graph,ratio_travelled=True, plot_stuck=False) # result like route weight of the desired path
        
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
    print(f'timing array zegt{timing_array}')
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

    average_stretch[average_stretch == 0] = np.nan # replace exact zeros with nan so they won't get plotted

    arrived_percentage = arrived / values

    font = { 'size'   : 16}

    mpl.rc('font', **font)

    # the below plot will tell us which percentage of a route of a certain binned weight will arrive
    plt.hist(base[:-1], base, weights=arrived_percentage) 
    plt.xlabel('snelste reistijd (s)')
    plt.ylabel('aankomst ratio')
    #plt.title(f'{name} arrival ratio')
    plt.savefig(f'./{foldername[x]}/{name}_percentage_arrived.png', bbox_inches='tight')
    plt.clf() 

    # average stretch per bin 
    plt.errorbar(base[:-1] + step_size/2, average_stretch, yerr=standard_dev_on_mean_stretch, linewidth=3,ecolor='tab:purple', elinewidth=3, linestyle='--', capsize=5,barsabove=True) # the :-1 because we only plot the middle and end value + half is outside our plotting region
    # +/2 because we want centered at center of bin
    plt.xlabel('snelste reistijd (s)')
    plt.ylabel('gemiddelde rek')
    plt.ylim(bottom=0)

    #plt.title(f'{name} average stretch')
    plt.savefig(f'./{foldername[x]}/{name}_average_stretch.png',bbox_inches='tight')
    plt.clf() 
    
    plt.errorbar(base[:-1] + step_size/2, average_ratio_travelled, yerr=standard_dev_on_mean_ratio_travelled, linewidth=3,ecolor='tab:purple', elinewidth=3, linestyle='--', capsize=5) # the :-1 because we only plot the middle and end value + half is outside our plotting region
    # +/2 because we want centered at center of bin
    plt.xlabel('snelste reistijd')
    plt.ylabel('ratio afgelegd')
    plt.ylim(bottom=0)
    #plt.title(f'{name} ratio travelled')
    plt.savefig(f'./{foldername[x]}/{name}_ratio_travelled.png',bbox_inches='tight')
    plt.clf() 
  
  # generate timing plot

  plt.barh(foldernames, timing_array)
  x = np.arange(len(foldernames))
  #plt.yticks(x, ['hemelsbrede afstand','H+B', 'manhattan', 'RPF','gemodificeerde GSpring', 'gravity-pressure', 'niet-toelaatbare A*'], fontsize='13', rotation=0)
  plt.yticks(x, ['gretig hyperbolisch', 'gravity-pressure', 'niet-toelaatbare A*'], fontsize='13', rotation=0)
  #plt.title(f'{name} execution time per succesful path')
  plt.xlabel('gemiddelde uitvoeringstijd per pad (s)')
  plt.xscale('log')
  plt.xlim([0.9*10**-3,10**-1])
  plt.savefig(f'./speed_comparison/{name} greedy_execution_time_per_path.png', bbox_inches='tight')
  plt.clf()

  
  
name_list = ['New Dehli','Brugge','Nairobi', 'Rio de Janeiro', 'Manhattan']

# name_list = ['Manhattan','New Dehli','Nairobi', 'Rio de Janeiro','Brugge']

# name_list = ['Manhattan']

# functions = [gp.priority_queue_new_evaluation_function,a_star.A_star_priority_queue ,gp.gravity_pressure, gf.greedy_forwarding, gtas.greedy_forwarding_then_a_star, grpf.greedy_forwarding_rpf, gm.manhattan_greedy_forwarding, hr.hyperbolic_greedy_forwarding] #,hr.hyperbolic_greedy_forwarding gf.greedy_forwarding ,gfwe.greedy_forwarding_with_edge_weight, gtas.greedy_forwarding_then_a_star,  grpf.greedy_forwarding_rpf, gm.manhattan_greedy_forwarding,gp.gravity_pressure, gp.gravity_pressure]# a_star.A_star_priority_queue, NNgp.gravity_pressure_embedding, 

#functions = [hr.hyperbolic_greedy_forwarding]
#functions = [a_star.A_star_priority_queue]
name_list = ['Brugge']

#functions = [gf.greedy_forwarding, gfwe.greedy_forwarding_with_edge_weight, gm.manhattan_greedy_forwarding, grpf.greedy_forwarding_rpf,gf.greedy_forwarding,gp.gravity_pressure, a_star.A_star_priority_queue]
functions = [hr.hyperbolic_greedy_forwarding,gp.gravity_pressure, a_star.A_star_priority_queue]
#foldernames = ['semester1/normal_greedy','semester1/greedy_with_edge_weight','semester1/greedy_manhattan','semester1/greedy_rpf','semester1/greedy_spring','semester1/gravity_pressure', 'semester1/greedy_then_a_star']
# foldernames = ['semester2/weighted_expansion', 'irrelevanat/', 'semester1/gravity_pressure', 'semester1/normal_greedy', 'semester1/greedy_then_a_star', 'semester1/greedy_rpf','semester1/greedy_manhattan','semester1/greedy_hyperbolic',] #, 'normal_greedy','greedy_with_edge_weight','greedy_then_a_star', 'greedy_rpf', 'greedy_manhattan', 'greedy_hyperbolic','gravity_pressure', 'greedy_hyperbolic']
#'pure_A_star'
#foldernames = ['pure_A_star']
#foldernames = ['greedy_hyperbolic']
foldernames = ['semester1/greedy_hyperbolic','semester1/gravity_pressure', 'semester1/greedy_then_a_star']

for name in name_list:
  data_generator(name, functions, foldernames,number_of_routes_pre_compute=50, step_size=150, amount_of_samples_per_bin=50)
  print(name)
