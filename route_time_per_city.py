import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

mpl.style.use('bmh')
np.random.seed(42)

name_list = ['New Dehli', 'Nairobi', 'Manhattan', 'Rio de Janeiro', 'Brugge']

def get_route_weight(name, number_of_routes): # generates the data for the desired function
  graph_basic =  nx.read_gpickle(f'./graph_pickle/{name}.gpickle')
  
  node_list = list(graph_basic.nodes())
  list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes) # first generate random numbers this is quicker
  list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes)

  weight_path = np.zeros(len(list_indices_start))
  
  for i in range(number_of_routes):
    
    while list_indices_start[i] == list_indices_end[i]: 
      list_indices_end[i] = np.random.randint(0, len(node_list)) # only change one now since it's a connected graph
      
    # calculate the shortest time once
    shortest_time = nx.shortest_path_length(graph_basic, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'travel_time')
    weight_path[i] = shortest_time

  values, base = np.histogram(weight_path, bins=7) # base gives bin edges and values number in each bin
  base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last 

  # how many routes are in each bin
  plt.hist(weight_path, base)
  plt.xlabel('fastest path (s)')
  plt.ylabel('number of fastest paths')
  plt.title(f'{name} distribution of fastest paths')
  plt.savefig(f'./route_weight/{name}_route_weight_distribution.png')
  plt.clf()
  print('done') 

for name in name_list:
  get_route_weight(name, 2000) 

  