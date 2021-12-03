import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

import matplotlib as mpl
mpl.style.use('tableau-colorblind10')

def plot_degree_dist(name):
  graph = nx.read_gpickle(f'./graph_pickle/{name}.gpickle')
  
  degrees = np.zeros(len(graph.nodes()),dtype=int)
  
  for i, vertex in enumerate(graph.nodes()): # for every node get how many neighbors
    degrees[i] = graph.degree(vertex)
  print(degrees)
  bins = np.array([i for i in range(1, np.max(degrees)+1)]) #
  font = { 'size'   : 16}

  mpl.rc('font', **font)
  x = np.arange(len(bins))
  plt.hist(degrees, bins=bins, align='left')
  plt.xlabel('graad')
  plt.ylabel('aantal')
  plt.xticks(x, bins-1)
  plt.savefig(f'./degree_distribution/{name}_node_distribution.png', bbox_inches='tight')
  plt.clf()

name_list = ['New Dehli','Nairobi', 'Rio de Janeiro', 'Brugge', 'Manhattan']

for name in name_list:
  plot_degree_dist(name)