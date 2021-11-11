import matplotlib.pyplot as plt
import numpy as np
import fix_graph_data as fgd

import matplotlib as mpl
mpl.style.use('tableau-colorblind10')

def plot_degree_dist(name):
  graph = fgd.load_graph(name)
  
  degrees = np.zeros(len(graph.nodes()),dtype=int)
  
  for i, vertex in enumerate(graph.nodes()): # for every node get how many neighbors
    degrees[i] = graph.degree(vertex)
  bins = [i for i in range(1, np.max(degrees)+2)] #

  plt.hist(degrees, bins=bins)
  plt.xlabel('degree')
  plt.ylabel('number of nodes')
  plt.title(f'{name} node distribution')
  plt.savefig(f'./degree_distribution/{name}_node_distribution.png')
  plt.clf()
name_list = ['new_dehli_5km_(28.644800, 77.216721)', 'nairobi_5km_(-1.28333, 36.81667)',  'manhattan_5km_(40.754932, -73.984016)', 'rio_de_janeiro_5km_(-22.908333, -43.196388)', 'brugge_5km_(51.209348, 3.224700)']

for name in name_list:
  plot_degree_dist(name)