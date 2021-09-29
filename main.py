from greedy_forwarding_route import *
graph_basic = ox.io.load_graphml('kort.graphml')
for node in graph_basic.nodes():
  print(greedy_forwarding(node,1676827988,graph_basic))