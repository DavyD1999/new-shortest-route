from fix_graph_data import load_graph
import osmnx as ox
import networkx as nx
import matplotlib

matplotlib.use('Agg')
graph_basic = load_graph('rio_de_janeiro_5km_(-22.908333, -43.196388)')

rc = ['r', 'y', 'c']
  
node_list = list(graph_basic.nodes())
start = node_list[1] # first generate random numbers this is quicker
end = node_list[-3]

    # calculate the shortest distance once
route = nx.shortest_path(graph_basic, start, end, 'length')
print(route)

fig, _ = ox.plot_graph_route(graph_basic, route, route_color='y', route_linewidth=2, node_size=0)

start = node_list[95] # first generate random numbers this is quicker
end = node_list[20]
route = nx.shortest_path(graph_basic, start, end, 'length')
fig.savefig('testroute.png')
fig.clf()
fig, _ = ox.plot_graph_route(graph_basic, route, route_color='y', route_linewidth=2, node_size=0)

fig.savefig('testroute2.png')
fig.clf()