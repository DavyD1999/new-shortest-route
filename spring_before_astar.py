import networkx as nx
import matplotlib.pyplot as plt

graph_name = 'Brugge'
graph = nx.read_gpickle(f'./graph_pickle/{graph_name}.gpickle')

coordinate_dictionary = dict()

for node in graph.nodes():
    x, y = graph.nodes[node]['x'], graph.nodes[node]['y']

    coordinate_dictionary[node] = [x, y]

new_layout = nx.spring_layout(graph, pos=coordinate_dictionary, weight='travel_time', seed=42)


nx.draw_networkx_nodes(graph, new_layout, node_size=10) # new layout specifies positions
labels = nx.get_edge_attributes(graph,'travel_time')
nx.draw_networkx_edges(graph, pos=new_layout, edgelist=None, width=0.5, node_size=10)
nx.draw_networkx_edge_labels(graph, pos=new_layout, edge_labels=labels, label_pos=0.5, font_size=3)
plt.savefig('./semester2/springlayout/Brugge_gespringed.png', dpi=500)
