import pickle
import fix_graph_data as fgd
import networkx as nx
"""
loading the graphml file takes a lot of time and also the simplifying takes too long so we will store it as a pickle object
"""
name_list = ['new_dehli_5km_(28.644800, 77.216721)', 'nairobi_5km_(-1.28333, 36.81667)',  'manhattan_5km_(40.754932, -73.984016)', 'rio_de_janeiro_5km_(-22.908333, -43.196388)', 'brugge_5km_(51.209348, 3.224700)']

new_name = ['New Dehli', 'Nairobi', 'Manhattan', 'Rio de Janeiro', 'Brugge']

for i, name in enumerate(name_list):
    graph = fgd.load_graph(f'./graph_graphml/{name}')
    # Store data (serialize)
    nx.write_gpickle(graph, f'./graph_pickle/{new_name[i]}.gpickle')
    
        