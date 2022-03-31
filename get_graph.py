import osmnx as ox
"""
get graph with this file
"""
"""
address, distance, output_name = ' '.join(sys.argv[1:]).split(',')
address = address.strip()
distance = float(distance.strip())
output_name = output_name.strip()

graph_basic = ox.graph.graph_from_address(address, dist=distance, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True, return_coords=False, clean_periphery=True, custom_filter=None)

ox.io.save_graphml(graph_basic, f'{output_name}.graphml')
"""

# (50.8503396, 4.3517103) brussel
graph_basic =  ox.graph.graph_from_point((50.8503396, 4.3517103), dist=20000, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True, clean_periphery=True, custom_filter=None)

print(len(graph_basic.nodes()))
ox.io.save_graphml(graph_basic, f'big_graph.graphml')
