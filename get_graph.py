import osmnx as ox
import sys
"""
to get a graph from an address use comma seperated values first the address then the distance from the address followed by the name of the output file without .graphml this will be added automatically
e.g.  python3.8 get_graph.py kortrijksesteenweg 121 gent, 1000, test will give the graph around the address 1km around the address and outputfile test.graphml
"""
"""
address, distance, output_name = ' '.join(sys.argv[1:]).split(',')
address = address.strip()
distance = float(distance.strip())
output_name = output_name.strip()

graph_basic = ox.graph.graph_from_address(address, dist=distance, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True, return_coords=False, clean_periphery=True, custom_filter=None)

ox.io.save_graphml(graph_basic, f'{output_name}.graphml')
"""
graph_basic =  ox.graph.graph_from_point((51.209348, 3.224700), dist=5000, dist_type='bbox', network_type='drive', simplify=True, retain_all=False, truncate_by_edge=True, clean_periphery=True, custom_filter=None)



ox.io.save_graphml(graph_basic, f'brugge_5km_(51.209348, 3.224700).graphml')

