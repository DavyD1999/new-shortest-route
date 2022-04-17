import networkx as nx
import random
import numpy as np
import fix_graph_data
import matplotlib.pyplot as plt
import a_star
import dijkstra_with_priority_que
import linear

random.seed(42)

def return_path(start_node, end_node, came_from):
    
    current_node = end_node
    path = {end_node}
    
    while start_node not in path:
        current_node = came_from[current_node]
        path.add(current_node)

    return path

def plot(visited, start, end, graph, path, name):

    node_list_start = list()
    node_list_end = list()
    node_list_path = list()
    node_list_expanded = list()
    node_list_rest = list()

    node_size_start = list()
    node_size_end = list()
    node_size_path = list()
    node_size_expanded = list()
    node_size_rest = list()

    color_start = list()
    color_end = list()
    color_path = list()
    color_expanded = list()
    color_rest = list()
    
    alpha_start = list()
    alpha_end = list()
    alpha_path = list()
    alpha_expanded = list()
    alpha_rest = list()
    
    for node in graph.nodes():
  
        if node==start:
            node_size_start.append(100)
            alpha_start.append(1)
            color_start.append('r')
            node_list_start.append(node)
        elif node == end:
            node_size_end.append(100)
            alpha_end.append(1)
            color_end.append('g')
            node_list_end.append(node)
        elif node in path:
            node_size_path.append(10)
            color_path.append('y')
            alpha_path.append(0.8)
            node_list_path.append(node)
        
        elif node in visited:
            node_size_expanded.append(6)
            color_expanded.append('k')
            alpha_expanded.append(0.8)
            node_list_expanded.append(node)
            
        else:
            node_size_rest.append(2)
            color_rest.append('b')
            alpha_rest.append(0.15)
            node_list_rest.append(node)


    nx.draw_networkx_nodes(graph, nodelist=node_list_expanded,pos=new_coordinates, node_color=color_expanded, node_size=node_size_expanded, alpha=alpha_expanded, node_shape='s', label='Geëxpandeerd')
    nx.draw_networkx_nodes(graph, nodelist=node_list_rest,pos=new_coordinates, node_color=color_rest, node_size=node_size_rest, alpha=alpha_rest, label='Overige')
    nx.draw_networkx_nodes(graph, nodelist=node_list_path,pos=new_coordinates, node_color=color_path, node_size=node_size_path, alpha=alpha_path, node_shape='v', label='Pad')
    nx.draw_networkx_nodes(graph, nodelist=node_list_start,pos=new_coordinates, node_color=color_start, node_size=node_size_start, alpha=alpha_start, node_shape='8', label='Start')
    nx.draw_networkx_nodes(graph, nodelist=node_list_end,pos=new_coordinates, node_color=color_end, node_size=node_size_end, alpha=alpha_end, node_shape='h', label='Bestemming')
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./semester2/expansions/{name}',  bbox_inches='tight')

    plt.clf()

city = 'big_graph'
number_of_paths = 1
number_of_landmarks = 8
graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
node_list = list(graph.nodes())

how_many = 2 * number_of_paths
random_generated = random.sample(node_list, k=how_many)
start_nodes = random_generated[:number_of_paths]
end_nodes = random_generated[number_of_paths:]

new_coordinates = dict()
max_velocity = fix_graph_data.get_max_velocity(graph) # gets the max velocity of all edges, useful for A*

clf = linear.add_amount_of_visited_weights(graph, number_of_landmarks=number_of_landmarks, cutoff=True)

for node in graph.nodes():
    new_coordinates[node] = np.array([graph.nodes[node]['x'], graph.nodes[node]['y']])


visited, came_from = a_star.A_star_priority_queue(start_nodes[0], end_nodes[0], graph, max_velocity-11, return_counter=False, return_visited=True)
path = return_path(start_nodes[0], end_nodes[0], came_from)
plot(visited, start_nodes[0], end_nodes[0], graph, path ,f'a*_minus_11{city}')

visited, came_from = a_star.A_star_priority_queue(start_nodes[0], end_nodes[0], graph, max_velocity, return_counter=False, return_visited=True)
path = return_path(start_nodes[0], end_nodes[0], came_from)
plot(visited, start_nodes[0], end_nodes[0], graph, path ,f'a*_max_vel{city}')

visited, came_from = dijkstra_with_priority_que.dijkstra_with_priority_queue_to_node(start_nodes[0], end_nodes[0], graph, return_visited=True)
path = return_path(start_nodes[0], end_nodes[0], came_from)
plot(visited, start_nodes[0], end_nodes[0], graph, path,f'dijkstra_{city}')

visited, came_from = linear.priority_queue_new_evaluation_function(start_nodes[0], end_nodes[0], graph, clf, return_visited=True)
path = return_path(start_nodes[0], end_nodes[0], came_from)

plot(visited, start_nodes[0], end_nodes[0], graph, path,f'linear_{city}')
