import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi

def force_ij(xi, xj, lij, kappa, travel_time):

    return kappa * (lij - np.linalg.norm(xi-xj)) * (xi-xj) /  np.linalg.norm(xi-xj) 

def springen(city, iterations): # NEW TECHNIQUES FOR GEOGRAPHIC ROUTING PHD PAPER
    infinite_edges = [[-10000,-10000],[0,-10000], [10000,-10000], [-10000,0],[-10000,10000], [10000,10000],[10000,0],[0,10000]]
    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')

    new_coordinates = dict()

    for node in graph.nodes():
        new_coordinates[node] = np.array([graph.nodes[node]['x'], graph.nodes[node]['y']])

    # for each node we will now add the percentage of common neighbors this way we don't have to calculate this every time we will just keep this in a seperate dictionary

    percentage_dict = dict() # will be a dict of a dict with percentage of common neighbors as values of this list dict

    for node in graph.nodes():
        si = set(nx.neighbors(graph, node))
        percentage_dict[node] = dict()
        
        for neighbor in si:
            sj = set(nx.neighbors(graph, neighbor))
            sij = si.intersection(sj)
                
            percentage_dict[node][neighbor] = len(sij) / (len(sij) + len(si) + len(sj))

            if neighbor in percentage_dict:
                assert percentage_dict[node][neighbor] == percentage_dict[neighbor][node], "asymmetry detected"


    for node1, node2, travel_time in graph.edges(data='travel_time'): # possible later adaptation with traveltime with lmin and lmax
        lmin = 0.00005 
        lmax = 0.01 * (1+travel_time/10) ** (1/2)
        
        graph[node1][node2]['rest_length'] = lmin + (1 - percentage_dict[node1][node2]) * (lmax - lmin) 
        
    kappa = 0.5
    delta = 0.5
    R_max = 10

    """
    nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions
    
    labels = nx.get_edge_attributes(graph,'travel_time')
    nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
    nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    plt.savefig('./semester2/springlayout/Brugge_gespringed_vooraf.png', dpi=500)
    plt.clf()
    """
    
    for i in range(iterations):
        force_dict = dict()
        for node in graph.nodes():
            force_dict[node] = 0

        for node1, node2, travel_time in graph.edges(data='travel_time'):
            coordinate_i = new_coordinates[node1]
            coordinate_j = new_coordinates[node2]

            force_on_i = force_ij(coordinate_i, coordinate_j, graph[node1][node2]['rest_length'], kappa, travel_time)
            force_on_j = - force_on_i # look it's newton

            force_dict[node1] += force_on_i
            force_dict[node2] += force_on_j

        repulsion_force = dict()

        for r, node in enumerate(graph.nodes()):
            print(r)
            repulsion_force[node] = 0
            
            coor_list = [new_coordinates[neighbor]]

            for neighbor in nx.neighbors(graph, node):
                coor_list.append(new_coordinates[neighbor])
            
            coor_list += infinite_edges # make sure our voronoi cell is closed
            vor = Voronoi(coor_list)
            
            index = vor.point_region[0] # there might be multiple closed cells so chose the right one for our region of ownership
            pol = Polygon([vor.vertices[q] for q in vor.regions[index]])
            
            for possible_intruder in graph.nodes(): # speed up possible by kdtrees
                if possible_intruder != node: # neighbors cant lie inside only our current node will be for sure
                    if pol.contains(Point(new_coordinates[possible_intruder])):
                        
                        unit_vector = (new_coordinates[node]-new_coordinates[possible_intruder]) / np.linalg.norm(new_coordinates[node]-new_coordinates[possible_intruder])
                        repulsion_force[node] += delta * unit_vector


        for node in graph.nodes():

            repulsion_force[node] = min(np.linalg.norm(repulsion_force[node]), R_max) / np.linalg.norm(repulsion_force[node]) * repulsion_force[node]
            print(repulsion_force)
            new_coordinates[node] += force_dict[node] + repulsion_force[node] 

        print(i)
    nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions

    labels = nx.get_edge_attributes(graph,'travel_time')
    nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
    nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    plt.savefig('./semester2/springlayout/Brugge_gespringed_achteraf.png', dpi=500)
    plt.clf()
springen('Brugge', 10)
    

    