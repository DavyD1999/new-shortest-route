import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, KDTree
import time
import pickle
import multiprocessing as mp
import random
import greedy_forwarding_route as gf
import coordinate_functions as cf


def add_coordinates(graph, dictionary):
    
    for node in graph.nodes():
        graph.add_node(node, coordinates=np.array(dictionary[node]))

    return graph

def force_ij(xi, xj, lij, kappa):
    return kappa * (lij - np.linalg.norm(xi - xj)) * (xi - xj) / (np.linalg.norm(xi - xj) + 0.00001)


def solve(x_new, y_new, neighbors, check_set ,new_coordinates, index_list):
    neighbor_coor_list = list()
    
    for neighbor in neighbors:
        neighbor_coor_list.append(new_coordinates[neighbor])

    neighbor_array = np.array(neighbor_coor_list)

    chosen = random.sample(check_set, 1)
    name = index_list[chosen[0]]

    coor_chosen = new_coordinates[name]

    dist_list = np.sum((neighbor_array - coor_chosen)**2, axis=1) # we neet as many distances as neighb
    
    
    min_dist = np.min(np.sqrt(dist_list))
    coor_current = np.array([x_new, y_new])

    m = (coor_current[1]-coor_chosen[1])/(coor_current[0]-coor_chosen[0])

    spr = np.sqrt(min_dist**2/ (1 + m**2))
    
    if coor_current[0] < coor_chosen[0]:
        x_new = -spr + coor_chosen[0] - 0.000001 
    else: # chose closest solution to original 
        x_new = spr + coor_chosen[0] + 0.000001 # to be sure it's outside

    y_new = m * (x_new - coor_chosen[0]) + coor_chosen[1]


    return x_new, y_new, coor_chosen

infinite_edges = [[-10000, -10000], [0, -10000], [10000, -10000], [-10000, 0], [-10000, 10000], [10000, 10000],
                  [10000, 0], [0, 10000]]



def calc_force(update_dict, conflict_dict, repulsion_force, node_list, new_coordinates, graph, kd, index_list, delta):


    for r, current_node in enumerate(node_list):
        if r % 50 == 0:
            print(r)
        repulsion_force_now = 0  # keep track of the repulsion force felt by this node
        coor_current = new_coordinates[current_node]

        # create region of ownership
        coor_list = [new_coordinates[current_node]]

        for neighbor in nx.neighbors(graph, current_node):
            coor_list.append(new_coordinates[neighbor])

        coor_list += infinite_edges  # make sure our voronoi cell is closed
        vor = Voronoi(coor_list)

        index = vor.point_region[0]  # there might be multiple closed cells so chose the right one for our region of ownership
        pol = Polygon([vor.vertices[q] for q in vor.regions[index]])

        # now check if other nodes are in it
        possibly_inside = kd.query_ball_point(pol.centroid,r=pol.centroid.hausdorff_distance(pol))  # get only nodes within this ball

        check_set = set()

        for ix in possibly_inside:

            possible_intruder = index_list[ix]

            coor_intruder = new_coordinates[possible_intruder]
            substract = coor_current - coor_intruder
            if (substract == 0).all():  # if the point we are actually determining region of ownership of
                continue

            elif pol.contains(Point(coor_intruder)) is True:

                check_set.add(ix)

        if check_set != set():
            if random.uniform(0.0,1.0) < 0.1:
                start = True
                previous_set = set()
                iters = 0
                repulsion_force_now = 0
                while len(check_set) < len(previous_set) or start is True :  # as long as we keep decreasing the number of nodes in the conflict set keep going
                    
                    previous_set = set(check_set)
                    if start is True:
                        x_new, y_new, coor_chosen = solve(new_coordinates[current_node][0], new_coordinates[current_node][1], nx.neighbors(graph, current_node), check_set, new_coordinates, index_list)
                        start = False
                    else:
                        x_new, y_new, coor_chosen = solve(x_new, y_new, nx.neighbors(graph, current_node), check_set, new_coordinates, index_list)
                        
                    
                    coor_list[0] = [x_new, y_new]
                    vor = Voronoi(coor_list)

                    index = vor.point_region[0]  # there might be multiple closed cells so chose the right one for our region of ownership
                    pol = Polygon([vor.vertices[q] for q in vor.regions[index]])

                    if pol.contains(Point(coor_chosen)):

                        print('oei dit punt ligt er nog steeds in')

                    possibly_inside = kd.query_ball_point(pol.centroid,r=pol.centroid.hausdorff_distance(pol))  # get only nodes within this ball

                    check_set = set()
            
                    for ix in possibly_inside:
            
                        possible_intruder = index_list[ix]
            
                        coor_intruder = new_coordinates[possible_intruder]
                        substract = coor_current - coor_intruder
                        if (substract == 0).all():  # if the point we are actually determining region of ownership of
                            continue
            
                        elif pol.contains(Point(coor_intruder)) is True:
            
                            check_set.add(ix)

                    iters += 1
                    if check_set == set() or iters > 5:
                        break
                    

                if check_set == set():   # only change if solved
                    new_coordinates[current_node] = np.array([x_new, y_new])
                    
            else:    
                #if not check_set.issubset(conflict_dict[current_node]):
                for conflictor in check_set:
                    intruder = index_list[conflictor]
                    coor_intruder = new_coordinates[intruder]
    
                    substract = coor_current - coor_intruder
                    unit_vector = (substract) / np.linalg.norm(substract)
                    repulsion_force_now += delta * unit_vector
    
                conflict_dict[current_node] = set.union(check_set, conflict_dict[
                    current_node])  # update the set of conflict, just add new ones

        else:
            repulsion_force_now = 0

        repulsion_force[current_node] = repulsion_force_now



def springen(city, iterations, start_number=0, coor_dict=False):  # NEW TECHNIQUES FOR GEOGRAPHIC ROUTING PHD PAPER

    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')

    new_coordinates = dict()

    if coor_dict is False:
        x_list = list()
        y_list = list()

        for node in graph.nodes(): # normalize data and multiply with 20 so good enough for voronoi region of ownership
            x_list.append(graph.nodes[node]['x'])
            y_list.append(graph.nodes[node]['y'])

        minx = min(x_list)
        maxx = max(x_list)

        miny = min(y_list)
        maxy = max(y_list)

        xdif = maxx - minx
        ydif = maxy - miny
        for node in graph.nodes():
            new_coordinates[node] = np.array([(graph.nodes[node]['x']-minx)/xdif, (graph.nodes[node]['y']-miny)/ydif]) * 20 

    else:
        print('geladen')
        new_coordinates = coor_dict
    # for each node we will now add the percentage of common neighbors this way we don't have to calculate this every time we will just keep this in a seperate dictionary

    graph_spring = add_coordinates(graph, new_coordinates)

    tot_aangekomen = 0

    node_list = list(graph.nodes())

    for i in range(0,100,2):
        result, ratio_travelled = gf.greedy_forwarding(node_list[i], node_list[10*i], graph_spring, distance_function=cf.euclidian_n_dimensions ,ratio_travelled=True)
        if result < np.inf:
            tot_aangekomen += 1

    print(f'er is zoveel aangekomen {tot_aangekomen}')
    l_min_prop = np.inf
    l_max_prop = - np.inf

    """
    new_coordinates = nx.spring_layout(graph, pos=new_coordinates, iterations=1000, weight='travel_time')

    with open(f'./networkx_gespringed.pickle', 'wb') as handle:
        pickle.dump(new_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    """

    kappa = 0.0005 * 10
    delta = 0.0005 * 10
    R_max = delta * 20
    alpha_max = 10 * kappa
    T = 100

    fig, ax = plt.subplots()

    nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions



    nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.savefig('./semester2/convex/Brugge_gespringed_vooraf.png', dpi=500)
    ax.clear()
    plt.clf()

    for node1, node2 in graph.edges(): # determine how close the nodes are
        coor1 = new_coordinates[node1]
        coor2 = new_coordinates[node2]

        dist = np.sqrt(np.sum((coor1-coor2)**2))

        if dist > l_max_prop:
            l_max_prop = dist

        if dist < l_min_prop:
            l_min_prop = dist
            


    for node1, node2, travel_time in graph.edges(data='travel_time'):
        si = set(nx.neighbors(graph, node1))
        sj = set(nx.neighbors(graph, node2))

        sij = si.intersection(sj)
        sj = sj.difference(sij)
        si = si.difference(sij)

        rij = len(sij) / (len(sij) + len(si) + len(sj))

        lmin = l_min_prop / 1.3
        lmax = l_max_prop * 1.3

        graph[node1][node2]['rest_length'] = lmin + (1 - rij) * (lmax - lmin)

    conflict_dict = dict()

    for node in graph.nodes():  # initialize empty
        conflict_dict[node] = set()

    for i in range(start_number, iterations):
        force_dict = dict()

        list_kdtree = list()

        index_list = list()
        for node in graph.nodes():
            force_dict[node] = 0
            list_kdtree.append(new_coordinates[node])
            index_list.append(node)  # return value for kdtree is just this index

        kd = KDTree(np.array(list_kdtree))  # we need to make new kdtree every time step because coordinates changed

        for node1, node2 in graph.edges():
            coordinate_i = new_coordinates[node1]
            coordinate_j = new_coordinates[node2]

            force_on_i = force_ij(coordinate_i, coordinate_j, graph[node1][node2]['rest_length'], kappa)
            force_on_j = - force_on_i  # look it's newton

            force_dict[node1] += 0.0000001#force_on_i
            force_dict[node2] += 0.0000001 #force_on_j
        
        
        with mp.Manager() as manager:  # multiprocessing step
            conflict_dict = manager.dict(conflict_dict)
            repulsion_force = manager.dict()
            update_dict = manager.dict()

            p1 = mp.Process(target=calc_force, args=(update_dict,
                conflict_dict, repulsion_force, node_list[:len(node_list) // 4], new_coordinates, graph, kd, index_list, delta)) # VERGEET NIET TERUG TE ZETTEN NAAR JUISTE VORM
            p2 = mp.Process(target=calc_force, args=(update_dict,
                conflict_dict, repulsion_force, node_list[len(node_list) // 4:2 * len(node_list) // 4], new_coordinates,
                graph, kd, index_list, delta))
            p3 = mp.Process(target=calc_force, args=(update_dict,
                conflict_dict, repulsion_force, node_list[2 * len(node_list) // 4:3 * len(node_list) // 4], new_coordinates,
                graph, kd, index_list, delta))
            p4 = mp.Process(target=calc_force, args=(update_dict,
                conflict_dict, repulsion_force, node_list[3 * len(node_list) // 4:], new_coordinates, graph, kd,
                index_list, delta))

            p1.start()
            p2.start()
            p3.start()
            p4.start()

            p1.join()
            p2.join()
            p3.join()
            p4.join()

            conflict_dict = dict(conflict_dict)
            repulsion_force = dict(repulsion_force)
            update_dict = dict(update_dict)        
        
        if i > T:
            alpha_now = alpha_max * np.exp(-i / T)  # hysteresis
        else:
            alpha_now = alpha_max

        for node in graph.nodes():
            if node in update_dict:
                print(len(update_dict))
                new_coordinates[node] = update_dict[node]
            
            else:
                if node in repulsion_force:
                    repulsion = delta * repulsion_force[node]
                    if np.linalg.norm(repulsion) == 0:
                        repulsion_force[node] == repulsion
                    else:
                        repulsion_force[node] = min(np.linalg.norm(repulsion), R_max) / np.linalg.norm(repulsion) * repulsion
                else:
                    repulsion_force[node] = 0

                new_coordinates[node] += min(np.linalg.norm(force_dict[node] + repulsion_force[node]),
                                             alpha_now) / np.linalg.norm(force_dict[node] + repulsion_force[node]) * (
                                                     force_dict[node] + repulsion_force[node])

        print(f' iteratie {i}')
        fig, ax = plt.subplots()

        nx.draw_networkx_nodes(graph, new_coordinates, node_size=1, ax=ax)  # new layout specifies positions

        # labels = nx.get_edge_attributes(graph, 'travel_time')
        nx.draw_networkx_edges(graph, pos=new_coordinates, width=0.5, node_size=0.1)
        #nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.savefig('./semester2/convex/Brugge_gespringed_achteraf.png', dpi=500)
        ax.clear()
        plt.clf()

        

        graph_spring = add_coordinates(graph, new_coordinates)

        tot_aangekomen = 0

        for i in range(0,100,2):
            result, ratio_travelled = gf.greedy_forwarding(node_list[i], node_list[10*i], graph_spring, distance_function=cf.euclidian_n_dimensions ,ratio_travelled=True)
            if result < np.inf:
                tot_aangekomen += 1

        print(f'er is zoveel aangekomen {tot_aangekomen}')

        with open(f'./semester2/super lange simulatie.pickle', 'wb') as handle:
            pickle.dump(new_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
graph = nx.Graph()
graph.add_nodes_from([1,2,3])
graph.add_edges_from([[1,2],[1,3]])
graph.nodes[1]['x'] = 0.
graph.nodes[1]['y'] = 0.
graph.nodes[2]['x'] = 1.
graph.nodes[2]['y'] = 0.
graph.nodes[3]['x'] = 1.3
graph.nodes[3]['y'] = 1.3

"""

springen('Brugge', 1000, coor_dict=False)




"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, KDTree
import time
import pickle
import multiprocessing as mp


def force_ij(xi, xj, lij, kappa, travel_time):

    return kappa * (lij - np.linalg.norm(xi-xj)) * (xi-xj) /  np.linalg.norm(xi-xj) 

def calc_force(conflict_dict, repulsion_force, node_list, new_coordinates, graph, kd, index_list):
    infinite_edges = [[-10000,-10000],[0,-10000], [10000,-10000], [-10000,0],[-10000,10000], [10000,10000],[10000,0],[0,10000]]
    
    for r, current_node in enumerate(node_list):
        print(r)
        repulsion_force_now = 0 # keep track of the repulsion force felt by this node
        coor_current = new_coordinates[current_node]

        # create region of ownership
        coor_list = [new_coordinates[current_node]]

        for neighbor in nx.neighbors(graph, current_node):
            coor_list.append(new_coordinates[neighbor])
        
        coor_list += infinite_edges # make sure our voronoi cell is closed
        vor = Voronoi(coor_list)
        
        index = vor.point_region[0] # there might be multiple closed cells so chose the right one for our region of ownership
        pol = Polygon([vor.vertices[q] for q in vor.regions[index]])

        # now check if other nodes are in it
        possibly_inside = kd.query_ball_point(pol.centroid,r=pol.centroid.hausdorff_distance(pol)) # get only nodes within this ball

        check_set = set()
        for ix in possibly_inside:
            
            possible_intruder = index_list[ix]
            coor_intruder = new_coordinates[possible_intruder]
            substract = coor_current - coor_intruder
            if (substract==0).all():  # if the point we are actually determining region of ownership of
                continue

            elif pol.contains(Point(coor_intruder)) is True:
                check_set.add(ix)

                
        if check_set: # later adapt this

            for conflictor in check_set:
                
                intruder = index_list[conflictor]
                coor_intruder = new_coordinates[intruder]
                
                substract = coor_current - coor_intruder
                unit_vector = (substract) / np.linalg.norm(substract)
                repulsion_force_now +=  unit_vector
            
            conflict_dict[current_node] = set.union(check_set, conflict_dict[current_node])  # update the set of conflict, just add new ones
        else:
            print('no')
            repulsion_force_now = 0

        repulsion_force[current_node] = repulsion_force_now

def springen(city, iterations, start_number=0, coor_dict=False): # NEW TECHNIQUES FOR GEOGRAPHIC ROUTING PHD PAPER

    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')

    new_coordinates = dict()

    if coor_dict is False:
        for node in graph.nodes():
            new_coordinates[node] = np.array([graph.nodes[node]['x'], graph.nodes[node]['y']])
    else:
        print('geladen')
        new_coordinates = coor_dict
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
        lmax = 0.01 #* (1+travel_time/10) ** (1/2)
        
        graph[node1][node2]['rest_length'] = lmin + (1 - percentage_dict[node1][node2]) * (lmax - lmin) 
        
    kappa = 0.5
    delta = 0.00005
    R_max = 10
    alpha_max = 0.005
    T = 50

    """
    #nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions
    
    #labels = nx.get_edge_attributes(graph,'travel_time')
    #nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
    #nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    #plt.savefig('./semester2/springlayout/Brugge_gespringed_vooraf.png', dpi=500)
    #plt.clf()
"""

conflict_dict = dict()

for node in graph.nodes(): # initialize empty
    conflict_dict[node] = set() 
      
for i in range(start_number, iterations):
    force_dict = dict()

    list_kdtree = list()

    index_list = list()
    for node in graph.nodes():
        force_dict[node] = 0
        list_kdtree.append(new_coordinates[node]) 
        index_list.append(node) # return value for kdtree is just this index

    kd = KDTree(np.array(list_kdtree)) # we need to make new kdtree every time step because coordinates changed

    for node1, node2, travel_time in graph.edges(data='travel_time'):
        coordinate_i = new_coordinates[node1]
        coordinate_j = new_coordinates[node2]

        force_on_i = force_ij(coordinate_i, coordinate_j, graph[node1][node2]['rest_length'], kappa, travel_time)
        force_on_j = - force_on_i # look it's newton

        force_dict[node1] += force_on_i
        force_dict[node2] += force_on_j
        
    repulsion_force = dict()

    node_list = list(graph.nodes())

    with mp.Manager() as manager: # multiprocessing step
        conflict_dict = manager.dict(conflict_dict)
        repulsion_force = manager.dict()
        
        p1 = mp.Process(target=calc_force, args=(conflict_dict,repulsion_force, node_list[:len(node_list)//4], new_coordinates, graph, kd, index_list))
        p2 = mp.Process(target=calc_force, args=(conflict_dict,repulsion_force, node_list[len(node_list)//4:2*len(node_list)//4], new_coordinates, graph, kd, index_list))
        p3 = mp.Process(target=calc_force, args=(conflict_dict,repulsion_force,  node_list[2*len(node_list)//4:3*len(node_list)//4], new_coordinates, graph, kd, index_list))
        p4 = mp.Process(target=calc_force, args=(conflict_dict,repulsion_force, node_list[3*len(node_list)//4:], new_coordinates, graph, kd, index_list))
        
        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        conflict_dict = dict(conflict_dict)
        repulsion_force = dict(repulsion_force)
    
    
    if i > T:
        alpha_now = alpha_max * np.exp(-i/T) # hysteresis
    else:
        alpha_now = alpha_max

    for node in graph.nodes():
        repulsion = delta * repulsion_force[node]
        if np.linalg.norm(repulsion) == 0:
            repulsion_force[node] == repulsion
        else:
            repulsion_force[node] = min(np.linalg.norm(repulsion), R_max) / np.linalg.norm(repulsion) * repulsion

        new_coordinates[node] += min(np.linalg.norm(force_dict[node] + repulsion_force[node]), alpha_now) / np.linalg.norm(force_dict[node] + repulsion_force[node]) * (force_dict[node] + repulsion_force[node])

    print(i)
    nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions

    labels = nx.get_edge_attributes(graph,'travel_time')
    nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
    nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    plt.savefig('./semester2/springlayout/Brugge_gespringed_achteraf.png', dpi=500)
    plt.clf()

    with open(f'./semester2/springlayout/{city}_{i}.pickle', 'wb') as handle:
        pickle.dump(new_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)

def add_coordinates(graph, dictionary):
for node in graph.nodes():
    graph.add_node(node, coordinates=np.array(dictionary[node]))

return graph

"""
#with open(f'./semester2/springlayout/Brugge_6.pickle', 'rb') as handle:
# coor_dict = pickle.load(handle)
        
"""


springen('Brugge', 100)      

"""


"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import Voronoi, KDTree
import time
import pickle
import multiprocessing as mp


def force_ij(xi, xj, lij, kappa, travel_time):
return kappa * (lij - np.linalg.norm(xi - xj)) * (xi - xj) / np.linalg.norm(xi - xj)


def calc_force(conflict_dict, repulsion_force, node_list, new_coordinates, graph, kd, index_list):
infinite_edges = [[-10000, -10000], [0, -10000], [10000, -10000], [-10000, 0], [-10000, 10000], [10000, 10000],
                  [10000, 0], [0, 10000]]

for r, current_node in enumerate(node_list):
    print(r)
    repulsion_force_now = 0  # keep track of the repulsion force felt by this node
    coor_current = new_coordinates[current_node]

    # create region of ownership
    coor_list = [new_coordinates[current_node]]

    for neighbor in nx.neighbors(graph, current_node):
        coor_list.append(new_coordinates[neighbor])

    coor_list += infinite_edges  # make sure our voronoi cell is closed
    vor = Voronoi(coor_list)

    index = vor.point_region[
        0]  # there might be multiple closed cells so chose the right one for our region of ownership
    pol = Polygon([vor.vertices[q] for q in vor.regions[index]])

    # now check if other nodes are in it
    possibly_inside = kd.query_ball_point(pol.centroid,
                                          r=pol.centroid.hausdorff_distance(pol))  # get only nodes within this ball

    check_set = set()
    for ix in possibly_inside:

        possible_intruder = index_list[ix]
        coor_intruder = new_coordinates[possible_intruder]
        substract = coor_current - coor_intruder
        if (substract == 0).all():  # if the point we are actually determining region of ownership of
            continue

        elif pol.contains(Point(coor_intruder)) is True:
            check_set.add(ix)

    if not check_set.issubset(conflict_dict[current_node]):

        for conflictor in check_set:
            intruder = index_list[conflictor]
            coor_intruder = new_coordinates[intruder]

            substract = coor_current - coor_intruder
            unit_vector = (substract) / np.linalg.norm(substract)
            repulsion_force_now += unit_vector

        conflict_dict[current_node] = set.union(check_set, conflict_dict[
            current_node])  # update the set of conflict, just add new ones
    else:
        print('no')
        repulsion_force_now = 0

    repulsion_force[current_node] = repulsion_force_now


def springen(city, iterations, start_number=0, coor_dict=False):  # NEW TECHNIQUES FOR GEOGRAPHIC ROUTING PHD PAPER

graph = nx.read_gpickle(f'Brugge.gpickle')

new_coordinates = dict()

if coor_dict is False:
    for node in graph.nodes():
        new_coordinates[node] = np.array([graph.nodes[node]['x'], graph.nodes[node]['y']])
else:
    print('geladen')
    new_coordinates = coor_dict
# for each node we will now add the percentage of common neighbors this way we don't have to calculate this every time we will just keep this in a seperate dictionary

percentage_dict = dict()  # will be a dict of a dict with percentage of common neighbors as values of this list dict
l_min_prop = np.inf
l_max_prop = - np.inf
for node1, node2 in graph.edges():
    x1 = graph.nodes[node1]['x']
    y1 = graph.nodes[node1]['y']

    x2 = graph.nodes[node2]['x']
    y2 = graph.nodes[node2]['y']

    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    if dist > l_max_prop:
        l_max_prop = dist

    if dist < l_min_prop:
        l_min_prop = dist


for node in graph.nodes():
    si = set(nx.neighbors(graph, node))
    percentage_dict[node] = dict()

    for neighbor in si:
        sj = set(nx.neighbors(graph, neighbor))
        sij = si.intersection(sj)

        percentage_dict[node][neighbor] = len(sij) / (len(sij) + len(si) + len(sj))

        if neighbor in percentage_dict:
            assert percentage_dict[node][neighbor] == percentage_dict[neighbor][node], "asymmetry detected"

for node1, node2, travel_time in graph.edges(
        data='travel_time'):  # possible later adaptation with traveltime with lmin and lmax
    lmin = l_min_prop
    lmax = l_max_prop

    graph[node1][node2]['rest_length'] = lmin + (1 - percentage_dict[node1][node2]) * (lmax - lmin)

kappa = 0.5
delta = 0.00005
R_max = 10
alpha_max = 0.0004
T = 900

"""
#nx.draw_networkx_nodes(graph, new_coordinates, node_size=1) # new layout specifies positions

#labels = nx.get_edge_attributes(graph,'travel_time')
#nx.draw_networkx_edges(graph, pos=new_coordinates, edgelist=None, width=0.5, node_size=1)
#nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
#plt.savefig('./semester2/springlayout/Brugge_gespringed_vooraf.png', dpi=500)
#plt.clf()
"""

conflict_dict = dict()

for node in graph.nodes():  # initialize empty
    conflict_dict[node] = set()

for i in range(start_number, iterations):
    force_dict = dict()

    list_kdtree = list()

    index_list = list()
    for node in graph.nodes():
        force_dict[node] = 0
        list_kdtree.append(new_coordinates[node])
        index_list.append(node)  # return value for kdtree is just this index

    kd = KDTree(np.array(list_kdtree))  # we need to make new kdtree every time step because coordinates changed

    for node1, node2, travel_time in graph.edges(data='travel_time'):
        coordinate_i = new_coordinates[node1]
        coordinate_j = new_coordinates[node2]

        force_on_i = force_ij(coordinate_i, coordinate_j, graph[node1][node2]['rest_length'], kappa, travel_time)
        force_on_j = - force_on_i  # look it's newton

        force_dict[node1] += force_on_i
        force_dict[node2] += force_on_j

    repulsion_force = dict()

    node_list = list(graph.nodes())

    with mp.Manager() as manager:  # multiprocessing step
        conflict_dict = manager.dict(conflict_dict)
        repulsion_force = manager.dict()

        p1 = mp.Process(target=calc_force, args=(
        conflict_dict, repulsion_force, node_list[:len(node_list) // 4], new_coordinates, graph, kd, index_list))
        p2 = mp.Process(target=calc_force, args=(
        conflict_dict, repulsion_force, node_list[len(node_list) // 4:2 * len(node_list) // 4], new_coordinates,
        graph, kd, index_list))
        p3 = mp.Process(target=calc_force, args=(
        conflict_dict, repulsion_force, node_list[2 * len(node_list) // 4:3 * len(node_list) // 4], new_coordinates,
        graph, kd, index_list))
        p4 = mp.Process(target=calc_force, args=(
        conflict_dict, repulsion_force, node_list[3 * len(node_list) // 4:], new_coordinates, graph, kd,
        index_list))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        conflict_dict = dict(conflict_dict)
        repulsion_force = dict(repulsion_force)

    if i > T:
        alpha_now = alpha_max * np.exp(-i / T)  # hysteresis
    else:
        alpha_now = alpha_max

    for node in graph.nodes():
        repulsion = delta * repulsion_force[node]
        if np.linalg.norm(repulsion) == 0:
            repulsion_force[node] == repulsion
        else:
            repulsion_force[node] = min(np.linalg.norm(repulsion), R_max) / np.linalg.norm(repulsion) * repulsion

        new_coordinates[node] += min(np.linalg.norm(force_dict[node] + repulsion_force[node]),
                                     alpha_now) / np.linalg.norm(force_dict[node] + repulsion_force[node]) * (
                                             force_dict[node] + repulsion_force[node])

    print(i)
    nx.draw_networkx_nodes(graph, new_coordinates, node_size=1)  # new layout specifies positions

    labels = nx.get_edge_attributes(graph, 'travel_time')
    nx.draw_networkx_edges(graph, pos=new_coordinates, width=0.5, node_size=0.1)
    #nx.draw_networkx_edge_labels(graph, pos=new_coordinates, edge_labels=labels, label_pos=0.5, font_size=3)
    plt.savefig('Brugge_gespringed_achteraf.png', dpi=500)
    plt.clf()

    with open(f'../super lange simulatie.pickle', 'wb') as handle:
        pickle.dump(new_coordinates, handle, protocol=pickle.HIGHEST_PROTOCOL)


def add_coordinates(graph, dictionary):
    for node in graph.nodes():
        graph.add_node(node, coordinates=np.array(dictionary[node]))

    return graph



if __name__ == '__main__':
    print('jaja')
    springen('Brugge', 1000)

print('uitgevoerd')

"""