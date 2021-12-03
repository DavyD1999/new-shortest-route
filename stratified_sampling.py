import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import coordinate_functions as cf
from scipy.optimize import curve_fit

np.random.seed(42)

def function(x, A): 
    return A*x

"""
does stratified sampling so that the amount of values in each bin of total travel time are equal
for this it first uses some precomputed totally random nodes and their distance, then it fits a straight line to translate the haversine distance to the travel Time
after this curve fit it uses the original pre computed values and choses two random nodes and looks if the haversine distance * fitted constant fits in the bin and if this is the case
then actually calcultate the true travel time with networkx dijkstra if not then chose two other random nodes

possible improvement, select a node then make a circle of the haversine distance and chose points random on this circle and look which nodes are close to this circle based on this
one can fasten the selection procedure by quite some time one can use: osmnx.distance.nearest_nodes for this
"""

def stratified_sampling(amount_of_samples_per_bin, number_of_routes_pre_compute, step_size, node_list, graph):
    
    np.random.seed(42)
    
    list_indices_start = np.random.randint(0, len(node_list), size=number_of_routes_pre_compute) # first generate random numbers this is quicker
    list_indices_end = np.random.randint(0, len(node_list), size=number_of_routes_pre_compute)

    weight_path =  np.zeros(len(list_indices_start))
    distance = np.zeros(len(list_indices_start))


    for i in range(number_of_routes_pre_compute):
        
        while list_indices_start[i] == list_indices_end[i]: 
            list_indices_end[i] = np.random.randint(0, len(node_list)) # only change one now since it's a connected graph
        
        # calculate the shortest distance once
        weight_path[i] = nx.shortest_path_length(graph, node_list[list_indices_start[i]], node_list[list_indices_end[i]], 'travel_time')
        distance[i] = cf.distance(node_list[list_indices_start[i]], node_list[list_indices_end[i]],graph)
    

    font = { 'size'   : 16}

    mpl.rc('font', **font)
    
    mpl.style.use('tableau-colorblind10')
    #plt.scatter(distance, weight_path, marker='x', s=17.5, c='tab:purple')

    popt, pcov = curve_fit(function, distance, weight_path) # your data x, y to fit
    
    """
    xvals = np.arange(np.min(distance)-5, np.max(distance)+5) # step is 1 by defalult
    plt.plot(xvals, xvals*popt,  linewidth=2.5)
    plt.xlabel('hemelsbrede afstand (m)')
    plt.ylabel('reistijd (s)')
    plt.savefig('./stratified/rechtevenredig.png', bbox_inches='tight')
    plt.clf()
    """

    
    list_indices_start = list(list_indices_start)
    list_indices_end = list(list_indices_end)

    values, base = np.histogram(weight_path, bins=np.arange(start=0,stop=max(weight_path) + step_size, step=step_size))
    weight_path_list = list(weight_path)

    if max(values) > amount_of_samples_per_bin:
        print('in pre sampling got too much data in one bin')
        print(f'set the amount of samples per bin to {max(values)} instead of the given {amount_of_samples_per_bin}')
        amount_of_samples_per_bin = max(values)
    
    for i, value in enumerate(values):
        
        tot_with_new = value

        while tot_with_new < amount_of_samples_per_bin:
            u = np.random.randint(0, len(node_list), size=1)[0]
            v = np.random.randint(0, len(node_list), size=1)[0] # here selection of not the same is taken care of by the if statement below
            
            if base[i] < cf.distance(node_list[u], node_list[v],graph) * popt < base[i+1]:
                
                length = nx.shortest_path_length(graph, node_list[u], node_list[v], 'travel_time')
                
                if base[i] < length < base[i+1]:
                    list_indices_start.append(u)
                    list_indices_end.append(v)
                    weight_path_list.append(length)
                    tot_with_new += 1

    weight_path = np.array(weight_path_list)
    list_indices_end = np.array(list_indices_end)
    list_indices_start = np.array(list_indices_start)

    return weight_path, list_indices_start, list_indices_end

