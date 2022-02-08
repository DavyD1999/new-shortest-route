import networkx as nx
import queue
import random
import numpy as np
import coordinate_functions as cf
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter

random.seed(42)
np.random.seed(42)

font = { 'size'   : 16}

mpl.style.use('tableau-colorblind10')

def make_d_regular_tree(given_depth=3, degree=3, random_travel_time=False): 
    """
    will just make a regular tree with a given depth and if random travel time is true it will also give every edge a random travel time between 1 and 100s
    """
    index = 0

    parent = index
    graph = nx.Graph()
    graph.add_node(parent, depth=0)

    parent_queue = queue.Queue()
    parent_queue.put(parent)

    while not parent_queue.empty():  
        parent = parent_queue.get()
        new_depth = nx.get_node_attributes(graph, 'depth')[parent] + 1

        if new_depth > given_depth:
            break

        while graph.degree[parent] < degree: # keep adding nodes to the parent till degreee is reached
            index += 1
            graph.add_node(index, depth=new_depth) # the id of the node is the index we are currently at

            if random_travel_time is True:
                graph.add_edge(parent, index, travel_time=random.randint(1,100))                
            else:
                graph.add_edge(parent, index)
            parent_queue.put(index)
	
	# visualise easily with nx.forest_str(make_d_regular_tree())
    
    assert len(graph.nodes) == 3 * 2**given_depth - 2, 'the amount of nodes does not seem to be okay'
    return graph

def make_scale_free_graph(number_of_nodes, exponent):
    # below makes an array of the powerlaw distribution values but  and sort it so higher degree is in front
    degree_array = np.sort(np.round(np.array(nx.utils.random_sequence.powerlaw_sequence(number_of_nodes, exponent=exponent, seed=42))))[::-1]
 
    graph = nx.Graph()
    for index, element in enumerate(degree_array[:-1]): # last one does not really matter since we don't look at it in the loop
        assert element <= number_of_nodes - index - 1, 'the degree is too high for the number of nodes so chose a higher number of nodes'
    
    c = Counter(degree_array)
    voorkomens = c.most_common()
    x_array = list()
    y_array = list()
    
    for x,y in voorkomens:
        x_array.append(x)
        y_array.append(y)
    """
    plt.loglog(x_array, y_array, 'o', markersize=3) 
    plt.xlabel('Graad')
    plt.ylabel('Aantal')
    
    plt.savefig('./randoms/scalefreedegree.png')
    """    
    
    for i in range(number_of_nodes - 1): # last one should fix it self or be 1 degree left then don't care about it
        if i not in graph.nodes():
            graph.add_node(i)
        while degree_array[i] != 0:
            
            randint = random.randint(i+1, number_of_nodes-1) # +1 cause you don't want to pick yourself

            if randint not in graph.nodes():
                graph.add_node(randint)

            while randint in graph.neighbors(i):
                randint = random.randint(i+1, number_of_nodes-1)
            
            graph.add_edge(i, randint, travel_time=1)
            degree_array[i] -= 1
        
    return graph

def spring(graph, dimensions=2, weight_bool=True, iterations=25):
    
    init_pos_dict = dict()
    
    assert dimensions >= 2, 'dimensions must be at least 2'
    
    if dimensions == 2:
        for node in graph.nodes(data=True):
            init_pos_dict[node[0]] = (node[1]['x'], node[1]['y'])
    
    else:
        for node in graph.nodes(data=True):
            coor_array = np.zeros(dimensions)

            coor_array[:2] = [node[1]['x'], node[1]['y']]
            coor_array[2:] = np.random.rand(dimensions-2) - 0.5 # -0.5 so half will be below and half above 0, fills the rest with random coordinates
            init_pos_dict[node[0]] = coor_array


    if weight_bool:
        for edge in graph.edges(data='travel_time'):
            node1 = edge[0]
            node2 = edge[1]
            #print(edge)
            graph.add_edge(node1,node2, weight=1/edge[2]) #edge 2 is the traveltime thus apply force inversely proportional to the traveltime

        new_dict = nx.drawing.layout.spring_layout(graph, pos=init_pos_dict, iterations=iterations, dim=dimensions, weight='weight') # no seed since we have starting dictionary
    else:
        new_dict = nx.drawing.layout.spring_layout(graph, pos=init_pos_dict,  iterations=iterations, dim=dimensions)
        #new_dict = nx.drawing.layout.spring_layout(graph, pos=intit_pos_dict, iterations=50, dim=dimensions, seed=None)
    
    for node in graph.nodes():

        graph.add_node(node, coordinates=np.array(new_dict[node]))

    return graph

#graph = nx.read_gpickle(f'./graph_pickle/Brugge.gpickle')
#spring(graph)

def plot_stuck(start_node, end_node, node_stuck, graph, given_depth=1): 
    """
    will plot the coordinates and links from a node where the greedy function got stuck on and will also plot source and destination node
    for this it will first do bfs till it gets to the given depth and then it will plot the things
    """
    

    row = queue.Queue()
    row.put(node_stuck)
    depth_dict = dict()

    depth_dict[node_stuck] = 0
    
    mpl.rc('font', **font) # for plotting

    y_start, x_start = cf.get_coordinates(start_node, graph)
    plt.plot([x_start], [y_start], marker="^", markersize=10, label='start')

    y_end, x_end = cf.get_coordinates(end_node, graph)
    plt.plot([x_end], [y_end], marker="X", markersize=10, label = 'bestemming')


    while not row.empty():
        parent = row.get()
        new_depth = depth_dict[parent] + 1

        if new_depth > given_depth: # not nicest but easiest way to say stop if depth is too big, we use a fifo queue
            break

        for neighbor in graph.neighbors(parent):
            depth_dict[neighbor] = new_depth
            y1, x1 = cf.get_coordinates(neighbor, graph)
            y2, x2 = cf.get_coordinates(parent, graph)
   
            row.put(neighbor)
            plt.plot([x1,x2], [y1,y2], c='k', mfc='red', marker='o', linewidth=2, markersize=5)

    plt.plot([x1,x2], [y1,y2], c='k', mfc='red', marker='o', linewidth=2, markersize=5, label='buren')    # make sure the label is only shown once 
   
    y_stuck, x_stuck = cf.get_coordinates(node_stuck, graph) 
    plt.plot([x_stuck], [y_stuck], marker="*", markersize=10, label = 'huidige node')  # er boven plotten zodat de eerste marker niet zichtbaar is
    
    plt.legend()
    plt.savefig(f'./stuck_greedy/{node_stuck}_{start_node}_{end_node}', bbox_inches='tight')
    plt.clf()

#make_scale_free_graph(10000, 2.1)
