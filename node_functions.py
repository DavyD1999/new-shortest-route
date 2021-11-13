import networkx as nx
import queue
import random
import numpy as np
random.seed(42)

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