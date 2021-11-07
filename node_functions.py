import networkx as nx
import queue
import random

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

    while not parent_queue.empty():  # but will break sooner for efficiency
        parent = parent_queue.get()
        new_depth = nx.get_node_attributes(graph, 'depth')[parent] + 1

        if new_depth > given_depth:
            break

        while graph.degree[parent] < degree:
            index += 1
            graph.add_node(index, depth=new_depth)

            if random_travel_time is True:
                graph.add_edge(parent, index, travel_time=random.randint(1,100))                
            else:
                graph.add_edge(parent, index)
            parent_queue.put(index)
	
	# visualise easily with nx.forest_str(make_d_regular_tree())
    
    assert len(graph.nodes) == 3 * 2**given_depth - 2, 'the amount of nodes does not seem to be okay'
    return graph


print(make_d_regular_tree(given_depth=10))