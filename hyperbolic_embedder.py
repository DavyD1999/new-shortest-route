import numpy as np
import queue
import matplotlib.pyplot as plt
import cmath
import random
import networkx as nx
import node_functions as nf
from mpmath import mp
import matplotlib as mpl
import sys

mpl.style.use('tableau-colorblind10')

mp.prec = 100 # sets the precision 
random.seed(42)
"""
calculates a hyperbolic embedding for a undirected graph based on it's travel time
"""

def find_start_node(min_tree): # finds a start node that will minimize tree depth look at https://sites.math.rutgers.edu/~ajl213/CLRS/Ch22.pdf
  '''
  two times bfs will give us what we want
  '''
  bfs_queue = queue.Queue()
  visited = set()
  node_id = random.choice(list(min_tree.nodes()))
  
  bfs_queue.put(node_id)
  while not bfs_queue.empty():
      node = bfs_queue.get()
      visited.add(node)
      for neighbor in min_tree.neighbors(node):
          if neighbor not in visited:
            bfs_queue.put(neighbor)

  furthest = node
  bfs_queue.put(furthest) # was empty now not anymore
  
  visited = set() # reset the visited since we need to start all over again
  while not bfs_queue.empty():
      node = bfs_queue.get()
      visited.add(node)
      for neighbor in min_tree.neighbors(node):
          if neighbor not in visited:
            bfs_queue.put(neighbor)
  
  other_furthest = node

  path = nx.shortest_path(min_tree, source=furthest, target=other_furthest) # returns the shortest path counted in hops (just going from one node to other node not taking into account traveltime)

  print(len(path))
 
  return path[len(path)//2] 
"""
# to test above funtion
tree = nf.make_d_regular_tree(given_depth=8,degree=3)

print(find_start_node(tree))

"""
def find_max_degree(min_tree): # returns node and the degree of this node
  # first gets us the degree of every node in the tree, which gives us tuples of the type (node_id, degree) makes a numpy array out of it and then finds the max of the second argument
  node_array = np.array(min_tree.degree((min_tree.nodes())))
  max_degree = np.max(node_array[:,1])
  # find which element this max_degree is
  index = np.where(node_array[:,1]==max_degree)[0][0] # just take the first one that was encountered
  return node_array[index][1] # first value is node id and second one is the degree

def find_max_hyperbolic_distance(node_dict, root_node):
    max_dist = 0
    for key, value in node_dict.items():
        if 2*mp.atanh(abs(value))>max_dist:
            max_dist = 2*mp.atanh(abs(value))
    
    return max_dist



def transformation(z_node, z_parent):
    alpha = 1 / z_node.conjugate()
    #z_return = (z_parent - z_node) / (1 - z_node.conjugate() * z_parent)
    z_return = (z_parent.conjugate() * alpha - 1) / (z_parent.conjugate()- alpha.conjugate())
    return z_return

def inverse_transformation(z_node, z_child):
    alpha = 1 / z_node.conjugate()
    #print((z_child + z_node) / ( 1 + z_child * z_node.conjugate()))
    #z_return = (z_child + z_node) / ( 1 + z_child * z_node.conjugate())
    z_return = ( alpha*z_child.conjugate() - 1 ) / (z_child.conjugate() - alpha.conjugate())
    return z_return

"""
#Example how to use transformations: 
a = complex(0.5, 0.6)
b = complex(0.2, 0.1)

b = transformation(a, b)
print(transformation(a,a)) # will map a to zero

#a = inverse_transformation(0,a) # will map a again to it's original value
print(inverse_transformation(a,b)) # with this old a or new new a it will map b again to it's original value so inverse_function(function(z)) = z
"""
pi = np.pi

def sarkar_formula(node_id, parent_id_of_node, node_dict, came_from, min_tree, tau_func, start=False): # from arxiv: 1804.03329, # coordinates is numpy array of two coordinates

    nodes_added = set()
    degree = min_tree.degree[node_id]
    
    if start is True:
        theta = 0
        k = 0 # starts at zero here since we want a full circle
        came_from[node_id] = parent_id_of_node # this should be parent==child if not something is wrong
        
        for child in min_tree.neighbors(node_id):
            
            #new_val  = tau_func * complex(np.cos(theta + 2 * pi * k/ degree),  np.sin(theta + 2 * pi * k/ degree))
            val = mp.exp(1j*(theta + 2 * pi * k / degree)) * tau_func

            #print(new_val)
            assert 2 * pi * k/ degree < 2 * pi - 0.001 # here willl be full circle
            #assert abs(val - new_val) < 0.0001
            node_dict[child] = val
            # print(abs(2*np.arctanh(abs((new_val)/(1))))) # check to look if right distance embedded
            came_from[child] = node_id # the child of the node is now embeded and we want to remember te structure since we'll need it later to embed the children of the newly embedded node
            nodes_added.add(child)
            
            k += 1

        
        return node_dict, came_from, nodes_added

    z_node = node_dict[node_id]
    z_parent = node_dict[parent_id_of_node]
    z_parent_new = transformation(z_node, z_parent)
    theta = mp.arg(z_parent_new) 

    k = 1

    for child in min_tree.neighbors(node_id):
        
        if child not in node_dict: # could be already embedded if parent 
            #new_val = tau_func * complex(np.cos(theta + 2 * pi * k/ degree), np.sin(theta + 2 * pi * k/ degree))
            
            val = mp.exp(1j*(theta + 2 * pi * k / degree)) * tau_func
            #assert abs(val - new_val) < 0.0001
            assert 2 * pi * k/ degree < 2 * pi -0.001 # not allowed to be full circle  since origin is at 0 or two pi
            z_child = val
            
            inv_transform_child = inverse_transformation(z_node, z_child)
            assert abs(inv_transform_child) < 1 # should be in poincare disk use mpmath for better precision
            # print(2*np.arctanh(abs((z_node-inv_transform_child)/(1-z_node.conjugate()*inv_transform_child))))

            node_dict[child] = inv_transform_child
            came_from[child] = node_id
            nodes_added.add(child)        
            k += 1

    
    return node_dict, came_from, nodes_added


def hyperbolic_embed(min_tree, scaling_factor=2.3, came_from_return=False): # voor verder zie http://math.haifa.ac.il/ROVENSKI/B2.pdf p. 368
    max_degree = find_max_degree(min_tree)
    print(f'{max_degree} is the max degree')
    
    constant = 2 * np.log10(max_degree*2/np.pi)
    scaling_factor = 2*constant + 0.001 # to make sure epsilon is smaller than 1 from Representation Tradeoffs for Hyperbolic Embeddings
    print(f'{scaling_factor} is the scaling factor')
    assert scaling_factor > 1
    
    current_node = find_start_node(min_tree)

    eerste = current_node #save to find max distance
    node_dict = dict() # of the form node id: complex coordinates
    came_from = dict()
    
    tau_func = (np.exp(scaling_factor) - 1) / (np.exp(scaling_factor) + 1) # for scaling

    node_dict[current_node] = mp.mpc(0.0, 0.0) # the start node
    node_dict, came_from, nodes_added = sarkar_formula(current_node, current_node, node_dict, came_from, min_tree, tau_func,start=True)
    
    to_visit = queue.Queue() # FIFO queue

    for el in nodes_added:
        to_visit.put(el)
    
    while not to_visit.empty(): # while not empty    
        current_node = to_visit.get(0)
        node_dict, came_from, nodes_added = sarkar_formula(current_node, came_from[current_node], node_dict, came_from, min_tree, tau_func,  start=False)    
        
        for el in nodes_added:
            to_visit.put(el)


    if came_from_return is True:
        return node_dict, came_from
    
    
    #print('gaat nu de afstand printen')
    print(find_max_hyperbolic_distance(node_dict, eerste))
    
    return node_dict

def plot_hyper_embedded_tree(node_dict, came_from):
    font = {
        
        'size'   : 16}

    mpl.rc('font', **font)

    for key, value in came_from.items():
        print(key)
        x1, y1 = node_dict[key].real, node_dict[key].imag
        x2, y2 = node_dict[value].real, node_dict[value].imag
        plt.plot([x1,x2], [y1,y2], 'ro-', linewidth=3)
    
    plt.rc('text')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig('embedding_tree.png', bbox_inches='tight')
    plt.clf()
    print('gedaan')


"""
drie reguliere boom
tree = nf.make_d_regular_tree(given_depth=8,degree=3)

node_dict, came_from = hyperbolic_embed(tree, came_from_return=True)
plot_hyper_embedded_tree(node_dict, came_from)

"""

def plot_hyperbolic_graph(graph, node_dict, name):
    listedg = list(graph.edges())
 
    font = {
        
        'size'   : 16}

    mpl.rc('font', **font)
    for i, edge in enumerate(listedg):
        print(i)
        x1, y1 = node_dict[edge[0]].real, node_dict[edge[0]].imag
        x2, y2 = node_dict[edge[1]].real, node_dict[edge[1]].imag
        plt.plot([x1,x2], [y1,y2], '-bD',  c='blue', mfc='red', mec='r', linewidth=0.01, markersize=0.3)


    plt.rc('text')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(f'./hyperbolic_embedding/{name}.png', bbox_inches='tight', dpi=200)
    plt.clf()

graph = nf.make_scale_free_graph(10000, 2.1)

node_dict, came_from = hyperbolic_embed(graph, came_from_return=True)
plot_hyperbolic_graph(graph, node_dict, 'scale free')