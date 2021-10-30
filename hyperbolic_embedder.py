import numpy as np
import time
import networkx as nx
import fix_graph_data as fgd
import queue
import matplotlib.pyplot as plt
"""
calculates a hyperbolic embedding for a undirected graph based on it's travel time
"""

def find_max_degree(min_tree): # returns node and the degree of this node

  # first gets us the degree of every node in the tree, which gives us tuples of the type (node_id, degree) makes a numpy array out of it and then finds the max of the second argument

  node_array = np.array(min_tree.degree((min_tree.nodes())))
  max_degree = np.max(node_array[:,1])
  # find which element this max_degree is
  index = np.where(node_array[:,1]==max_degree)[0][0] # just take the first one that was encountered

  return node_array[index] # first value is node id and second one is the degree

def transformation(z_node, z_parent):
	
	z_return = (z_parent - z_node) / (1 - z_node.conjugate() * z_parent)
	return z_return

def inverse_transformation(z_node, z_child):

	z_return = (z_child + z_node) / ( 1 + z_child * z_node.conjugate())
	return z_return

"""
Example how to use transformations: 
a = complex(0.5, 0.6)
b = complex(0.1, 0.1)

b = transformation(a, b)
print(transformation(a,a)) # will map a to zero

a = inverse_transformation(0,a) # will map a again to it's original value
print(inverse_transformation(a,b)) # with this old a or new new a it will map b again to it's original value so inverse_function(function(z)) = z
""" 
pi = np.pi

def sarkar_formula(node_id, parent_id_of_node, node_dict, came_from, degree, min_tree, tau_func, start=False): # from arxiv: 1804.03329, # coordinates is numpy array of two coordinates
	
	nodes_added = set()
	degree_dos =  len(list(min_tree.neighbors(node_id)))
	if start is True:
		theta = 0
		k = 1 # starts with one since technically the parent node of the current selected node for expansion is node zero
		came_from[node_id] = parent_id_of_node # this should be parent==child if not something is wrong
		
		for child in min_tree.neighbors(node_id):
			node_dict[child] = complex(tau_func * np.cos(theta + 2 * pi * k/ degree_dos), tau_func * np.sin(theta + 2 * pi * k/ degree_dos))
			came_from[child] = node_id # the child of the node is now embeded and we want to remember te structure since we'll need it later to embed the children of the newly embedded node
			nodes_added.add(child)
			k += 1
		
		return node_dict, came_from, nodes_added

	z_node = node_dict[node_id]
	z_parent = node_dict[parent_id_of_node]
	z_parent_new = transformation(z_node, z_parent)
	theta = np.angle(z_parent_new) 
	
	k = 1

	for child in min_tree.neighbors(node_id):
		
		if child not in node_dict: # could be already embedded if parent 
			
			z_child = complex(tau_func * np.cos(theta + 2 * pi * k/ degree_dos), tau_func * np.sin(theta + 2 * pi * k/ degree_dos))
			node_dict[child] = inverse_transformation(z_node, z_child)
			came_from[child] = node_id
			nodes_added.add(child)		
			k += 1
			
	
	return node_dict, came_from, nodes_added


def hyperbolic_embed(min_tree, scaling_factor=0.1): # voor verder zie http://math.haifa.ac.il/ROVENSKI/B2.pdf p. 368
	
	# dees klopt
	#current_node = list(min_tree.nodes())[0]
	current_node, max_degree = find_max_degree(min_tree)
	
	node_dict = dict() # of the form node id: complex coordinates
	came_from = dict()
	
	tau_func = (np.exp(scaling_factor) - 1) / (np.exp(scaling_factor) + 1) # for scaling

	node_dict[current_node] = complex(0, 0) # the start node
	node_dict, came_from, nodes_added = sarkar_formula(current_node, current_node, node_dict, came_from, max_degree, min_tree, tau_func, start=True)
	
	to_visit = queue.Queue() # FIFO queue
	#list(map(to_visit.put, came_from.keys())) # puts all the new nodes at the back of the queue
	for el in nodes_added:
		to_visit.put(el)
	
	while not to_visit.empty(): # while not empty	
		current_node = to_visit.get(0)
		node_dict, came_from, nodes_added = sarkar_formula(current_node, came_from[current_node], node_dict, came_from, max_degree, min_tree, tau_func, start=False)	

		for el in nodes_added:
			to_visit.put(el)

	return node_dict

def plot_hyper_embedded_tree(node_dict):
	listx = list()
	listy = list()
	for _, z in node_dict.items():
		listx.append(z.real)
		listy.append(z.imag)
	plt.scatter(listx,listy, marker='o')
	plt.savefig('eerste_embedding.png')

#plot_hyper_embedded_tree(hyperbolic_embed(graph))


"""
# Test to test if placed in circle if start==true
print(find_max_degree(min_tree))
node_dict = dict()
node_dict[26343841] = complex(0, 0)

for node in min_tree.neighbors(26343841):
	print(node)

ret = sarkar_formula(26343841, node_dict, 4, min_tree,(np.exp(1) - 1) / (np.exp(1) + 1),start=True)
print(ret[0])
print(ret[1])
"""