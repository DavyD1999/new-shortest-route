import numpy as np
import queue
import matplotlib.pyplot as plt
import cmath
import random
import networkx as nx
import node_functions as nf
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
  bfs_queue.put(furthest)
  
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
	degree =  min_tree.degree[node_id]
	if start is True:
		theta = 0
		k = 0 # starts at zero here since we want a full circle
		came_from[node_id] = parent_id_of_node # this should be parent==child if not something is wrong
		
		for child in min_tree.neighbors(node_id):
			new_val  = tau_func * complex(np.cos(theta + 2 * pi * k/ degree),  np.sin(theta + 2 * pi * k/ degree))
			val = cmath.exp(1j*(theta + 2 * pi * k / degree)) * tau_func

			#print(new_val)
			assert 2 * pi * k/ degree < 2 * pi - 0.001 # here willl be full circle
			assert abs(val - new_val) < 0.0001
			node_dict[child] = val
			# print(abs(2*np.arctanh(abs((new_val)/(1))))) # check to look if right distance embedded
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
			#new_val = tau_func * complex(np.cos(theta + 2 * pi * k/ degree), np.sin(theta + 2 * pi * k/ degree))
			
			val = cmath.exp(1j*(theta + 2 * pi * k / degree)) * tau_func
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


def hyperbolic_embed(min_tree, scaling_factor=1.1, came_from_return=False): # voor verder zie http://math.haifa.ac.il/ROVENSKI/B2.pdf p. 368
	
	assert scaling_factor > 1

	current_node = find_start_node(min_tree)
	
	node_dict = dict() # of the form node id: complex coordinates
	came_from = dict()
	
	tau_func = (np.exp(scaling_factor) - 1) / (np.exp(scaling_factor) + 1) # for scaling

	node_dict[current_node] = complex(0, 0) # the start node
	node_dict, came_from, nodes_added = sarkar_formula(current_node, current_node, node_dict, came_from, min_tree, tau_func, start=True)
	
	to_visit = queue.Queue() # FIFO queue

	for el in nodes_added:
		to_visit.put(el)
	
	while not to_visit.empty(): # while not empty	
		current_node = to_visit.get(0)
		node_dict, came_from, nodes_added = sarkar_formula(current_node, came_from[current_node], node_dict, came_from, min_tree, tau_func, start=False)	

		for el in nodes_added:
			to_visit.put(el)

	if came_from_return is True:
		return node_dict, came_from
	
	return node_dict

def plot_hyper_embedded_tree(node_dict, came_from):

	for key, value in came_from.items():
		x1, y1 = node_dict[key].real, node_dict[key].imag
		x2, y2 = node_dict[value].real, node_dict[value].imag
		plt.plot([x1,x2], [y1,y2], 'ro-')
    
	plt.savefig('embedding_test.png')
	plt.clf()
	print('gedaan')

#tree = nf.make_d_regular_tree(given_depth=8,degree=3)

#node_dict, came_from = hyperbolic_embed(tree, came_from_return=True)
#plot_hyper_embedded_tree(node_dict, came_from)
