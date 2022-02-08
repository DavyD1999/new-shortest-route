"""
this file has all kinds of functions mainly to help calculate the haversine distance between two nodes in a graph
"""
import numpy as np
#import hyperbolic_embedder as he # just so i can use the transformation function
from mpmath import mp

mp.prec = 1800 # sets the precision (1800) for traffic networks

def get_coordinates(node_id, graph):
    coordinate = graph.nodes[node_id]
    return coordinate['y'], coordinate['x']

def degree_to_radian(angle):
    return angle / 180 * np.pi

def haversine(couple_1, couple_2): # gets two couples of latitude longitude couples

    y1, x1, y2, x2 = degree_to_radian(np.array([couple_1[0],couple_1[1],couple_2[0],couple_2[1]]))
    
    radius = 6371000 # in m
    argument = np.sin((y1 - y2) / 2) ** 2 + np.sin((x1 - x2) / 2) ** 2 * np.cos(y1) * np.cos(y2)
    return 2 * radius * np.arcsin(np.sqrt(argument))

def distance(node_id1, node_id2, graph): # calculates the haversine distance 
  y1, x1 = get_coordinates(node_id1, graph)
  y2, x2 = get_coordinates(node_id2, graph)
    
  return haversine((y1,x1),(y2,x2))

conv_fac = np.pi / 180
R = 6357000 # earth radius
pi_over_4 = np.pi / 4
factor = R*2*np.pi/360
def euclid_distance(node_id1, node_id2, graph): #fine to use
 
  lat0, lon0 = get_coordinates(node_id1, graph) 
  lat1, lon1 = get_coordinates(node_id2, graph) 

  #deglen = 110.25 * 1000 # en in kilometer source https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/
  # earth circumference divided by 360
  a = lat1 - lat0
  b = (lon0 - lon1)*np.cos(lat0*conv_fac)
  
  return factor*np.sqrt(a*a + b*b)



def rpf_distance(current_node_id, go_to_node_id, destination_id, graph):
  
  u = np.array(get_coordinates(current_node_id, graph))
  v = np.array(get_coordinates(go_to_node_id, graph))
  d = np.array(get_coordinates(destination_id, graph))
  
  # calculate cos between ud and uv aka project our goto on the straight line between current and destination
  c = np.linalg.norm(v-d)
  a = np.linalg.norm(u-v)
  b = np.linalg.norm(u-d)

  cos_C = (a**2 + b**2 - c**2) / (2 * b * a)
  return - a * cos_C # we select on the minimum in our greedy rpf function thus have a minus here on see yujuninfocom09 paper
  # now use the cosine rule
  

def projector(tup): # uses mercator projection
  lat = tup[0]
  lon = tup[1]

  lat *= conv_fac
  lon *= conv_fac

  x = R * lon
  y = R * np.log(np.tan(pi_over_4 + lat / 2))

  return np.array((x, y))

def distance_hyperbolic(z1, z2): # give two coordinates here

   #dis1 = 2*np.arctanh(abs((z2-z1)/(1-z1.conjugate()*z2))) # this definition given in cambridge geometry and topology is wrong by a factor of two so use the one in the paper https://arxiv.org/pdf/1804.03329.pdf
    #z2_new = he.transformation(z1, z2) # will take a bit longer
    #z2_new = abs(z2_new)
    #dis2 = np.arccosh(1 + 2* z2_new**2/(1-z2_new**2))
    #assert  abs(2*np.arctanh(abs((z2-z1)/(1-z1.conjugate()*z2))) - dis2)<10**-5 # extra check on the distance
    ding1 = mp.fabs(1-z1.conjugate()*z2)
    ding2 = mp.fabs(z2-z1)

    #terug_geven = (ding1 + ding2)/(ding1 - ding2)
    #print(2*mp.atanh(abs((z2-z1)/(1-z1.conjugate()*z2))))


    return (ding1 + ding2)/(ding1 - ding2) # not actual distance but monotic function so don't care it should be mp.log(terug_geven) but log is monotic function
	#return np.arctanh(abs((z2-z1)/(1-z1.conjugate()*z2))) # p.368 Geometry SECOND EDITION D AV I D A . B R A N N A N M AT T H E W F. E S P L E N J E R E M Y J . G R AY

def distance_manhattan(node_id1, node_id2, graph):
    y1, x1 = get_coordinates(node_id1, graph)
    y2, x2 = get_coordinates(node_id2, graph)
    
    return abs(y1-y2) + abs(x1-x2)

def get_coordinate_array(node_id, graph):
    data = graph.nodes[node_id]
    return data['coordinates']

def euclidian_n_dimensions(node_id1, node_id2, graph):
    coor_node1 = get_coordinate_array(node_id1, graph)
    coor_node2 = get_coordinate_array(node_id2, graph)

    return np.sqrt(np.sum((coor_node1-coor_node2)**2))

def supremum(node_id1, node_id2, graph):
    coor_node1 = get_coordinate_array(node_id1, graph)
    coor_node2 = get_coordinate_array(node_id2, graph)

    return np.max(np.abs(coor_node1-coor_node2))

