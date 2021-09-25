import pickle
import numpy as np
import matplotlib.pyplot as plt
import coordinate_functions as cf
import osmnx as ox
"""
now make a graph horizontal axis says how far the haversine distance is and vertical the route distance/haversine distance
"""

graph_basic = ox.io.load_graphml('zw6_5km.graphml') # change file name here

with open('all_distance_dict.pk', 'rb') as file: # reloads to dijkstra to all file
    all_distance_dict = pickle.load(file)

xvalues = list()
yvalues = list()


for start_key, value in all_distance_dict.items(): # key is start node value is a dict
  for destination_key, route_distance in value.items():

    haversine_distance = cf.distance(start_key, destination_key, graph_basic) # make sure not to calculate twice 
    if haversine_distance > 0: # only useful if distance is actually there
      xvalues.append(haversine_distance)
      yvalues.append(route_distance / haversine_distance)


plt.plot(xvalues, yvalues, "ro")
plt.hlines(1.0, xmin=0, xmax=10000)
plt.xlim([0,10000])
plt.ylim([0.7,1.5])
plt.xlabel('haversine distance (m)')
plt.ylabel('route devided by true distance')
plt.title('longer route-->less direct route?')
plt.savefig('test1_5km.png')
plt.clf()

"""
not all the points will be relevant in the fitting procedure, only those who are extremal in distance and quotient will affect are minima curve so remove the unnecesary ones
"""
xlist = list() # impossible to know the length beforehand
ylist = list() 

for i in range(len(xvalues)):
  add_this = True
  for j in range(len(xvalues)):
    if yvalues[j] < yvalues[i] and xvalues[j] > xvalues[i]:
      add_this = False
      break
  if add_this:
    xlist.append(xvalues[i])
    ylist.append(yvalues[i]) 

print(xlist)
print(ylist)
"""
the below plot marks every extremal point in the sence that of each data mark none have a lower distance ratio if the data mark is located further away then the data mark in question.
Using this info one can now start fitting any strictly rising function going through all of these points or lay below them, one method might be just use every point and fit a straight line in between them. 
"""

plt.plot(xlist, ylist, "ro")
plt.hlines(1.0, xmin=0, xmax=10000)
plt.xlim([0,10000])
plt.ylim([0.7,1.5])
plt.xlabel('haversine distance (m)')
plt.ylabel('route devided by true distance')
plt.title('longer route-->less direct route? (filtered)')
plt.savefig('test2(filtered)_5km.png')
plt.clf()