import numpy as np

inf = np.inf
# THIS FILE IS  DEPRECATED DO NOT USE 
"""
def get_edge_length(id1, id2, graph):
  
  edge_val = list()
  try: # it might be a one way street
    edge_val += list(graph[id1][id2].values())
  except KeyError:
    pass
  try:
    edge_val += list(graph[id2][id1].values())
  except KeyError:
    pass # no actual path seems to exist 
  
  edge_length = inf
  for edge in edge_val: # if two roads or more roads do connect one chose the shortest one of both
    if edge_length > edge.get('length'):
      edge_length = edge.get('length')
  if edge_length == inf:
    print('das toch geen buur')
  return edge_length

"""