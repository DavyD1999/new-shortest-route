import pickle
import tensorflow as tf
import NNforSearch as nns
import networkx as nx
from gensim.models import KeyedVectors
import time
"""
test for saving the scaler
"""
"""
with open('./saved_scalers/parrot.pkl', 'wb') as f:
    pickle.dump(mylist, f)

with open('./saved_scalers/parrot.pkl', 'rb') as f:
    mynewlist = pickle.load(f)

print(mynewlist)
"""


model = tf.keras.models.load_model('./NNcities/Manhattan.h5', compile=False)

with open('./saved_scalers/Manhattan.pkl', 'rb') as f:
    myscaler = pickle.load(f)


graph = nx.read_gpickle(f'./graph_pickle/Manhattan.gpickle')
    
embedding = KeyedVectors.load(f'./node2vec_models/Manhattan.wordvectors', mmap='r')

nodes = list(graph.nodes())
start = time.time()
som = 0
for i in range(0,50):
    a = nns.A_star_priority_queue_NN(nodes[i+10], nodes[5*i+2000], graph, myscaler, embedding, model)
    #b = nx.shortest_path_length(graph, nodes[i+10], nodes[5*i+2000], weight='travel_time')
    #som += abs(a-b)
    print(i)
    
print((time.time()-start)/50)
print(som/50)