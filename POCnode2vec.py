import numpy as np
import networkx as nx
from node2vec import Node2Vec
import time
import random
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.models

graph = nx.read_gpickle(f'./graph_pickle/Brugge.gpickle')

def create_model(city, graph, dimensions):
    """
    given a graph this creates an embedding for the nodes in that graph the parameters apart from dimensions are the same as in the paper 2002.05257, the embedding keys are 
    strings of the actual node ids
    """
    
    node2vec_paths = Node2Vec(graph, dimensions=dimensions, walk_length=80, num_walks=10, workers=4, weight_key='travel_time') # in paper dimensions could be different and q=p=1
    
    print('done with creating paths will start fitting model')
    start_time = time.time()
    model = node2vec_paths.fit(window=5, min_count=0, batch_words=4)
    print(time.time()-start_time)
    print('starts saving')
    
    model.save(f'./node2vec_models/{city}.model')

    word_vectors = model.wv

    word_vectors.save(f'./node2vec_models/{city}.wordvectors')

# for the moment chose landmarks randomly
#create_model('Brugge', graph, dimensions=64)
#model.wv.vocab


def calculate_shortest_path(graph, amount_of_landmarks_training, amount_of_landmarks_test):
    """
    calculates shortest path for a set of given landmarks to all other nodes
    """
    node_list = list(graph.nodes())

    random_generated = random.sample(node_list, k=amount_of_landmarks_training + amount_of_landmarks_test) # sample gives us unique nodes
    
    landmark_list_training = random_generated[:amount_of_landmarks_training] # make split between test and train
    landmark_list_test = random_generated[amount_of_landmarks_training:]
    
    distance_list_training = list()
    for i, landmark in enumerate(landmark_list_training):
        distances, _ = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        distance_list_training.append(distances)

    distance_list_test = list()
    for i, landmark in enumerate(landmark_list_test):
        distances, _ = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        distance_list_test.append(distances)

    return landmark_list_training, distance_list_training, landmark_list_test, distance_list_test

landmark_list_training, distance_list_training, landmark_list_test, distance_list_test = calculate_shortest_path(graph, 50, 10)
embedding = KeyedVectors.load('./node2vec_models/Brugge.wordvectors', mmap='r')

def minus(vec1, vec2):
    return vec1-vec2

def concatenation(vec1, vec2):
    return [vec1,vec2]

def average(vec1, vec2):
    return (vec1 + vec2) / 2

def hadamard(vec1, vec2):
    return vec1 * vec2

def transformer(embedding, landmark_list, distance_list, operation):
    """
    given a certain operation it transforms the distances to a manageable input for the neural network, each row of input array should be a sample and the output should be the 
    total traveltime
    """  
    input_array = list()
    output_array =list()
    
    for i, landmark in enumerate(landmark_list):
        for node, distance in distance_list[i].items():
            input_array.append(operation(embedding[str(landmark)], embedding[str(node)]))
            output_array.append(distance)

    return np.array(input_array, dtype=np.float16), np.array(output_array, dtype=np.float16)
            

def neural_network(input_training, output_training, input_test, output_test, graph):
    # starting with - operation

    model = tf.keras.Sequential([
        tf.keras.layers.ReLU(), # input layer has relu activation function too
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='softplus')
    ])
    for i in range(1,100):
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001*i),
            loss='mse',
            metrics=[tf.keras.losses.MeanAbsoluteError()
        ]
        )
    
        model.fit(input_training, output_training, epochs=4, workers=4)
    
        print(model.evaluate(input_test, output_test, workers=4))

    
    #print(model.predict(input_test[:100]))
    #print(output_test[:100])

    
    
input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, minus)
input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, minus)

neural_network(input_training, output_training, input_test, output_test, graph)