import numpy as np
import networkx as nx
from node2vec import Node2Vec
import time
import random
import tensorflow as tf
from gensim.models import KeyedVectors
import sklearn
from sklearn import preprocessing
import coordinate_functions as cf
import pickle
import gc
from sklearn.decomposition import PCA



def create_model(city, graph, dimensions, weight='travel_time'):
    """
    given a graph this creates an embedding for the nodes in that graph the parameters apart from dimensions are the same as in the paper 2002.05257, the embedding keys are 
    strings of the actual node ids
    """
    
    node2vec_paths = Node2Vec(graph, dimensions=dimensions, walk_length=107, num_walks=17, workers=4, 
       weight_key=weight, q=1, p=1,seed=42) # in paper dimensions could be different and q=p=1
    
    print('done with creating paths will start fitting model')
    start_time = time.time()
    model = node2vec_paths.fit(window=8, min_count=0, batch_words=4, seed=42, workers=4)
    print(time.time()-start_time)
    print('starts saving')
    
    model.save(f'./node2vec_models/{city}.model')

    word_vectors = model.wv

    word_vectors.save(f'./node2vec_models/{city}.wordvectors')

# create_model('Brugge', graph, dimensions=128) 

def calculate_shortest_path(graph, amount_of_landmarks_training, amount_of_landmarks_test):
    """
    calculates shortest path for a number of landmarks at this moment the landmarks are chosen randomly
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

def minus(vec1, vec2): # not symmetric so weird
    return vec1-vec2

def concatenation(vec1, vec2):
    vec3 = np.append(vec1, vec2)
    return vec3

def average(vec1, vec2):
    return (vec1 + vec2) / 2

def hadamard(vec1, vec2):
    return vec1 * vec2

def add_inverse_travel_time(graph):
    for node1, node2, travel_time in graph.edges(data='travel_time'): # adds inverse traveltime to graph
        graph[node1][node2]['inverse_travel_time'] = 1/travel_time

    return graph

def transformer(embedding, landmark_list, distance_list, operation, graph ,double_use=False, add_distance=False):
    """
    given a certain operation it transforms the distances to a manageable input for the neural network, each row of input array should be a sample and the output should be the 
    total traveltime
    """  
    input_array = list()
    output_array =list()
    
    for i, landmark in enumerate(landmark_list):
        for node, distance in distance_list[i].items():
            if node != landmark: # since this would automatically give us a infinite relative error
                to_add = operation(embedding[str(landmark)], embedding[str(node)])
                if add_distance is True: # if we want to add the euclidian distance as extra input
                    euclid_distance = cf.euclid_distance(landmark, node, graph)
                    np.append(to_add, euclid_distance)
                input_array.append(operation(embedding[str(landmark)], embedding[str(node)]))
                output_array.append(distance)
                if double_use is True: # use the data twice since the graph is 'symmetric' which means A to B takes as long as B to A
                    input_array.append(operation(embedding[str(node)], embedding[str(landmark)]))
                    output_array.append(distance)
        
    input_array, output_array = sklearn.utils.shuffle(input_array, output_array, random_state=42) # we want to shuffle because for each landmark the distances were in ascending order which is 
    input_array = np.array(input_array, dtype=np.float16)
    output_array = np.array(output_array, dtype=np.float16)
    return input_array, output_array
            
def neural_network(input_training, output_training, input_test, output_test, city):

    scaler = preprocessing.StandardScaler() # normalize it here because normalisation in the model does not really work since tensorflow 2.2.0  
    input_training = scaler.fit_transform(input_training)

    input_test = scaler.transform(input_test) # same normalization on the test data

    pca = PCA(n_components=150)
    input_training = pca.fit_transform(input_training)
    input_test = pca.transform(input_test)

    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.LayerNormalization(), # normalize the layer
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(75, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42), # dropout should help prevent overfitting
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )
          
    print(model.fit(input_training, output_training, epochs=4, workers=8, verbose=1, use_multiprocessing=True))

    print(model.evaluate(input_test, output_test, verbose=2))

    
    # save needed stuff for later in the A* function with this input
    with open(f'./saved_scalers/{city}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(f'./saved_pca/{city}.pkl', 'wb') as f:
        pickle.dump(pca, f)

        
    model.save(f'./NNcities/{city}.h5') # load it in with tf.keras.models.load_model('saved_model/my_model')
    
    tf.keras.backend.clear_session()

name_list = ['Manhattan','Brugge','New Dehli','Nairobi', 'Rio de Janeiro']

def evaluate_network(city, trained_city):
    model = tf.keras.models.load_model(f'./NNcities/{trained_city}.h5', compile=False) # false compile because of bug
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )
    with open(f'./saved_scalers/{city}.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(f'./saved_pca/{city}.pkl', 'rb') as f:
        pca = pickle.load(f)

    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
    #create_model(city, graph, dimensions=128) # create and save model if not created yet
    samples = 150000 // (graph.number_of_nodes()-1) # we want with the double function later end up with approx 700000million of samples
    _, _, landmark_list_test, distance_list_test = calculate_shortest_path(graph, samples, samples//10) # 100 training 10 test cities
    
    embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')
    
    input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, 
         concatenation, graph,add_distance=True)

    input_test = scaler.transform(input_test)
    input_test = pca.transform(input_test)
    print(city)
    print(model.evaluate(input_test, output_test, verbose=2))




def train_network(city):
    print(city)
    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')

    #create_model(city, graph, dimensions=128) # create and save model if not created yet
    samples = 150000 // (graph.number_of_nodes()-1) # we want with the double function later end up with approx 700000million of samples
    landmark_list_training, distance_list_training, landmark_list_test, distance_list_test = calculate_shortest_path(graph, samples, samples//10) # 100 training 10 test cities
    
    embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')
    
    input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, concatenation, graph ,double_use=True, add_distance=True) # big file
    input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, 
         concatenation, graph,add_distance=True) # big file, FOR TEST DATA DOULB

    neural_network(input_training, output_training, input_test, output_test, city)

    gc.collect()

#train_network('New Dehli')
#train_network('Brugge')
#evaluate_network('New Dehli', 'Brugge')

for city in name_list:
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    create_model(city, graph, dimensions=128, weight='travel_time')
    
