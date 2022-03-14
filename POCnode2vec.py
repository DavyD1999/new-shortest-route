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
import osmnx as ox
import matplotlib as mpl
import matplotlib.cm as cm

def create_model(city, graph, dimensions, weight):
    """
    given a graph this creates an embedding for the nodes in that graph the parameters apart from dimensions are the same as in the paper 2002.05257, the embedding keys are 
    strings of the actual node ids
    """


    for node1, node2, _ in graph.edges(data=weight): # adds inverse traveltime to graph
        graph[node1][node2][weight] # just make sure the weight is actually in the graph
        break
        
    
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

def calculate_shortest_path(graph, amount_of_landmarks_training, amount_of_landmarks_test, amount_of_landmarks_validation=0, recalculate_weights=False):
    """
    calculates shortest path for a number of landmarks at this moment the landmarks are chosen randomly
    """
    node_list = list(graph.nodes())
    
    random_generated = random.sample(node_list, k=amount_of_landmarks_training + amount_of_landmarks_test + amount_of_landmarks_validation) # sample gives us unique nodes
    
    landmark_list_training = random_generated[:amount_of_landmarks_training] # make split between test and train
    landmark_list_test = random_generated[amount_of_landmarks_training:amount_of_landmarks_training + amount_of_landmarks_test]
    landmark_list_validation = random_generated[amount_of_landmarks_training + amount_of_landmarks_test:]

    distance_list_training = list()
    maxval = 1

    if recalculate_weights is True: # possible not every edge will be used so make sure at least has a probability
        for u, v in graph.edges():
            graph[u][v]['transition_probability'] = 1
            
    for  j, landmark in enumerate(landmark_list_training):
        # print(j/len(landmark_list_training))
        distances, paths = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        distance_list_training.append(distances)

        if recalculate_weights is True: #
            for value in paths.values():
                for x in range(len(value)-1): # -1 cause we can't count too for out of the list range
                    graph[value[x]][value[x+1]]['transition_probability'] += 0.1

                    if graph[value[x]][value[x+1]]['transition_probability'] > maxval:
                        maxval = graph[value[x]][value[x+1]]['transition_probability']
                    
                        
                        
    if recalculate_weights is True:        
        multi_graph = nx.MultiGraph(graph)
        # edge_color = ox.plot.get_edge_colors_by_attr(multi_graph, 'transition_probability') # uses attribute to show how big transition probability
    
        norm = mpl.colors.Normalize(vmin=1, vmax=maxval, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        ec = [mapper.to_rgba(data['transition_probability']) for u, v, key, data in multi_graph.edges(keys=True, data=True)]
        fig, ax = ox.plot.plot_graph(multi_graph,node_size=0, show=False ,edge_color=ec, save=False)
        #ax.set_facecolor('w')
        cb = fig.colorbar(mapper, ax=ax, orientation='horizontal')
        cb.set_label('transition weights', fontsize = 20)
        fig.savefig('./test3.png',dpi=500)
        
    distance_list_test = list()
    for landmark in landmark_list_test:
        distances, _ = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        distance_list_test.append(distances)

    if amount_of_landmarks_validation == 0:
        return landmark_list_training, distance_list_training, landmark_list_test, distance_list_test

    distance_list_validation = list()
    for landmark in landmark_list_validation:
        distances, _ = nx.single_source_dijkstra(graph, landmark, weight='travel_time')
        distance_list_validation.append(distances)

    return landmark_list_training, distance_list_training, landmark_list_test, distance_list_test, landmark_list_validation, distance_list_validation

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

def transformer(embedding, landmark_list, distance_list, operation, graph ,add_distance, double_use=False, ): # add distance always needs new keyword
    """
    given a certain operation it transforms the distances to a manageable input for the neural network, each row of input array should be a sample and the output should be the 
    total traveltime
    """  
    input_array = list()
    output_array =list()
    
    for i, landmark in enumerate(landmark_list):
        for node, travel_time in distance_list[i].items():
            if node != landmark: # since this would automatically give us a infinite relative error
                to_add = operation(embedding[str(landmark)], embedding[str(node)])
                if add_distance is True: # if we want to add the euclidian distance as extra input
                    euclid_distance = cf.euclid_distance(landmark, node, graph)
                    to_add = np.append(to_add, euclid_distance)
                input_array.append(to_add)
                output_array.append(travel_time)
                if double_use is True: # use the data twice since the graph is 'symmetric' which means A to B takes as long as B to A
                    to_add = operation(embedding[str(node)], embedding[str(landmark)])
                    if add_distance is True: # if we want to add the euclidian distance as extra input
                        euclid_distance = cf.euclid_distance(landmark, node, graph)
                    to_add = np.append(to_add, euclid_distance)
                    input_array.append(to_add)
                    output_array.append(travel_time)
        
    input_array, output_array = sklearn.utils.shuffle(input_array, output_array, random_state=42) # we want to shuffle because for each landmark the distances were in ascending order which is 
    input_array = np.array(input_array, dtype=np.float16)
    output_array = np.array(output_array, dtype=np.float16)
    return input_array, output_array
            
def neural_network(input_training, output_training, input_test, output_test,input_validation, output_validation ,city):

    scaler = preprocessing.StandardScaler() # normalize it here because normalisation in the model does not really work since tensorflow 2.2.0  
    input_training = scaler.fit_transform(input_training)
    
    input_test = scaler.transform(input_test) # same normalization on the test data
    input_validation = scaler.transform(input_validation)
    #pca = PCA(n_components=250)
    #input_training = pca.fit_transform(input_training)
    #input_test = pca.transform(input_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.LayerNormalization(), # normalize the layer
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42), # dropout should help prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.2, seed=42),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )
          
    print(model.fit(input_training, output_training, epochs=10, workers=8, verbose=2,validation_data=(input_validation, output_validation), batch_size=128 ,use_multiprocessing=True, callbacks=[callback]))
    
    print(model.evaluate(input_test, output_test, verbose=2))

    
    # save needed stuff for later in the A* function with this input
    with open(f'./saved_scalers/{city}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    #with open(f'./saved_pca/{city}.pkl', 'wb') as f:
        #pickle.dump(pca, f)

        
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


def train_network(city, inverse=False, create_model_now=False, recalculate_weights=True):
    """
    train network for 1 city
    """
    print(city)
    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
    samples = 350000 // (graph.number_of_nodes()-1) # we want with the double function later end up with approx 700000million of samples
    landmark_list_training, distance_list_training, landmark_list_test, distance_list_test, landmark_list_validation, distance_list_validation = calculate_shortest_path(graph, samples, samples//5, samples//5, recalculate_weights=recalculate_weights) # 100 training 10 test cities
    
    embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')

    if create_model_now is True:
        assert inverse != True or recalculate_weights != True, "not allowed"
        if inverse is True:
            graph = add_inverse_travel_time(graph)
            create_model(city, graph, dimensions=128, weight='inverse_travel_time')

        elif recalculate_weights is True:
            # the recalculated weights were done if necessary in de calculate shortest path function this modifies the graph (pointerwise like c++)
            create_model(city, graph, dimensions=128, weight='transition_probability')
            
        else:
            create_model(city, graph, dimensions=128, weight='travel_time') # create and save model if not created yet
        gc.collect()
        
    input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, concatenation, graph ,double_use=True, add_distance=True) # big file
    input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, 
         concatenation, graph,add_distance=True) # big file, FOR TEST DATA DOULB
    input_validation, output_validation = transformer(embedding, landmark_list_validation, distance_list_validation, 
         concatenation, graph,add_distance=True)

    neural_network(input_training, output_training, input_test, output_test, input_validation, output_validation, city)
    
    gc.collect()

def NN_multiple_cities(input_training, output_training, input_test, output_test, input_validation, output_validation, name):
    """
    actually trains for multiple cities
    """
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.LayerNormalization(), # normalize the layer
            tf.keras.layers.Dropout(0.3, seed=42),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3, seed=42), # dropout should help prevent overfitting
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3, seed=42),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3, seed=42),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)
    
    print(model.fit(input_training, output_training, epochs=10, workers=8, verbose=1,  validation_data=(input_validation, output_validation), use_multiprocessing=True, callbacks=[callback], batch_size=128))
    print(model.evaluate(input_test, output_test, verbose=2))

    model.save(f'./NNcities/{name}.h5')

def train_network_cities(cities, name):
    """
    combines multiple cities 
    """

    tot_input_training, tot_output_training, tot_input_test, tot_output_test, tot_input_validation, tot_output_validation = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    first = True
    
    for city in cities:
        print(city)
        graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
        samples = 20 # we want with the double function later end up with approx 700000million of samples
        
        landmark_list_training, distance_list_training, landmark_list_test, distance_list_test, landmark_list_validation, distance_list_validation = calculate_shortest_path(graph, samples, samples//5, samples//5) # 100 training 10 test cities

        embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')
        
        input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, concatenation, graph ,double_use=True, add_distance=True) # big file
        input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, 
             concatenation, graph,add_distance=True) # big file, FOR TEST DATA DOULB
        input_validation, output_validation = transformer(embedding, landmark_list_validation, distance_list_validation, 
             concatenation, graph,add_distance=True)

        if first is False:
            tot_input_training = np.concatenate((tot_input_training, input_training))
            tot_input_test = np.concatenate((tot_input_test, input_test))
            tot_output_training = np.concatenate((tot_output_training, output_training))
            tot_output_test = np.concatenate((tot_output_test, output_test))
            tot_input_validation = np.concatenate((tot_input_validation, input_validation))
            tot_output_validation = np.concatenate((tot_output_validation, output_validation))
            
        else:
            tot_input_training = input_training
            tot_input_test = input_test
            tot_output_training = output_training
            tot_output_test = output_test
            tot_input_validation = input_validation
            tot_output_validation = output_validation
        
        gc.collect()

        first = False

    scaler = preprocessing.StandardScaler() # normalize it here because normalisation in the model does not really work since tensorflow 2.2.0  
    tot_input_training = scaler.fit_transform(tot_input_training)
    tot_input_test = scaler.transform(tot_input_test) # same normalization on the test data
    tot_input_validation = scaler.transform(tot_input_validation)
    
    pca = PCA(n_components=256)
    tot_input_training = pca.fit_transform(tot_input_training)
    tot_input_test = pca.transform(tot_input_test)
    tot_input_validation = pca.transform(tot_input_validation)

    tot_input_training, tot_output_training = sklearn.utils.shuffle(tot_input_training, tot_output_training, random_state=42)
    tot_input_test, tot_output_test = sklearn.utils.shuffle(tot_input_test, tot_output_test, random_state=42) # not that necessary for test data or validation data
    tot_input_validation, tot_output_validation = sklearn.utils.shuffle(tot_input_validation, tot_output_validation, random_state=42)

    NN_multiple_cities(tot_input_training, tot_output_training, tot_input_test, tot_output_test, tot_input_validation, tot_output_validation, name)

    
def evaluate_network_new_transformations(graph_name, NN_name, pca=False, add_distance=True):
    """
    model calculated and then applied to new normalised and pca'd dataset
    """
    graph = nx.read_gpickle(f'./graph_pickle/{graph_name}.gpickle')
    samples = 10 # we want with the double function later end up with approx 700000million of samples
        
    landmark_list_training, distance_list_training, _, _ = calculate_shortest_path(graph, samples, 0)# 100 training 10 test cities
    embedding = KeyedVectors.load(f'./node2vec_models/{graph_name}.wordvectors', mmap='r')

    input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, concatenation, graph ,double_use=False, add_distance=add_distance)

    scaler = preprocessing.StandardScaler()
    input_training = scaler.fit_transform(input_training)

    if pca is True:
        pca = PCA(n_components=256)
        input_training = pca.fit_transform(input_training)

    model = tf.keras.models.load_model(f'./NNcities/{NN_name}.h5', compile=False) # false compile because of bug
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )

    print(model.evaluate(input_training, output_training, verbose=2))
    
    
#train_network('New Dehli')
#train_network('Brugge')
#evaluate_network('New Dehli', 'Brugge')

# train_network_cities(['Manhattan', 'Brugge'], 'Manhattan Brugge')
#evaluate_network_new_transformations('Manhattan', 'Manhattan Brugge', pca=True, add_distance=True)

def train_network_CNN(city):
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
    samples = 40 # we want with the double function later end up with approx 700000million of samples
    
    landmark_list_training, distance_list_training, landmark_list_test, distance_list_test, landmark_list_validation, distance_list_validation = calculate_shortest_path(graph, samples, samples//5, samples//5) # 100 training 10 test cities

    embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')

    # distance left out here for ez purpouse
    input_training, output_training = transformer(embedding, landmark_list_training, distance_list_training, concatenation, graph ,double_use=True, add_distance=False) # big file
    input_test, output_test = transformer(embedding, landmark_list_test, distance_list_test, 
         concatenation, graph,add_distance=False) # big file, FOR TEST DATA DOULB
    input_validation, output_validation = transformer(embedding, landmark_list_validation, distance_list_validation, 
         concatenation, graph,add_distance=False)

    model = tf.keras.Sequential([
            tf.keras.layers.Reshape((16, 16, 1), input_shape=(256,)),
            tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1)), # 1 filters
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2, seed=42),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2, seed=42),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2, seed=42),
            tf.keras.layers.Dense(1, activation='relu')
        ])
    print(model.summary())
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
        loss='mse', 
        metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)
    
    print(model.fit(input_training, output_training, epochs=10, workers=8, verbose=1,  validation_data=(input_validation, output_validation), use_multiprocessing=True, callbacks=[callback], batch_size=128))
    print(model.evaluate(input_test, output_test, verbose=2))

    model.save(f'./NNcities/{city}.h5')
    

#train_network_CNN('Manhattan')
#evaluate_network_new_transformations('Brugge', 'Manhattan',add_distance=False)

# train_network('Manhattan', inverse=True) # 10.16666030883789
#train_network('Manhattan', inverse=False) #10.130
train_network('New Dehli', inverse=False, create_model_now=False, recalculate_weights=True) #  9.8444824218
#train_network('Brugge', inverse=False) # 9.399
#train_network('New Dehli', inverse=True) #11.31
#train_network('New Dehli', inverse=False) 13.698349952697754
