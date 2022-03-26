import matplotlib.pyplot as plt
from fix_graph_data import get_weigted_average_velocity
import NNforSearch
import matplotlib as mpl
import a_star
import numpy as np
import tensorflow as tf
import random
import pickle
import networkx as nx
from gensim.models import KeyedVectors
import time
import gravity_pressure as gp

mpl.style.use('tableau-colorblind10')
np.random.seed(42)
random.seed(42)
font = { 'size'   : 16}
mpl.rc('font', **font)

def prepare_plot_errorbar(computed_list, weight_path,  binsize):

    stretch_astar = np.array(computed_list) / weight_path    
    
    values, base = np.histogram(weight_path, bins=np.arange(start=0,stop=max(weight_path) + binsize, step=binsize)) # + stepsize makes sure we actually get a last bin too
    base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last value since this one concludes the bin

    indices = np.digitize(weight_path, base) - 1
 
    average_stretch = np.zeros(len(values))
    arrived = np.zeros(len(values))
    
    for k, element in enumerate(stretch_astar):
        average_stretch[indices[k]] += element
        arrived[indices[k]] += 1
    
    
    average_stretch /= arrived   # to actually be an average we still need to devide by the number in each bin that arrived  
    
    squared_sum = np.zeros(len(average_stretch))
    n = np.zeros(len(average_stretch)) # to keep track how many are in each bin
        
    for k, element in enumerate(stretch_astar):
        n[indices[k]] += 1
        squared_sum[indices[k]] += (element - average_stretch[indices[k]]) ** 2 

    standard_dev_on_mean_stretch = (squared_sum/(n * (n-1))) ** (0.5)

    xvals = base[:-1] + binsize / 2

    average_stretch[average_stretch == 0] = np.nan
    return xvals, average_stretch, standard_dev_on_mean_stretch
    

def make_graphs_NN(city, number_of_paths):
    model = tf.keras.models.load_model(f'./semester2/NNcities/{city}.h5', compile=False) # false compile because of bug
    
    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
            loss='mse', 
            metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
        )
    with open(f'./semester2/saved_scalers/{city}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    embedding = KeyedVectors.load(f'./semester2/node2vec_models/{city}.wordvectors', mmap='r')
    
    node_list = list(graph.nodes())
    
    how_many = 2 * number_of_paths
    random_generated = random.sample(node_list, k=how_many)
    start_nodes = random_generated[:number_of_paths]
    end_nodes = random_generated[number_of_paths:]
    
    len_astar, teller_astar_list = list(), list()
    len_NN, teller_NN = list(), list()
    weight_path = list()

    time_normal = 0
    time_NN = 0
    
    for i in range(len(start_nodes)):
        average_vel = get_weigted_average_velocity(graph)

        start_time = time.time()
        a = a_star.A_star_priority_queue(start_nodes[i], end_nodes[i], graph, average_vel, return_counter = True)
        time_normal += time.time() - start_time
            
        len_astar.append(a[0])
        teller_astar_list.append(a[1])

        start_time = time.time()
        b = NNforSearch.A_star_priority_queue_NN(start_nodes[i], end_nodes[i], graph, scaler, embedding, model, return_counter=True)
        time_NN = time.time() - start_time
        
        len_NN.append(b[0])
        teller_NN.append(b[1])
    
        weight_path.append(nx.shortest_path_length(graph, start_nodes[i], end_nodes[i], weight='travel_time', method='dijkstra'))    
    
        print(i)
    
    xvals_astar, average_stretch_astar, standard_dev_on_mean_stretch_astar = prepare_plot_errorbar(len_astar, weight_path,  binsize=150)                                    
    plt.ylabel('Gemiddelde rek')
    plt.xlabel('Snelste reistijd (s)')
    plt.errorbar(xvals_astar, average_stretch_astar, yerr=standard_dev_on_mean_stretch_astar, linewidth=3, elinewidth=3, linestyle='--', capsize=5,barsabove=True, label='Hemelsbrede afstand met snelheid')
    
    
    xvals_NN, average_stretch_NN, standard_dev_on_mean_stretchNN = prepare_plot_errorbar(len_NN, weight_path,  binsize=150)                                    
    
    plt.errorbar(xvals_NN, average_stretch_NN, yerr=standard_dev_on_mean_stretchNN, linewidth=3,elinewidth=3, linestyle='-', capsize=5,barsabove=True, label='Neuraal netwerk')
    plt.legend()
    plt.savefig(f'./semester2/NN_files/rek_NN_{city}.png', bbox_inches="tight")
    plt.clf()
    
    
    bins=np.histogram(np.hstack((teller_astar_list,teller_NN)), bins=5)[1] #get the bin edges
    
    plt.hist(teller_astar_list, bins=bins)
    print(sum(teller_astar_list))
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    
    plt.savefig(f'./semester2/NN_files/aantal_expansies_aster_{city}.png', bbox_inches="tight")
    plt.clf()
    
    plt.hist(teller_NN, bins=bins)
    print(sum(teller_NN))
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    plt.savefig(f'./semester2/NN_files/aantal_expansies_NN_{city}.png', bbox_inches="tight")
    plt.clf()

    print(time_normal/number_of_paths)
    print(time_NN/number_of_paths)    

def make_graphs_logistic(city, number_of_paths, number_of_landmarks):

    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    
    node_list = list(graph.nodes())
    
    how_many = 2 * number_of_paths
    random_generated = random.sample(node_list, k=how_many)
    start_nodes = random_generated[:number_of_paths]
    end_nodes = random_generated[number_of_paths:]
    
    len_astar, teller_astar_list = list(), list()
    len_ML, teller_ML = list(), list()
    weight_path = list()

    time_normal = 0
    time_ML = 0

    average_vel = get_weigted_average_velocity(graph)
    clf, scaler = gp.add_amount_of_visited_weights(graph, 20)
    
    for i in range(len(start_nodes)):
        
        start_time = time.time()
        a = a_star.A_star_priority_queue(start_nodes[i], end_nodes[i], graph, average_vel, return_counter = True)
        time_normal += time.time() - start_time
            
        len_astar.append(a[0])
        teller_astar_list.append(a[1])

        start_time = time.time()
        b = gp.priority_queue_new_evaluation_function(start_nodes[i], end_nodes[i], graph, clf, scaler=scaler,ratio_travelled=True ,return_counter=True)

        time_ML += time.time() - start_time
        
        len_ML.append(b[0])
        teller_ML.append(b[2])
    
        weight_path.append(nx.shortest_path_length(graph, start_nodes[i], end_nodes[i], weight='travel_time', method='dijkstra'))    
    
        print(i)
    
    xvals_astar, average_stretch_astar, standard_dev_on_mean_stretch_astar = prepare_plot_errorbar(len_astar, weight_path,  binsize=150)                                    
    plt.ylabel('Gemiddelde rek')
    plt.xlabel('Snelste reistijd (s)')
    plt.errorbar(xvals_astar, average_stretch_astar, yerr=standard_dev_on_mean_stretch_astar, linewidth=3, elinewidth=3, linestyle='--', capsize=5,barsabove=True, label='Hemelsbrede afstand met snelheid')
    
    
    xvals_ML, average_stretch_ML, standard_dev_on_mean_stretchML = prepare_plot_errorbar(len_ML, weight_path,  binsize=150)                                    
    
    plt.errorbar(xvals_ML, average_stretch_ML, yerr=standard_dev_on_mean_stretchML, linewidth=3,elinewidth=3, linestyle='-', capsize=5,barsabove=True, label='logistische regressie')
    plt.legend()
    plt.savefig(f'./semester2/ML_files/rek_ML_{city}.png', bbox_inches="tight")
    plt.clf()
    
    
    bins=np.histogram(np.hstack((teller_astar_list,teller_ML)), bins=5)[1] #get the bin edges
    
    plt.hist(teller_astar_list, bins=bins)
    print(sum(teller_astar_list))
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    
    plt.savefig(f'./semester2/ML_files/aantal_expansies_aster_{city}.png', bbox_inches="tight")
    plt.clf()
    
    plt.hist(teller_ML, bins=bins)
    print(sum(teller_ML))
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    plt.savefig(f'./semester2/ML_files/aantal_expansies_ML_{city}.png', bbox_inches="tight")
    plt.clf()

    print(time_normal/number_of_paths)
    print(time_ML/number_of_paths)

def prepare_plot_errorbar_logistic(computed_list, weight_path,  binsize):

    stretch_astar = np.array(computed_list) / weight_path    
    
    values, base = np.histogram(weight_path, bins=np.arange(start=0,stop=max(weight_path) + binsize, step=binsize)) # + stepsize makes sure we actually get a last bin too
    base[len(base)-1] = base[len(base)-1] + 0.001 # else digitize will give a wrong index for the last value since this one concludes the bin

    indices = np.digitize(weight_path, base) - 1
 
    average_stretch = np.zeros(len(values))
    arrived = np.zeros(len(values))
    
    for k, element in enumerate(stretch_astar):
        average_stretch[indices[k]] += element
        arrived[indices[k]] += 1
    
    average_stretch /= arrived   # to actually be an average we still need to devide by the number in each bin that arrived  
    
    squared_sum = np.zeros(len(average_stretch))
    n = np.zeros(len(average_stretch)) # to keep track how many are in each bin
        
    for k, element in enumerate(stretch_astar):
        n[indices[k]] += 1
        squared_sum[indices[k]] += (element - average_stretch[indices[k]]) ** 2 

    standard_dev_on_mean_stretch = (squared_sum/(n * (n-1))) ** (0.5)

    xvals = base[:-1] + binsize / 2

    average_stretch[average_stretch == 0] = np.nan
    return xvals, average_stretch, standard_dev_on_mean_stretch
       
# make_graphs_NN('Brugge', 50)

make_graphs_logistic('Nairobi', 50, 20)