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
    

def make_graphs(city, number_of_paths):
    model = tf.keras.models.load_model(f'./NNcities/{city}.h5', compile=False) # false compile because of bug
    
    model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001*46), # batch size is standard 32 
            loss='mse', 
            metrics=[tf.keras.losses.MeanAbsolutePercentageError()]
        )
    with open(f'./saved_scalers/{city}.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    
    graph = nx.read_gpickle(f'./graph_pickle/{city}.gpickle')
    embedding = KeyedVectors.load(f'./node2vec_models/{city}.wordvectors', mmap='r')
    
    node_list = list(graph.nodes())
    
    how_many = 2 * number_of_paths
    random_generated = random.sample(node_list, k=how_many)
    start_nodes = random_generated[:number_of_paths]
    end_nodes = random_generated[number_of_paths:]
    
    len_astar, teller_astar_list = list(), list()
    len_NN, teller_NN = list(), list()
    weight_path = list()
    
    for i in range(len(start_nodes)):
        average_vel = get_weigted_average_velocity(graph)
        a = a_star.A_star_priority_queue(start_nodes[i], end_nodes[i], graph, average_vel, return_counter = True)
    
        len_astar.append(a[0])
        teller_astar_list.append(a[1])
    
        b = NNforSearch.A_star_priority_queue_NN(start_nodes[i], end_nodes[i], graph, scaler, embedding, model, return_counter=True)
        len_NN.append(b[0])
        teller_NN.append(b[1])
    
        weight_path.append(nx.shortest_path_length(graph, start_nodes[i], end_nodes[i], weight='travel_time', method='dijkstra'))    
    
        print(i)
    
    xvals_astar, average_stretch_astar, standard_dev_on_mean_stretch_astar = prepare_plot_errorbar(len_astar, weight_path,  binsize=150)                                    
    plt.ylabel('Gemiddelde rek')
    plt.xlabel('Snelste reistijd (s)')
    plt.errorbar(xvals_astar, average_stretch_astar, yerr=standard_dev_on_mean_stretch_astar, linewidth=3, elinewidth=3, linestyle='--', capsize=5,barsabove=True, label='A*')
    
    
    xvals_NN, average_stretch_NN, standard_dev_on_mean_stretchNN = prepare_plot_errorbar(len_NN, weight_path,  binsize=150)                                    
    
    plt.errorbar(xvals_NN, average_stretch_NN, yerr=standard_dev_on_mean_stretchNN, linewidth=3,elinewidth=3, linestyle='-', capsize=5,barsabove=True, label='Neuraal netwerk')
    plt.legend()
    plt.savefig(f'./NN_files/rek_NN_{city}_inverse_weights.png', bbox_inches="tight")
    plt.clf()
    
    
    bins=np.histogram(np.hstack((teller_astar_list,teller_NN)), bins=5)[1] #get the bin edges
    
    plt.hist(teller_astar_list, bins=bins)
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    
    plt.savefig(f'./NN_files/aantal_expansies_aster_{city}_inverse_weights.png', bbox_inches="tight")
    plt.clf()
    
    plt.hist(teller_NN, bins=bins)
    plt.xlabel('Aantal node expansies')
    plt.ylabel('Aantal in bin')
    plt.savefig(f'./NN_files/aantal_expansies_NN_{city}_inverse_weights.png', bbox_inches="tight")
    plt.clf()
    
    
make_graphs('Manhattan', 50)