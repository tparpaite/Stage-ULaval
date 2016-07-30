import sys
import os
import pickle
import time
import operator
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import gp_deap_tensorflow as gpdt
import tensorflow_computation as tfc

sys.path.append('../../')
from datasets import load_utils_tmp as load
from stats import stats_to_graph as stats
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


####################
# Hyperparametres  #
####################

NEVALS_TOTAL = 10000
N_EPOCHS = 150

# Chemin relatif du repertoire logbook
LOGBOOK_PATH = "../../stats/logbook/"

classical_hyperparameters = {
    'pop_size': 500,
    'n_tournament': 5,
    'init_depth': 5,
}


######################################
# Optimisation des hyperparametres   #
######################################

def deaptensorflow_hyperparameters(pset, dataset, dataX, dataY, kfold):
    file_hypers = "./hyperparameters/hypers_gpcoef_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(file_hypers):
        with open(file_hypers, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################

    # On cree le fichier de log
    file_log = LOGBOOK_PATH + "logbook_hyperparameters/logbook_hypers_gpcoef_" + dataset + ".txt"
    fd = open(file_log, 'w')

    # On recupere le premier pli
    n_inputs = len(dataX[0])
    kfold = list(kfold)
    tr_index, te_index = kfold[0]
    trX, teX = dataX[tr_index], dataX[te_index]
    trY, teY = dataY[tr_index], dataY[te_index]
    
    # Initialisation avec une valeur de mse maximale
    best_params = { 'mse': sys.float_info.max }

    # Grid search 
    # Taille de la population
    pop_size_grid = [200, 500, 1000, 2000, 5000]

    # Nombre de participants a chaque ronde du tournoi
    n_tournament_grid = [3, 4, 5, 6, 7]

    # Profondeur a l'initilisation
    init_depth_grid = [4, 5, 6, 7]

    # Creation des 100 combinaisons possibles
    zipper = []
    for pop_size in pop_size_grid:
        for n_tournament in n_tournament_grid:
            for init_depth in init_depth_grid:
                zipper.append([pop_size, n_tournament, init_depth]) 

    # On lance 100 fois l'entrainement sur le pli avec toutes les combinaisons d'hyperparametres
    for pop_size, n_tournament, init_depth in zipper:    
        # On stocke les hyperparametres dans un dictionnaire
        hyperparameters = {
            'pop_size': pop_size,
            'n_tournament': n_tournament,
            'init_depth': init_depth,
        }
        
        # Initialisation de la GP
        toolbox = create_toolbox(hyperparameters, pset)

        # Logbook : statistiques
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        # Evolution de la population et retour du meilleur individu
        best_individual, log = deaptensorflow_launch_evolution(hyperparameters, toolbox, pset,
                                                               mstats, trX, trY, teX, teY)
        
        # On recupere les informations du meilleur individu
        hyperparameters['mse'] = best_individual.fitness.values[0][0]
        
        # On ecrit les hyperparametres et le mse associe dans le logbook dedie
        fd.write(str(hyperparameters) + "\n")

        # Flush temporaire pour debug
        fd.flush()
        os.fsync(fd.fileno())

        # Sauvegarde des hyperparametres en tant que best s'ils sont meilleurs
        if hyperparameters['mse'] < best_params['mse']:
            best_params = hyperparameters.copy()

    # On sauvegarde les hyperparametres optimaux en dur avec pickle
    with open(file_hypers, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params


########################################################################
# Definition des fonctions necessaires a la programmation genetique    #
########################################################################

# On calcule le MSE par rapport a l'ensemble test
def mean_squarred_error(func, optimized_weights, teX, teY):
    sqerrors = 0
    n_elements = len(teX)

    for x, y in zip(teX, teY):
        sqerrors += (y - func(optimized_weights, *x)) ** 2

    return sqerrors / n_elements


# Fonction d'evaluation
def eval_symbreg(individual, pset, trX, trY, teX, teY):
    n_inputs = len(trX[0])

    # On transforme l'expression symbolique en une fonction executable auquelle
    # on a ajoute l'impact des coefficients
    func, n_weights = gpdt.compile_with_weights(individual, pset)

    # On genere des coefficients aleatoires
    # TODO : check random
    individual.optimized_weights = [random.uniform(-1, 1) for _ in range (n_weights)]

    # Creation du graphe TensorFlow correspondant a l'individu
    individual_tensor = tfc.tensorflow_init(individual, n_inputs, n_weights, individual.optimized_weights)

    # Optimisation des coefficients avec TensorFlow sur l'ensemble train
    tmp_optimized_weights = tfc.tensorflow_run(individual_tensor, trX, trY, teX, teY, N_EPOCHS)

    # On verifie que TensorFlow n'a pas diverge (dans ce cas il retourne NaN pour le mse)
    if not(tmp_optimized_weights is None):
        individual.optimized_weights = tmp_optimized_weights
    
    # Evaluation du MSE sur l'ensemble test
    return mean_squarred_error(func, individual.optimized_weights, teX, teY),


# Definition des nouvelles primitives 
def max_rectifier(x):
    return max(x, 0)


def min_rectifier(x):
    return min(x, 0)


# On regroupe les primitives dans un ensemble
def create_primitive_set(n_inputs):
    pset = gp.PrimitiveSet("MAIN", n_inputs)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(max_rectifier, 1)
    pset.addPrimitive(min_rectifier, 1)
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

    return pset


def create_toolbox(hyperparameters, pset):
    # Recuperation des informations sur les hyperparametres
    n_tournament = hyperparameters['n_tournament']
    init_depth = hyperparameters['init_depth']

    # Caracteristiques de l'individu et de la fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Creation de la toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=init_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=n_tournament)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # A mettre a jour plus tard en fonction du pli
    toolbox.register("evaluate", eval_symbreg, pset=pset, trX=None, trY=None, teX=None, teY=None)

    # Controle du bloat
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


# Permet de mettre a jour la fonction d'evaluation pour chaque nouveau pli (train/test)
def update_toolbox_evaluate(toolbox, pset, trX, trY, teX, teY):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_symbreg, pset=pset, trX=trX, trY=trY, teX=teX, teY=teY)
    

########################################################################
# Fonctions principales pour la GP avec optimisation des coefficients  #
########################################################################

# deaptensorflow_launch_evolution
# Cette fonction permet de lancer la phase d'evolution sur un pli

def deaptensorflow_launch_evolution(hyperparameters, toolbox, pset, mstats, trX, trY, teX, teY):
    # Recuperation des informations sur les hyperparametres
    pop_size = hyperparameters['pop_size']

    # On met a jour la toolbox avec le pli courant
    update_toolbox_evaluate(toolbox, pset, trX, trY, teX, teY)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Processus d'evolution proprement dit avec HARM-GP (extended)
    pop, log = gpdt.harm(pop, toolbox, 0.5, 0.1, NEVALS_TOTAL, alpha=0.05, beta=10,
                         gamma=0.25, rho=0.9, stats=mstats, halloffame=hof, verbose=True)
   
    # On retourne le meilleur individu a la fin du processus d'evolution ains que les logs
    best_individual = hof[0] 
    return best_individual, log


# deaptensorflow_run
# Cette fonction sert a faire tourner la programmation genetique + coefficients sur du 5-fold x4
# On cree tout d'abord les outils permettant de faire de la GP (initialisation)
# Puis on lance le processus d'evolution de la GP sur chaque pli du 5-fold x4

def deaptensorflow_run(n_fold, hyperparameters, pset, dataX, dataY, kfold):
    random.seed(318)

    # Initialisation de la GP
    logbook_list = []
    toolbox = create_toolbox(hyperparameters, pset)

    # Logbook : statistiques
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # On boucle sur le 5-fold x4 (cross validation) 
    # NB : POUR LE MOMENT UN SEUL pli de 5-fold
    stats_dic = { 'mse_train': [], 'mse_test': [], 'size': [] }

    kfold = list(kfold)

    tr_index, te_index = kfold[n_fold]
    trX, teX = dataX[tr_index], dataX[te_index]
    trY, teY = dataY[tr_index], dataY[te_index]
        
    # Evolution de la population et retour du meilleur individu
    best_individual, log = deaptensorflow_launch_evolution(hyperparameters, toolbox, pset,
                                                           mstats, trX, trY, teX, teY)
    
    # Evaluation de l'individu en train
    func = toolbox.compile(expr=best_individual)
    optimized_weights = best_individual.optimized_weights
    mse_train = mean_squarred_error(func, optimized_weights, trX, trY)
    
    # On recupere les informations dans le dictionnaire de stats
    mse_test = best_individual.fitness.values[0][0]
    size = best_individual.height
    stats_dic['mse_train'].append(mse_train)
    stats_dic['mse_test'].append(mse_test)
    stats_dic['size'].append(size)
    
    #####################################################################
    # /!\ TODO : optimiser encore plus loin les coef du best_individual #
    #####################################################################
    
    # print "Coefficients optimaux  : ", best_individual.optimized_weights
    # print "MSE : ", mse_test
    # Affichage de l'exp symbolique avec coefficients
    # print gpdt.exp_to_string_with_weights(best_individual)
                                                        
    # On retourne le dictionnaire contenant les informations sur les stats ainsi que le lobgook
    return stats_dic, log


###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 3 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_deap_tensorflow.py data_name n_fold\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    n_fold = int(sys.argv[2])
    run = "load.load_" + dataset + "()"
    dataX, dataY, kfold = eval(run)

    # On creer le pset ici, sinon on a une erreur pour la creation des constantes ephemeres
    n_args = len(dataX[0])
    pset = create_primitive_set(n_args)

    # Recherche des hyperparametres optimaux (ou chargement si deja calcule)
    # hyperparameters = deaptensorflow_hyperparameters(pset, dataset, dataX, dataY, kfold)
    hyperparameters = classical_hyperparameters

    # Aprentissage automatique
    begin = time.time()
    stats_dic, logbook = deaptensorflow_run(n_fold, hyperparameters, pset, dataX, dataY, kfold)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # Sauvegarde du dictionnaire contenant les stats
    logbook_filename = LOGBOOK_PATH + "logbook_stats/logbook_stats_gpcoef_" + dataset + "_fold" + str(n_fold) + ".pickle"
    pickle.dump(stats_dic, open(logbook_filename, 'w'))

    # Sauvegarde du logbook
    logbook_filename = LOGBOOK_PATH + "logbook_gp/logbook_gpcoef_" + dataset + "_fold" + str(n_fold) + ".pickle"
    pickle.dump(logbook, open(logbook_filename, 'w'))

    # Sauvegarde du mse
    mse_train_mean = np.mean(stats_dic['mse_train'])
    mse_test_mean = np.mean(stats_dic['mse_test'])
    size_mean = np.mean(stats_dic['size'])
    log_mse = dataset + " (fold" + str(n_fold) + ") | MSE (train) : " + str(mse_train_mean) + " | MSE (test) : " + str(mse_test_mean) 
    log_mse += " | size : " + str(size_mean) + " | " + runtime + "\n"
    logbook_filename = LOGBOOK_PATH + "logbook_mse/logbook_mse_gpcoef.txt"
    fd = open(logbook_filename, 'a')
    fd.write(log_mse)


if __name__ == "__main__":
    main()
