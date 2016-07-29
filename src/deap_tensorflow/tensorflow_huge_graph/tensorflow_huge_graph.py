#######################################################################
# tensorflow_huge_graph.py                                            #
# Ce programme python a pour objectif de comparer deux approches      #
# pour entrainer des individus de GP avec TensorFlow                  #
# Pour cela on considere une population de 100 individus sur un pli   #
# - 100x1 : creer un graphe TF pour chaque individu                   #
# - 1x100 : creer un unique graphe TF contenant tous les individus    #
# NB : La toolbox de GP ne nous sert que pour generer la population   #
#######################################################################

import sys
sys.path.append('../')
sys.path.append('../../../')

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
import tensorflow_huge_graph_computation as tfhudge
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

POP_SIZE = 100
N_EPOCHS = 150
INIT_DEPTH = 5

# Chemin relatif du repertoire logbook
LOGBOOK_PATH = "../../../stats/logbook/"


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


def create_toolbox(pset):
    # Caracteristiques de l'individu et de la fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Creation de la toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=INIT_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


# Permet de mettre a jour la fonction d'evaluation pour chaque nouveau pli (train/test)
def update_toolbox_evaluate(toolbox, pset, trX, trY, teX, teY):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_symbreg, pset=pset, trX=trX, trY=trY, teX=teX, teY=teY)
    

#############################
# Fonctions methode 100x1   #
#############################

# training_100x1
# Permet de lancer un entrainement (optimisation des coef) sur un seul individu

def training_100x1(individual, pset, trX, trY, teX, teY):
    n_inputs = len(trX[0])

    # On transforme l'expression symbolique en une fonction executable auquelle
    # on a ajoute l'impact des coefficients
    func, n_weights = gpdt.compile_with_weights(individual, pset)

    # On genere des coefficients aleatoires
    individual.optimized_weights = [random.uniform(-1, 1) for _ in range (n_weights)]

    # Creation du graphe TensorFlow correspondant a l'individu
    individual_tensor = tfc.tensorflow_init(individual, n_inputs, n_weights, individual.optimized_weights)

    # Optimisation des coefficients avec TensorFlow sur l'ensemble train
    tmp_optimized_weights = tfc.tensorflow_run(individual_tensor, trX, trY, teX, teY, N_EPOCHS)

    # On verifie que TensorFlow n'a pas diverge (dans ce cas il retourne NaN pour le mse)
    if tmp_optimized_weights is None:
        individual.optimized_weights = tmp_optimized_weights
    
    # Evaluation du MSE sur l'ensemble test
    return mean_squarred_error(func, individual.optimized_weights, teX, teY),


# launch_training_100x1
# Permet de lancer l'entrainement de 100 individus sur 100 graphes differents

def launch_training_100x1(pop, pset, trX, trY, teX, teY):
    best_mse = sys.float_info.max

    # On lance l'entrainement sur tous les individus dans des graphes differents
    for individual in pop:
        current_mse = training_100x1(individual, pset, trX, trY, teX, teY)

        # On met a jour le best_mse
        if current_mse < best_mse:
            best_mse = current_mse

    return best_mse


#############################
# Fonctions methode 1x100   #
#############################

def generate_pop_info(pop, pset):
    n_individuals = len(pop)
    pop_info = [{}] * n_individuals

    for i in range(n_individuals):
        individual = pop[i]

        # On transforme l'expression symbolique en une fonction executable auquelle
        # on a ajoute l'impact des coefficients
        func, n_weights = gpdt.compile_with_weights(individual, pset)

        # Sauvegarde des informations sur l'individu
        pop_info[i]['individual'] = individual
        pop_info[i]['func'] = func
        pop_info[i]['n_weights'] = n_weights

    return pop_info
    

# launch_training_1x100
# Genere d'abord un graphe unique pour les 100 individus
# Puis lance l'entrainement sur ce graphe

def launch_training_1x100(pop, pset, trX, trY, teX, teY):
    n_individuals = len(pop)
    n_inputs = len(trX[0])

    # Recuperation des infos sur l'individu (individu compile, n_weights)
    pop_info = generate_pop_info(pop, pset)

    # Generation du graphe
    pop_graph = tfhudge.tensorflow_init(pop_info, n_inputs)

    # Optimisation des coefficients avec TensorFlow sur l'ensemble train
    # Les poids optimaux sont mis a jour dans pop_info (effet de bord)
    # Retourne le index_individual du meilleur individu
    best_index_individual = tfhudge.tensorflow_run(pop_info, pop_graph, trX, trY, teX, teY, N_EPOCHS)
    
    # Evaluation du MSE sur l'ensemble test pour le meilleur individu
    func = pop_info[best_index_individual]['func']
    optimized_weights = pop_info[best_index_individual]['optimized_weights']
    best_mse = mean_squarred_error(func, optimized_weights, teX, teY)
            
    return best_mse
    

#############################
# Fonction principale       #
#############################

# tensorflow_huge_graph_run
# Cette fonction sert a generer une population aleatoire de 100 individus
# Puis on entraine les individus avec TensorFlow sur un pli (le premier)
# en utilisant les deux methodes decrites ci-dessus (100x1 vs 1x100)

def tensorflow_huge_graph_run(dataX, dataY, kfold):
    random.seed(318)

    # On recupere le premier pli
    n_inputs = len(dataX[0])
    kfold = list(kfold)
    tr_index, te_index = kfold[0]
    trX, teX = dataX[tr_index], dataX[te_index]
    trY, teY = dataY[tr_index], dataY[te_index]

    # Initialisation de la GP
    n_args = len(dataX[0])
    pset = create_primitive_set(n_args)
    toolbox = create_toolbox(pset)

    # Creation de la population de 100 individus
    pop = toolbox.population(n=POP_SIZE)

    # Evolution TF 100x1
    begin = time.time()
    mse_100x1 = launch_training_100x1(pop, pset, trX, trY, teX, teY)
    runtime_100x1 = "{:.2f} seconds".format(time.time() - begin)

    # Evolution TF 1x100
    begin = time.time()
    mse_1x100 = launch_training_1x100(pop, pset, trX, trY, teX, teY)
    runtime_1x100 = "{:.2f} seconds".format(time.time() - begin)

    # Stockage des resultats
    stats_dic = { 
        'runtime_100x1': runtime_100x1,
        'mse_100x1': mse_100x1,
        'runtime_1x100': runtime_1x100,
        'mse_1x100': mse_1x100
    }

    return stats_dic


###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python tensorflow_huge_graph_run data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "load.load_" + dataset + "()"
    dataX, dataY, kfold = eval(run)

    # On lance notre methode principale
    stats_dic = tensorflow_huge_graph_run(dataX, dataY, kfold)

    # Sauvegarde du dictionnaire contenant les stats en texte brut
    logbook_filename = LOGBOOK_PATH + "logbook_tfhugegraph/logbook_tfhugegraph_" + dataset + ".txt"
    fd = open(logbook_filename, 'a')
    fd.write(str(stats_dic))


if __name__ == "__main__":
    main()
