import sys
import os
import pickle
import time
import operator
import math
import random
import numpy 
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import tensorflow as tf
import gp_deap_tensorflow as gpdt
import tensorflow_computation as tfc

sys.path.append('../../')
from datasets import load_utils_tmp as load
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


####################
# Hyperparametres  #
####################

# Note : on deduit le nombre de generations de la taille de la population
# (sachant qu'on veut 100 000 evaluations)

N_EPOCHS = 150

classical_hyperparameters = {
    'pop_size': 300,
    'n_tournament': 3,
    'init_depth': 2,
}


######################################
# Optimisation des hyperparametres   #
######################################

def deaptensorflow_hyperparameters(dataset, dataX, dataY, kfold):
    filepath = "./hyperparameters/hypers_deaptensorflow_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################

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
        pset = create_primitive_set(n_inputs)
        toolbox = create_toolbox(hyperparameters, pset)
        
        print hyperparameters
        # Evolution de la population et retour du meilleur individu
        best_individual, log = deaptensorflow_launch_evolution(hyperparameters, toolbox, pset,
                                                               trX, trY, teX, teY)
        
        # On recupere les informations du meilleur individu
        hyperparameters['mse'] = best_individual.fitness.values[0]
        
        # Sauvegarde des hyperparametres s'ils sont meilleurs
        if hyperparameters['mse'] < best_params['mse']:
            best_params = hyperparameters.copy()

    # On sauvegarde les hyperparametres optimaux en dur avec pickle
    with open(filepath, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params


###################################################
# Definition des fonctions d'affichage DEAP       #
###################################################

def display_graph(expr):
    nodes, edges, labels = gp.graph(expr)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # On recupere la position des noeuds
    pos =  nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()


def display_stats(logbook):
    # Recuperation des donnes a partir du logbook
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    size_avgs = logbook.chapters["size"].select("avg")

    # On cree la courbe qui represente l'evolution de la fitness en fonction des generations
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    # On cree la courbe qui represente l'evolution de la taille des individus
    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    # Affichage graphique
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper center")

    plt.show()


def merge_logbook(logbook_list):
    # On suppose que les logbooks ont le meme nombre d'entrees
    n_entry = len(logbook_list[0])
    
    # Le logbook qui contient les entrees fusionnees
    res = tools.Logbook()

    # Les champs a fusionner
    chapters = ["fitness", "size"]
    fields = ["avg", "min", "max", "std"]

    # On realise la fusion proprement dite
    for i in range(n_entry):
        record = {}
        # Moyenne du nombre d'evaluations
        nevals = numpy.mean([logbook[i]["nevals"] for logbook in logbook_list])

        # Moyenne de chaque champ pour chaque chapitre
        for chapter in chapters:
            record[chapter] = {}
            for field in fields:
                record[chapter][field] = numpy.mean([logbook.chapters[chapter][i][field] for logbook in logbook_list])

        res.record(gen=i, nevals=nevals, **record)

    return res


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
def eval_symbreg(individual, n_epochs, pset, trX, trY, teX, teY):
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
    individual.optimized_weights = tfc.tensorflow_run(individual_tensor, trX, trY, teX, teY, n_epochs)

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

def deaptensorflow_launch_evolution(hyperparameters, toolbox, pset, trX, trY, teX, teY):
    # Recuperation des informations sur les hyperparametres
    pop_size = hyperparameters['pop_size']
    # On en deduit le nombre de generations (etant donne qu'on veut 100 000 evaluations)
    n_gen = 100000 / pop_size

    # On met a jour la toolbox avec le pli courant
    update_toolbox_evaluate(toolbox, pset, trX, trY, teX, teY)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Logbook : statistiques
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    # Processus d'evolution proprement dit avec HARM-GP (extended)
    pop, log = gpdt.harm_weights(pop, toolbox, 0.5, 0.1, n_gen, N_EPOCHS, alpha=0.05, beta=10,
                                 gamma=0.25, rho=0.9, stats=mstats, halloffame=hof, verbose=True)
   
    # On retourne le meilleur individu a la fin du processus d'evolution ains que les logs
    best_individual = hof[0] 
    return best_individual, log


# deaptensorflow_run
# Cette fonction sert a faire tourner la programmation genetique + coefficients sur du 5-fold x4
# On cree tout d'abord les outils permettant de faire de la GP (initialisation)
# Puis on lance le processus d'evolution de la GP sur chaque pli du 5-fold x4

def deaptensorflow_run(hyperparameters, dataX, dataY, kfold):
    random.seed(318)

    # Initialisation de la GP
    logbook_list = []
    n_inputs = len(dataX[0])
    pset = create_primitive_set(n_inputs)
    toolbox = create_toolbox(hyperparameters, pset)

    # On boucle sur le 5-fold x4 (cross validation)
    mse_sum = 0
    size_sum = 0

    for tr_index, te_index in kfold:
        trX, teX = dataX[tr_index], dataX[te_index]
        trY, teY = dataY[tr_index], dataY[te_index]
        
        # Evolution de la population et retour du meilleur individu
        best_individual, log = deaptensorflow_launch_evolution(hyperparameters, toolbox, pset,
                                                               trX, trY, teX, teY)
    
        # On recupere les informations pour faire la moyenne
        mse_sum += best_individual.fitness.values[0]
        size_sum += best_individual.height
        logbook_list.append(log)

        #####################################################################
        # /!\ TODO : optimiser encore plus loin les coef du best_individual #
        #####################################################################
    
        print "Coefficients optimaux  : ", best_individual.optimized_weights
        print "MSE : ", mse
        # Affichage de l'exp symbolique avec coefficients
        print gpdt.exp_to_string_with_weights(best_individual)
        # Affiche l'arbre DEAP representant le modele le plus proche
        # display_graph(best_individual)
        # Affiche les statistiques sous forme de graphe
        # display_stats(log)
    
    logbook = merge_logbook(logbook_list)

    # On retourne la moyenne du MSE et size obtenue en appliquant la 5-fold cross-validation
    mse_mean = (mse_sum / 5.0)[0]
    size_mean = size_sum / 5.0

    return mse_mean, size_mean, logbook


###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_deap_tensorflow.py data_name\n"
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

    # Recherche des hyperparametres optimaux (ou chargement si deja calcule)
    hyperparameters = deaptensorflow_hyperparameters(dataset, dataX, dataY, kfold)

    # Aprentissage automatique
    begin = time.time()
    mse, size, logbook = deaptensorflow_run(hyperparameters, dataX, dataY, kfold)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le logbook et le mse en dur
    logbook_filename = "logbook_" + dataset + ".pickle"
    pickle.dump(logbook, open(logbook_filename, "w"))

    log_mse = dataset + " | MSE : " + str(mse) + " | size : " + str(size) + " | " + runtime + "\n"
    file = open("./logbook/logbook_mse_deaptensorflow.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
