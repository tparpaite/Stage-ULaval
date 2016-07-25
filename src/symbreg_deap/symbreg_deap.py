import sys
import os
import pickle
import time
import operator
import math
import random
import numpy as np

sys.path.append('../../')
from datasets import load_utils as load
from stats import stats_to_graph as stats
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


####################
# Hyperparametres  #
####################

POP_SIZE = 300
NEVALS_TOTAL = 10000

# Chemin relatif du repertoire logbook
LOGBOOK_PATH = "../../stats/logbook/"

classical_hyperparameters = {
    'pop_size': 300,
    'n_tournament': 3,
    'init_depth': 2,
}


######################################
# Optimisation des hyperparametres   #
######################################

def hyperparameters_optimization(pset, dataset, dataX, dataY, kf_array):
    file_hypers = "./hyperparameters/hypers_gpclassic_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(file_hypers):
        with open(file_hypers, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################

    # On cree le fichier de log
    file_log = LOGBOOK_PATH + "logbook_hyperparameters/logbook_hypers_gpclassic_" + dataset + ".txt"
    fd = open(file_log, 'w')

    # On recupere le premier pli
    kfold = list(kf_array[0])
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
        
        #print hyperparameters
        
        # Evolution de la population et retour du meilleur individu
        best_individual, log = deap_launch_evolution(hyperparameters, toolbox, pset,
                                                     mstats, trX, trY, teX, teY)
        
        # On recupere les informations du meilleur individu
        hyperparameters['mse'] = best_individual.fitness.values[0][0]
        
        # On ecrit les hyperparametres et le mse associe dans le logbook dedie
        fd.write(str(hyperparameters) + "\n")

        # Sauvegarde des hyperparametres en tant que best s'ils sont meilleurs
        if hyperparameters['mse'] < best_params['mse']:
            best_params = hyperparameters.copy()

    # On sauvegarde les hyperparametres optimaux en dur avec pickle
    with open(file_hypers, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params


##########################################################################################
# GP classique                                                                           #
# On modidife l'algorithme pour qu'il s'arrete lorsqu'il atteint un nombre d'evaluation  #
# donne au lieu s'arreter a une generation precise                                       #
##########################################################################################

def eaSimple(population, toolbox, cxpb, mutpb, nevals_total, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population and a :class:`~deap.tools.Logbook`
              with the statistics of the evolution.
    
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution (if
    any). The logbook will contain the generation number, the number of
    evalutions for each generation and the statistics if a
    :class:`~deap.tools.Statistics` if any. The *cxpb* and *mutpb* arguments
    are passed to the :func:`varAnd` function. The pseudocode goes as follow
    ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # FIRST GENERATION
    gen = 0
    nevals = 0

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(population)

    # Append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    nevals_gen = len(invalid_ind)
    nevals += nevals_gen
    logbook.record(gen=gen, nevals=nevals, **record)
    if verbose:
        print logbook.stream    

    # Begin the generational process
    while nevals < nevals_total:
        gen += 1

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        nevals_gen = len(invalid_ind)
        nevals += nevals_gen
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print logbook.stream        

    return population, logbook


########################################################################
# Definition des fonctions necessaires a la programmation genetique    #
########################################################################

# On calcule le MSE par rapport a l'ensemble test
def mean_squarred_error(func, teX, teY):
    sqerrors = 0
    n_elements = len(teX)

    for x, y in zip(teX, teY):
        sqerrors += (y - func(*x))**2

    return sqerrors / n_elements


# Fonction d'evaluation
def eval_symbreg(individual, toolbox, teX, teY):
    # On transforme l'expression symbolique en une fonction executable
    func = toolbox.compile(expr=individual)

    # Evaluation de la MSE sur l'ensemble test
    return mean_squarred_error(func, teX, teY),


# Definition des nouvelles primitives
def max_rectifier(x):
    return max(x, 0)


def min_rectifier(x):
    return min(x, 0)


# On regroupe les primitives dans un ensemble
def create_primitive_set(n_args):
    pset = gp.PrimitiveSet("MAIN", n_args)
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
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_symbreg, toolbox=toolbox, teX=None, teY=None)
    toolbox.register("select", tools.selTournament, tournsize=n_tournament)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Controle du bloat
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    return toolbox


# Permet de mettre a jour la fonction d'evaluation pour chaque nouveau fold (train/test)
def update_toolbox_evaluate(toolbox, teX, teY):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", eval_symbreg, toolbox=toolbox, teX=teX, teY=teY)


########################################
# Fonctions principales GP classique   #
########################################

# deap_launch_evolution
# Cette fonction permet de lancer la phase d'evolution sur un pli

def deap_launch_evolution(hyperparameters, toolbox, pset, mstats, trX, trY, teX, teY):
    # Recuperation des informations sur les hyperparametres
    pop_size = hyperparameters['pop_size']

    # On met a jour la toolbox avec le pli courant
    update_toolbox_evaluate(toolbox, teX, teY)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Classic GP (attention au bloat)
    pop, log = eaSimple(pop, toolbox, 0.5, 0.1, NEVALS_TOTAL, stats=mstats,
                        halloffame=hof, verbose=True)
   
    # On retourne le meilleur individu a la fin du processus d'evolution ains que les logs
    best_individual = hof[0] 
    return best_individual, log


# deap_run
# Cette fonction sert a faire tourner la programmation genetique classique sur du 5-fold x4
# On cree tout d'abord les outils permettant de faire de la GP (initialisation)
# Puis on lance le processus d'evolution de la GP sur chaque pli du 5-fold x4

def deap_run(hyperparameters, pset, dataX, dataY, kf_array):
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
    stats_dic = { 'mse_train_array': [], 'mse_test_array': [], 'size_array': [] }

    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]
        
            # Evolution de la population et retour du meilleur individu
            best_individual, log = deap_launch_evolution(hyperparameters, toolbox, pset,
                                                         mstats, trX, trY, teX, teY)
            
            # On recupere les informations dans le dictionnaire de stats
            mse_train = eval_symbreg(best_individual, toolbox, trX, trY)[0][0]
            mse_test = best_individual.fitness.values[0][0]
            size = best_individual.height
            stats_dic['mse_train_array'].append(mse_train)
            stats_dic['mse_test_array'].append(mse_test)
            stats_dic['size_array'].append(size)
            logbook_list.append(log)
    
    logbook = stats.merge_logbook(logbook_list)

    # On retourne le dictionnaire contenant les informations sur les stats ainsi que le lobgook
    return stats_dic, logbook


################
# Main call    #
################

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_deap.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "load.load_" + dataset + "()"
    dataX, dataY, kf_array = eval(run)

    # On creer le pset ici, sinon on a une erreur pour la creation des constantes ephemeres
    n_args = len(dataX[0])
    pset = create_primitive_set(n_args)

    # Recherche des hyperparametres optimaux (ou chargement si deja calcule)
    hyperparameters = hyperparameters_optimization(pset, dataset, dataX, dataY, kf_array)

    # Aprentissage automatique
    begin = time.time()
    stats_dic, logbook = deap_run(hyperparameters, pset, dataX, dataY, kf_array)
    runtime = "{:.2f} seconds".format(time.time() - begin)
    
    # Sauvegarde du dictionnaire contenant les stats
    logbook_filename = LOGBOOK_PATH + "logbook_stats/logbook_stats_gpclassic_" + dataset + ".pickle"
    pickle.dump(stats_dic, open(logbook_filename, 'w'))

    # Sauvegarde du logbook
    logbook_filename = LOGBOOK_PATH + "logbook_gp/logbook_gpclassic_" + dataset + ".pickle"
    pickle.dump(logbook, open(logbook_filename, 'w'))

    # Sauvegarde du mse
    mse_train_mean = np.mean(stats_dic['mse_train_array'])
    mse_test_mean = np.mean(stats_dic['mse_test_array'])
    size_mean = np.mean(stats_dic['size_array'])
    log_mse = dataset + " | MSE (train) : " + str(mse_train_mean) + " | MSE (test) : " + str(mse_test_mean) 
    log_mse += " | size : " + str(size_mean) + " | " + runtime + "\n"
    logbook_filename = LOGBOOK_PATH + "logbook_mse/logbook_mse_gpclassic.txt"
    fd = open(logbook_filename, 'a')
    fd.write(log_mse)


if __name__ == "__main__":
    main()
