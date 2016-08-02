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
    file_hypers = "./hyperparameters/hypers_gpharm_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(file_hypers):
        with open(file_hypers, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################

    # On cree le fichier de log
    file_log = LOGBOOK_PATH + "logbook_hyperparameters/logbook_hypers_gpharm_" + dataset + ".txt"
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
        
        print hyperparameters
        
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
# GP harm                                                                                #
# On modidife l'algorithme pour qu'il s'arrete lorsqu'il atteint un nombre d'evaluation  #
# donne au lieu s'arreter a une generation precise                                       #
##########################################################################################

def harm(population, toolbox, cxpb, mutpb, nevals_total, alpha, 
         beta, gamma, rho, nbrindsmodel=-1, mincutoff=20,
         stats=None, halloffame=None, verbose=__debug__):
    """Implement bloat control on a GP evolution using HARM-GP, as defined in
    [Gardner2015]. It is implemented in the form of an evolution algorithm
    (similar to :func:`~deap.algorithms.eaSimple`).

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param alpha: The HARM *alpha* parameter.
    :param beta: The HARM *beta* parameter.
    :param gamma: The HARM *gamma* parameter.
    :param rho: The HARM *rho* parameter.
    :param nbrindsmodel: The number of individuals to generate in order to
                            model the natural distribution. -1 is a special
                            value which uses the equation proposed in
                            [Gardner2015] to set the value of this parameter :
                            max(2000, len(population))
    :param mincutoff: The absolute minimum value for the cutoff point. It is
                        used to ensure that HARM does not shrink the population
                        too much at the beginning of the evolution. The default
                        value is usually fine.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. note::
       The recommended values for the HARM-GP parameters are *n_epochs=100*, *alpha=0.05*,
       *beta=10*, *gamma=0.25*, *rho=0.9*. However, these parameters can be
       adjusted to perform better on a specific problem (see the relevant
       paper for tuning information). The number of individuals used to
       model the natural distribution and the minimum cutoff point are less
       important, their default value being effective in most cases.

    .. [Gardner2015] M.-A. Gardner, C. Gagne, and M. Parizeau, Controlling
        Code Growth by Dynamically Shaping the Genotype Size Distribution,
        Genetic Programming and Evolvable Machines, 2015,
        DOI 10.1007/s10710-015-9242-8

    """
    def _genpop(n, pickfrom=[], acceptfunc=lambda s: True, producesizes=False):
        # Generate a population of n individuals, using individuals in
        # *pickfrom* if possible, with a *acceptfunc* acceptance function.
        # If *producesizes* is true, also return a list of the produced
        # individuals sizes.
        # This function is used 1) to generate the natural distribution
        # (in this case, pickfrom and acceptfunc should be let at their
        # default values) and 2) to generate the final population, in which
        # case pickfrom should be the natural population previously generated
        # and acceptfunc a function implementing the HARM-GP algorithm.
        producedpop = []
        producedpopsizes = []
        while len(producedpop) < n:
            if len(pickfrom) > 0:
                # If possible, use the already generated
                # individuals (more efficient)
                aspirant = pickfrom.pop()
                if acceptfunc(len(aspirant)):
                    producedpop.append(aspirant)
                    if producesizes:
                        producedpopsizes.append(len(aspirant))
            else:
                opRandom = random.random()
                if opRandom < cxpb:
                    # Crossover
                    aspirant1, aspirant2 = toolbox.mate(*map(toolbox.clone,
                                                             toolbox.select(population, 2)))
                    del aspirant1.fitness.values, aspirant2.fitness.values
                    if acceptfunc(len(aspirant1)):
                        producedpop.append(aspirant1)
                        if producesizes:
                            producedpopsizes.append(len(aspirant1))

                    if len(producedpop) < n and acceptfunc(len(aspirant2)):
                        producedpop.append(aspirant2)
                        if producesizes:
                            producedpopsizes.append(len(aspirant2))
                else:
                    aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                    if opRandom - cxpb < mutpb:
                        # Mutation
                        aspirant = toolbox.mutate(aspirant)[0]
                        del aspirant.fitness.values
                    if acceptfunc(len(aspirant)):
                        producedpop.append(aspirant)
                        if producesizes:
                            producedpopsizes.append(len(aspirant))

        if producesizes:
            return producedpop, producedpopsizes
        else:
            return producedpop

    halflifefunc = lambda x: (x * float(alpha) + beta)
    if nbrindsmodel == -1:
        nbrindsmodel = max(2000, len(population))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # FIRST GENERATION
    gen = 0
    nevals = 0

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    # Individus dont la fitness a ete evaluee (normalement tableau vide a la premiere generation)
    valid_ind = [ind for ind in population if ind.fitness.valid]

    # Affichage de l'evolution du MSE en fonction du nombre d'individus evalues tous les 100 points
    for ind, fit in zip(invalid_ind, fitnesses):
        nevals += 1

        # On met a jour l'individu et la population valide
        ind.fitness.values = fit
        valid_ind.append(ind)
        
        # Si on est sur un multiple de 100 on ajoute au logbook
        if nevals%100 == 0:
            # Append the current generation statistics to the logbook (every 100 points)
            record = stats.compile(valid_ind) if stats else {}
            logbook.record(gen=gen, nevals=nevals, **record)
            if verbose:
                print logbook.stream 

    # Premiere generation : toute la population est evaluee)
    if halloffame is not None:
        halloffame.update(population)

    # Begin the generational process
    while nevals < nevals_total:
        gen += 1

        # Estimation population natural distribution of sizes
        naturalpop, naturalpopsizes = _genpop(nbrindsmodel, producesizes=True)
        
        naturalhist = [0] * (max(naturalpopsizes) + 3)
        for indsize in naturalpopsizes:
            # Kernel density estimation application
            naturalhist[indsize] += 0.4
            naturalhist[indsize - 1] += 0.2
            naturalhist[indsize + 1] += 0.2
            naturalhist[indsize + 2] += 0.1
            if indsize - 2 >= 0:
                naturalhist[indsize - 2] += 0.1

        # Normalization
        naturalhist = [val * len(population) / nbrindsmodel for val in naturalhist]

        # Cutoff point selection
        sortednatural = sorted(naturalpop, key=lambda ind: ind.fitness)
        cutoffcandidates = sortednatural[int(len(population) * rho - 1):]
        # Select the cutoff point, with an absolute minimum applied
        # to avoid weird cases in the first generations
        cutoffsize = max(mincutoff, len(min(cutoffcandidates, key=len)))

        # Compute the target distribution
        targetfunc = lambda x: (gamma * len(population) * math.log(2) /
                                halflifefunc(x)) * math.exp(-math.log(2) *
                                                            (x - cutoffsize) / halflifefunc(x))
        targethist = [naturalhist[binidx] if binidx <= cutoffsize else
                      targetfunc(binidx) for binidx in range(len(naturalhist))]

        # Compute the probabilities distribution
        probhist = [t / n if n > 0 else t for n, t in zip(naturalhist, targethist)]
        probfunc = lambda s: probhist[s] if s < len(probhist) else targetfunc(s)
        acceptfunc = lambda s: random.random() <= probfunc(s)

        # Generate offspring using the acceptance probabilities
        # previously computed
        offspring = _genpop(len(population), pickfrom=naturalpop,
                            acceptfunc=acceptfunc, producesizes=False)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        # Individus dont la fitness a ete evaluee a la generation precedente
        valid_ind = [ind for ind in offspring if ind.fitness.valid]

        # Affichage de l'evolutionen en fonction du nombre d'individus evalues tous les 100 points
        for ind, fit in zip(invalid_ind, fitnesses):
            nevals += 1
            
            # On arrete la GP si on a depasse 100 000 evaluations
            if nevals > NEVALS_TOTAL:
                break

            # On met a jour l'individu et la population valide
            ind.fitness.values = fit
            valid_ind.append(ind)

            # Si on est sur un multiple de 100, on ajoute au logbook
            if nevals%100 == 0:
                # Append the current generation statistics to the logbook (every 100 points)
                record = stats.compile(valid_ind) if stats else {}
                logbook.record(gen=gen, nevals=nevals, **record)
                if verbose:
                    print logbook.stream 

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(valid_ind)
            
        # Replace the current population by the offspring (evaluated)
        population[:] = valid_ind

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


#########################################
# Fonctions principales GP harm         #
#########################################

# deap_launch_evolution
# Cette fonction permet de lancer la phase d'evolution sur un pli

def deap_launch_evolution(hyperparameters, toolbox, pset, mstats, trX, trY, teX, teY):
    # Recuperation des informations sur les hyperparametres
    pop_size = hyperparameters['pop_size']

    # On met a jour la toolbox avec le pli courant
    update_toolbox_evaluate(toolbox, teX, teY)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Using HARM-GP extended
    pop, log = harm(pop, toolbox, 0.5, 0.1, NEVALS_TOTAL, alpha=0.05, beta=10, gamma=0.25, 
                    rho=0.9, stats=mstats, halloffame=hof, verbose=True)
   
    # On retourne le meilleur individu a la fin du processus d'evolution ains que les logs
    best_individual = hof[0] 
    return best_individual, log


# deap_run
# Cette fonction sert a faire tourner la programmation genetique avec harm sur du 5-fold x4
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
    stats_dic = { 'mse_train': [], 'mse_test': [], 'size': [] }    

    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]

            # Evolution de la population et retour du meilleur individu
            best_individual, log = deap_launch_evolution(hyperparameters, toolbox, pset,
                                                         mstats, trX, trY, teX, teY)
            
            # Evaluation de l'individu en train
            func = toolbox.compile(expr=best_individual)
            mse_train = mean_squarred_error(func, trX, trY)
            
            # On recupere les informations dans le dictionnaire de stats
            mse_test = best_individual.fitness.values[0][0]
            size = len(best_individual)
            stats_dic['mse_train'].append(mse_train)
            stats_dic['mse_test'].append(mse_test)
            stats_dic['size'].append(size)
            logbook_list.append(log)
    
    logbook = stats.merge_logbook(logbook_list)

    # On retourne le dictionnaire contenant les informations sur les stats ainsi que le lobgook
    return stats_dic, logbook


################
# Main call    #
################

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_deap_harm.py data_name\n"
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
    logbook_filename = LOGBOOK_PATH + "logbook_stats/logbook_stats_gpharm_100k_" + dataset + ".pickle"
    pickle.dump(stats_dic, open(logbook_filename, 'w'))

    # Sauvegarde du logbook
    logbook_filename = LOGBOOK_PATH + "logbook_gp/logbook_gpharm_100k_" + dataset + ".pickle"
    pickle.dump(logbook, open(logbook_filename, 'w'))

    # Sauvegarde du mse
    mse_train_mean = np.mean(stats_dic['mse_train'])
    mse_test_mean = np.mean(stats_dic['mse_test'])
    size_mean = np.mean(stats_dic['size'])
    log_mse = dataset + " | MSE (train) : " + str(mse_train_mean) + " | MSE (test) : " + str(mse_test_mean) 
    log_mse += " | size : " + str(size_mean) + " | " + runtime + "\n"
    logbook_filename = LOGBOOK_PATH + "logbook_mse/logbook_mse_gpharm.txt"
    fd = open(logbook_filename, 'a')
    fd.write(log_mse)


if __name__ == "__main__":
    main()
