import time
import random
import math
import numpy as np
import tensorflow as tf

from deap import gp
from deap import tools

#################################################
# GP Tree compilation functions with weights    #
#################################################

# compile_with_weights
#
# Compile l'expression symbolique expr en ajoutant l'impact 
# des poids sur les differents parametres 
# La fonction retournee prend en dernier argument un tableau
# contenant la valeur des poids (coefficients)
#
# Parametres :
# - (PrimitiveTree) exp, l'expression a compiler
# - (PrimitiveSet) pset, l'ensemble des primitives necessaires a la compilation
#
# Retour :
# - (bin) eval(...), une fonction prenant en dernier parametre la liste des poids
#                    si PrimitiveSet a au moins un argument,
#                    sinon on retourne l'evaluation de l'arbre
# - (int) w_size, le nombre de poids a appliquer

def compile_with_weights(expr, pset):
    code, w_size = exp_to_string_with_weights(expr)

    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = "w,"
        args += ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)

    try:
        return eval(code, pset.context, {}), w_size
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError, ("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort."), traceback


# exp_to_string_with_weights 
#
# Un individu DEAP represente une expression symbolique
# Pour le modeliser on utilise un arbre de type PrimitiveTree
# Cette fonction sert a convertir l'arbre en question vers un
# une chaine de caracteres evaluable en Python comme code executable
# De plus, on prend soin de multiplier chaque argument (terminal)
# par un coefficient (son poids)
#
#
# Parametres :
# - (PrimitiveTree) individual, l'individu a convertir
# - (int) index, indique l'avancement de la convertion (recursion)
# - (int) w_index, indique l'indice du poids courant dans le tableau des coefficients
#
# Retour :
# - (str) retVal, (int) w_index la chaine finale (lorsque index == 0) et le nombre de coef
# - (str) retVal, (int) last_id, (int) w_index 
#   La chaine intermediaire, l'id du dernier noeud traite et l'id du dernier poids traite

def exp_to_string_with_weights(individual, index = 0, w_index = 0):
    x = individual[index]

    if x.arity == 2:
        left_tree, next_id, w_index = exp_to_string_with_weights(individual, index + 1, w_index)
        right_tree, last_id, w_index = exp_to_string_with_weights(individual, next_id + 1, w_index)
        retVal = create_string(x, left_tree, right_tree, w_index)

    elif x.arity == 1:
        under_tree, last_id, w_index = exp_to_string_with_weights(individual, index + 1, w_index)
        retVal = create_string(x, under_tree, None, w_index)

    # x.arity == 0 (leaf)
    else: 
        retVal = create_string(x, None, None, w_index)
        w_index += 1
        last_id = index

    # root
    if index == 0:
        return retVal, w_index
    
    return retVal, last_id, w_index


# create_string
#
# Cette fonction sert a creer une chaine de caractere (str)
# Une chaine peut correspondre a une operation
# Une chaine peut aussi correspondre a un terminal, 
# et dans ce cas on le multiplie par un coefficient
#
#
# Parametres :
# - (Primitive, Terminal, rand101) x
#      | (Primitive) le nom de l'operation
#      | (Terminal) le nom de l'argument
#      | (rand101) la valeur de constante ephemere (coefficient, variable)
# - (str) left_tree, la chaine correspondant au membre gauche de l'operation
# - (str) right_tree, la chaine correspondant au membre droit de l'operation
# - (int) w_index, l'indice du poids courant dans le tableau des coefficients
#
# Retour :
# - (str) la chaine correspondant a l'operation ou au terminal

def create_string(x, left_tree, right_tree, w_index):
    # x est un noeud interne (operation)
    if isinstance(x, gp.Primitive):
        op = x.name
        
        if op == "add":
            return "add({}, {})".format(left_tree, right_tree)
        
        elif op == "sub":
            return "sub({}, {})".format(left_tree, right_tree)

        elif op == "mul":
            return "mul({}, {})".format(left_tree, right_tree)

        elif op == "max_rectifier":
            return "max_rectifier({})".format(left_tree)

        elif op == "min_rectifier":
            return "min_rectifier({})".format(left_tree)

    # x est une feuille qui correspond un arg ou const ephemere (var)
    else:
        return "w[{}]*{}".format(w_index, x.value)


###################################################
# Convertion de DEAP vers TensorFlow              #
###################################################

# Retourne l'index de l'argument
# Cela correspond a la colonne dans la matrice d'inputs X

def get_arg_index(arg_name):
    return int(arg_name[3])


# primitivetree_to_tensor
#
# Un individu DEAP represente une expression symbolique
# Pour le modeliser on utilise un arbre de type PrimitiveTree
# Cette fonction sert a convertir l'arbre en question vers un
# graphe qui correspond au modele (fonction de prediction)
#
# Parametres :
# - (IndividualTensor) individual_tensor, l'objet qui contient les infos
#                      de l'individu a convertir
#
# Retourne :
# - (Tensor) model, le modele (fonction pred) associe a l'expression

def primitivetree_to_tensor(index_individual, pop_graph):
    return primitivetree_to_tensor_bis(index_individual, pop_graph, 0, 0)


# primitivetree_to_tensor_bis
#
# Convertion proprement dite d'un arbre DEAP en graphe
# (tenseur) interpretable par TensorFlow
#
# Parametres :
# - (PrimitiveTree) individual, l'individu a convertir
# - (int) index, indique l'avancement de la convertion (recursion)
#
# Retour :
# - (Tensor) retVal, le graphe final (lorsque index == 0)
# - (Tensor) retVal, (int) last_id, le graphe intermediaire et l'id du dernier noeud traite

def primitivetree_to_tensor_bis(index_individual, pop_graph, index, w_index):
    individual = pop_graph.pop_info[index_individual]['individual']
    x = individual[index]

    if x.arity == 2:
        left_tree, next_id, w_index = primitivetree_to_tensor_bis(index_individual, pop_graph, index + 1, w_index)
        right_tree, last_id, w_index = primitivetree_to_tensor_bis(index_individual, pop_graph, next_id + 1, w_index)
        retVal = create_tensor_node(index_individual, pop_graph, x, left_tree, right_tree, w_index)

    elif x.arity == 1:
        under_tree, last_id, w_index = primitivetree_to_tensor_bis(index_individual, pop_graph, index + 1, w_index)
        retVal = create_tensor_node(index_individual, pop_graph, x, under_tree, None, w_index)

    # x.arity == 0 (leaf)
    else: 
        retVal = create_tensor_node(index_individual, pop_graph, x, None, None, w_index)
        w_index += 1
        last_id = index

    # Sommet de la recursion
    if index == 0:
        return retVal

    return retVal, last_id, w_index


# create_tensor_node
#
# Cette fonction sert a creer un noeud du graphe Tensorflow (tenseur)
# Un noeud peut-etre une operation, auquel cas on indique les parametres
# Un noeud peut aussi etre un argument ou une constante (feuille)
#
#
# Parametres :
# - (Primitive, Terminal, rand101) x
#      | (Primitive) le nom de l'operation
#      | (Terminal) le nom de l'argument
#      | (rand101) la valeur de constante ephemere (coefficient, variable)
# - (Tensor) left_tree, le membre gauche de l'operation
# - (Tensor) right_tree, le membre droit de l'operation
#
# Retour :
# - (Tensor) le noeud correspondant a l'operation ou au terminal

def create_tensor_node(index_individual, pop_graph, x, left_tree, right_tree, w_index):
    # x est un noeud interne (operation)
    if isinstance(x, gp.Primitive):
        op = x.name
        
        if op == "add":
            return tf.add(left_tree, right_tree)
        
        elif op == "sub":
            return tf.sub(left_tree, right_tree)

        elif op == "mul":
            return tf.mul(left_tree, right_tree)

        elif op == "max_rectifier":
            return tf.maximum(left_tree, 0)

        elif op == "min_rectifier":
            return tf.minimum(left_tree, 0)

    # x est une feuille
    else:
        value = x.value

        # La feuille correspond a une constante ephemere (var)
        if isinstance(x, gp.rand101):
            W = pop_graph.weights_tab[index_individual]
            const_eph = tf.constant(value, dtype="float")

            return tf.mul(const_eph, W[w_index])

        # La feuille correspond a un argument
        else:
            # On recupere les informations
            arg_index = get_arg_index(value)
            X = individual_tensor.input
            W = pop_graph.weights_tab[index_individual]
            column = X[:, arg_index]

            # On cree le noeud en choissisant le bon argument (colonne)
            return tf.mul(column, W[w_index])


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


################################################################################
# GP bloat control algorithms with weights optimization   TMP DISABLED         #
################################################################################

# HARM_WEIGHTS
# Algorithme de programmation genetique permettant 
# Changements par rapport a l'algorithme HARM GP classique
#
# On ajoute l'optimisation des coefficients pour tous les individus
# a chaque nouvelle generation 
# (sur un nombre restreint d'epoques pour ne pas surcharger l'algorithme
# en calcul pour des individus voues a l'echec)
#
# A chaque generation, pour les 10 % meilleurs individus (a ajuster)
# on procede a une optimisation des coefficients plus en profondeur
# pour augmenter la fitness des individus prometteurs (qui ne seraient 
# normalement pas sortis du lot avec des coefficients par defaut a 1)

def harm_weights(population, toolbox, cxpb, mutpb, ngen, n_epochs,
                 alpha, beta, gamma, rho, nbrindsmodel=-1, mincutoff=20,
                 stats=None, halloffame=None, verbose=__debug__):
    """Implement bloat control on a GP evolution using HARM-GP, as defined in
    [Gardner2015]. It is implemented in the form of an evolution algorithm
    (similar to :func:`~deap.algorithms.eaSimple`).

    [Thibault2016] I took harm gp algorithm and I added the optimization
    of coefficients for each generation

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param n_epochs: The number of epochs during optimization.
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

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, [n_epochs] * len(invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

        

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen + 1):
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
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, [n_epochs] * len(invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

    return population, logbook
