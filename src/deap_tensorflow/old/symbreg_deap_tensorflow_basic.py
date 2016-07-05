#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random
import numpy 
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import tensorflow as tf
import gpDeapTensorflow as gpdt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp



# Creation des donnees artificielles pour test avec TensorFlow
data_train = []
for i in range(-100,100):
    x = i/10.0
    y = x**4 + x**3 + x**2 + 3.14 * x
    data_train.append([x, y])

data_test = []
for i in range(-10,10):
    x = i/10.0
    y = x**4 + x**3 + x**2 + 3.14 * x
    data_test.append([x, y])

data_train = numpy.array(data_train)
data_test = numpy.array(data_test)


###################################################
# Definition de fonctions utiles (MSE, affichage) #
###################################################


# data : measured
# func : prediction
def meanSquarredError(func, data, optimized_weights):
    sqerrors = 0
    
    inputs = data[:, :1]
    outputs = data[:, -1:]

    for x, y in zip(inputs, outputs):
        sqerrors += (y - func(x, optimized_weights))**2

    res = sqerrors / data.size

    return res


# Fonction d'evaluation
def evalSymbReg(individual, nepochs, points):
    # On transforme l'expression symbolique en une fonction executable auquelle
    # on a ajoute l'impact des coefficients
    func, nb_weights = gpdt.compile_with_weights(individual, pset)

    if nepochs == 0:
        individual.optimized_weights = [1.] * nb_weights
    else:
        # Optimisation des constantes avec TensorFlow
        individual_tensor = gpdt.primitivetree_to_tensor(individual, nb_args, nb_weights)
        optimized_weights = gpdt.tensorflow_run(individual_tensor, data_train, data_test, nepochs)

        # On garde seulement l'optimisation si la fitness est reduite
        if meanSquarredError(func, points, optimized_weights)[0] < individual.fitness.values[0][0]:
            individual.optimized_weights = optimized_weights

    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x

    return meanSquarredError(func, points, individual.optimized_weights),


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


#########################################
# Programmation genetique               #
#########################################

# Definition des nouvelles primitives

def max_rectifier(x):
    return max(x, 0)

def min_rectifier(x):
    return min(x, 0)

# On regroupe les primitives dans un ensemble
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(max_rectifier, 1)
pset.addPrimitive(min_rectifier, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

nb_args = len(pset.arguments)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Creation de la toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg, points=data_train)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Controle du bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Using HARM-GP
    pop, log = gpdt.harm_weights(pop, toolbox, 0.5, 0.1, 40, 100, alpha=0.05, beta=10, gamma=0.25, 
                                 rho=0.9, stats=mstats, halloffame=hof, verbose=True)
    
    best_individual = hof[0]

    # Affiche l'exp symbolique sous forme prefixe du modele le plus proche
    print "\n", best_individual

    # Affichage de l'exp symbolique avec coefficients
    print gpdt.exp_to_string_with_weights(best_individual)

    # Todo : optimiser les coefficients du best_individual

    func, nb_weights = gpdt.compile_with_weights(best_individual, pset)

    print "Coefficients optimaux  : ", best_individual.optimized_weights
    print "MSE : ", meanSquarredError(func, data_train, best_individual.optimized_weights)
    
    # Affiche l'arbre DEAP representant le modele le plus proche
    display_graph(best_individual)

    # Affiche les statistiques au cours de l'evolution
    display_stats(log)

    return pop, log, hof

if __name__ == "__main__":
    main()
