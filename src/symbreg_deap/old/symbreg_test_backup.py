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
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#########################
# Fonctions utiles (MSE, affichage)
#########################

# points -> prediction
# func -> real
def meanSquarredError(func, points):
    sqerrors = 0

    for p in points:
        inputs = [float(p[x]) for x in range(len(p)-1)]
        outputs = float(p[len(p)-1])
        sqerrors += (outputs - func(*inputs))**2

    # On arrondit sinon suraprentissage
    res = round(sqerrors / len(points), 10)
    return res,


def display_graph(expr):
    nodes, edges, labels = gp.graph(expr)

    print labels

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

    # On cree la courbe qui represente l'evolution de la taille des individus en fonction des generations 
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


#########################
# Suite du code generique
#########################

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
# pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
# pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# points ~= prediction
# func ~= real
def meanSquarredError(func, points):
    sqerrors = 0
    for p in points:
        inputs = p  
        outputs = p**4 + p**3 + p**2 + p
        sqerrors += (outputs - func(inputs))**2

    # On arrondit sinon suraprentissage
    res = round(sqerrors / len(points), 10)
    return res,
   

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x

    return meanSquarredError(func, points)


toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    
    best_individual = hof[0]

    # Affiche l'arbre DEAP qui represente l'exp symbolique du modele le plus proche
    print "\n", best_individual
    print "best_individual.arity[4]", best_individual[4].arity
    print "[0]", best_individual[0].name
    print "[1]", best_individual[1].name
    print "[2]", best_individual[2].name
    print "[3]", best_individual[3].name
    print "[4]", best_individual[4].name
    print "[5]", best_individual[5].name
    print "[6]", best_individual[6].name
    print "[7]", best_individual[7].name
    print "[8]", best_individual[8].name
    print "[9]", best_individual[9].name
    print "[10]", best_individual[10].name

    print type(best_individual[9])

    # Affiche l'arbre representant le modele le plus proche
    display_graph(hof[0])

    # Affiche les statistiques au cours de l'evolution
    display_stats(log)

    return pop, log, hof

if __name__ == "__main__":
    main()
