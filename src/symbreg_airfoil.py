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
import csv
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Recuperation des donnees du fichier csv
reader = csv.reader(open("../res/airfoil/airfoil.dat",'r'), delimiter='\t')
reader.next()
points = list(reader)

# Partitionnement grossier d'un jeu de donnees test
# Les donnees restantes dans la liste points correspondent a la partition d'apprentissage
test = []

for i in range(len(points)/3):
    ind = random.randint(0, len(points)-1)
    test.append(points[ind])
    del points[ind]


# Pour le profil aeronautique, les donnees sont organisees comme ci-dessous
# Total de 5 variables d'entree
# Input variables : 
# 1. Frequency, in Hertzs. 
# 2. Angle of attack, in degrees. 
# 3. Chord length, in meters. 
# 4. Free-stream velocity, in meters per second. 
# 5. Suction side displacement thickness, in meters. 
# Output variable :
#6. Scaled sound pressure level, in decibels. 

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def testPerformance(individual, points):
    func = toolbox.compile(expr=individual)
    somme = 0

    for p in points:
        inputs = [float(p[x]) for x in range(len(p)-1)]
        outputs = float(p[len(p)-1])
        
        if abs((outputs - func(*inputs))) <= 5:
            somme += 1

    return float(somme) / len(points)


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
   

def evalSymbReg(points, individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and data in points

    return meanSquarredError(func, points)

toolbox.register("evaluate", evalSymbReg, points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


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


def main():
    random.seed(318)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats,
    #                               halloffame=hof, verbose=True)

    # Using HARM-GP
    pop, log = gp.harm(pop, toolbox, 0.5, 0.1, 100, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats,
                                   halloffame=hof, verbose=True)
    
    # Validation du modele sur l'ensemble d'apprentissage (en %)
    print("\nBase d'apprentissage : " + str(testPerformance(hof[0], points) * 100) + " %")

    # Validation du modele sur l'ensemble d'apprentissage (en %)
    print("Base de validation (test) : " + str(testPerformance(hof[0], test) * 100) + " %")

    # Affiche l'arbre representant le modele le plus proche
    display_graph(hof[0])

    # Affiche les statistiques au cours de l'evolution
    display_stats(log)

    return pop, log, hof

if __name__ == "__main__":
    main()
