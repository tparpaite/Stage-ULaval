#    This file is part of DEAP.
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

import sys
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
import load_utils as load

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# Hyperparameters

POP_SIZE = 300
N_GEN = 40


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


def create_toolbox(pset):
    # Caracteristiques de l'individu et de la fitness
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Creation de la toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_symbreg, toolbox=toolbox, teX=None, teY=None)
    toolbox.register("select", tools.selTournament, tournsize=3)
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
# Boucle principale GP sans harm        #
#########################################

def deap_run(dataX, dataY, kfold):
    random.seed(318)

    mse_sum = 0
    size_sum = 0
    logbook_list = []
    n_args = len(dataX[0])
    pset = create_primitive_set(n_args)
    toolbox = create_toolbox(pset)

    # On split et boucle sur les 5-fold en updatant l'evaluate
    for tr_index, te_index in kfold:
        trX, teX = dataX[tr_index], dataX[te_index]
        trY, teY = dataY[tr_index], dataY[te_index]

        update_toolbox_evaluate(toolbox, teX, teY)

        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        # Classic GP (attention au bloat)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, N_GEN, stats=mstats,
                                       halloffame=hof, verbose=True)
    
        best_individual = hof[0]
        mse = best_individual.fitness.values[0]
        size = best_individual.height
        mse_sum += mse
        size_sum += size

        logbook_list.append(log)

    logbook = merge_logbook(logbook_list)

    # On retourne la moyenne du MSE et size obtenue en appliquant la 5-fold cross-validation
    mse_mean = (mse_sum / 5)[0]
    size_mean = size_sum / 5

    return mse_mean, size_mean, logbook


################
# Main call    #
################

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dict_load): 
        err_msg = "Usage : python symbreg_deap.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([key for key in load.dict_load.keys()]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    arg = sys.argv[1]
    run = "load." + load.dict_load[arg]
    dataX, dataY, kfold = eval(run)

    # Aprentissage automatique
    begin = time.time()
    mse, size, logbook = deap_run(dataX, dataY, kfold)
    deap_runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le logbook et le mse en dur
    logbook_filename = "logbook_" + arg + ".pkl"
    pickle.dump(logbook, open(logbook_filename, "w"))

    log_mse = arg + " | MSE : " + str(mse) + " | size : " + str(size) + " | " + deap_runtime + "\n"
    file = open("logbook_mse_deap.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
