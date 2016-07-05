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

# Pour la prediction de la popularite d'un article media :
# Total de 58 variables d'entree utiles
# Input variables : 
# 0. url: URL of the article (non-predictive) 
# 1. timedelta: Days between the article publication and the dataset acquisition (non-predictive) 
# 2. n_tokens_title: Number of words in the title 
# 3. n_tokens_content: Number of words in the content 
# 4. n_unique_tokens: Rate of unique words in the content 
# 5. n_non_stop_words: Rate of non-stop words in the content 
# 6. n_non_stop_unique_tokens: Rate of unique non-stop words in the content 
# 7. num_hrefs: Number of links 
# 8. num_self_hrefs: Number of links to other articles published by Mashable 
# 9. num_imgs: Number of images 
# 10. num_videos: Number of videos 
# 11. average_token_length: Average length of the words in the content 
# 12. num_keywords: Number of keywords in the metadata 
# 13. data_channel_is_lifestyle: Is data channel 'Lifestyle'? 
# 14. data_channel_is_entertainment: Is data channel 'Entertainment'? 
# 15. data_channel_is_bus: Is data channel 'Business'? 
# 16. data_channel_is_socmed: Is data channel 'Social Media'? 
# 17. data_channel_is_tech: Is data channel 'Tech'? 
# 18. data_channel_is_world: Is data channel 'World'? 
# 19. kw_min_min: Worst keyword (min. shares) 
# 20. kw_max_min: Worst keyword (max. shares) 
# 21. kw_avg_min: Worst keyword (avg. shares) 
# 22. kw_min_max: Best keyword (min. shares) 
# 23. kw_max_max: Best keyword (max. shares) 
# 24. kw_avg_max: Best keyword (avg. shares) 
# 25. kw_min_avg: Avg. keyword (min. shares) 
# 26. kw_max_avg: Avg. keyword (max. shares) 
# 27. kw_avg_avg: Avg. keyword (avg. shares) 
# 28. self_reference_min_shares: Min. shares of referenced articles in Mashable 
# 29. self_reference_max_shares: Max. shares of referenced articles in Mashable 
# 30. self_reference_avg_sharess: Avg. shares of referenced articles in Mashable 
# 31. weekday_is_monday: Was the article published on a Monday? 
# 32. weekday_is_tuesday: Was the article published on a Tuesday? 
# 33. weekday_is_wednesday: Was the article published on a Wednesday? 
# 34. weekday_is_thursday: Was the article published on a Thursday? 
# 35. weekday_is_friday: Was the article published on a Friday? 
# 36. weekday_is_saturday: Was the article published on a Saturday? 
# 37. weekday_is_sunday: Was the article published on a Sunday? 
# 38. is_weekend: Was the article published on the weekend? 
# 39. LDA_00: Closeness to LDA topic 0 
# 40. LDA_01: Closeness to LDA topic 1 
# 41. LDA_02: Closeness to LDA topic 2 
# 42. LDA_03: Closeness to LDA topic 3 
# 43. LDA_04: Closeness to LDA topic 4 
# 44. global_subjectivity: Text subjectivity 
# 45. global_sentiment_polarity: Text sentiment polarity 
# 46. global_rate_positive_words: Rate of positive words in the content 
# 47. global_rate_negative_words: Rate of negative words in the content 
# 48. rate_positive_words: Rate of positive words among non-neutral tokens 
# 49. rate_negative_words: Rate of negative words among non-neutral tokens 
# 50. avg_positive_polarity: Avg. polarity of positive words 
# 51. min_positive_polarity: Min. polarity of positive words 
# 52. max_positive_polarity: Max. polarity of positive words 
# 53. avg_negative_polarity: Avg. polarity of negative words 
# 54. min_negative_polarity: Min. polarity of negative words 
# 55. max_negative_polarity: Max. polarity of negative words 
# 56. title_subjectivity: Title subjectivity 
# 57. title_sentiment_polarity: Title polarity 
# 58. abs_title_subjectivity: Absolute subjectivity level 
# 59. abs_title_sentiment_polarity: Absolute polarity level 
# Output variable :
# 60. shares: Number of shares (target)


# Recuperation des donnees du fichier csv
reader = csv.reader(open("../../res/OnlineNewsPopularity/OnlineNewsPopularity.csv",'r'), delimiter=',')
reader.next()
points = list(reader)

# Partitionnement grossier d'un jeu de donnees test
# Les donnees restantes dans la liste points correspondent a la partition d'apprentissage
test = []

for i in range(len(points)/3):
    ind = random.randint(0, len(points)-1)
    test.append(points[ind])
    del points[ind]


def testPerformance(individual, points):
    func = toolbox.compile(expr=individual)
    somme = 0

    for p in points:
        inputs = [float(p[x]) for x in range(2, len(p)-1)]
        outputs = float(p[len(p)-1])
        
        if abs((outputs - func(*inputs))) <= 500:
            somme += 1

    return float(somme) / len(points)


# points -> prediction
# func -> real
def meanSquarredError(func, points):
    sqerrors = 0

    for p in points:
        inputs = [float(p[x]) for x in range(2, len(p)-1)]
        outputs = float(p[len(p)-1])
        sqerrors += (outputs - func(*inputs))**2

    # On arrondit sinon suraprentissage
    res = round(sqerrors / len(points), 10)
    return res,


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


#########################
# Suite du code generique
#########################


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 58)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
   

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

def main():
    random.seed(318)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 160, stats=mstats,
    #                               halloffame=hof, verbose=True)

    # Using HARM-GP
    pop, log = gp.harm(pop, toolbox, 0.5, 0.1, 50, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats,
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
