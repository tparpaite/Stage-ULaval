#############################################################################
# stats.py                                                                  #
# Ce module permet de manipuler les stats recuperees lors de l'execution    #
# - Permet d'afficher un individu DEAP sous forme de graphe                 #
# - Fusionner des logbook                                                   #
# - Creer des figures et courbes pour representer graphiquement les stats   #
#############################################################################

import sys
import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append('../')
from datasets import load_utils as load
from deap import tools
from deap import gp

# Chemin relatif du repertoire logbook
LOGBOOK_PATH = "./logbook/"
FIG_PATH = "./fig/"

# Affiche l'individu passe en parametre sous forme d'abre infixe
def display_individual_graph(expr):
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


# Retourne un dictionnaire qui contient les logbooks GP d'un dataset
def load_logbooks(dataset):
    logbook_dic = {}
    logbook_dic['dataset'] = dataset
    gpmethod_list = ['gpclassic', 'gpharm']#, 'gpcoef']

    for gpmethod in gpmethod_list:
        path = LOGBOOK_PATH + "logbook_gp/logbook_" + gpmethod + "_" + dataset + ".pickle"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                logbook_dic[gpmethod] = pickle.load(f)
        else:
            sys.stderr.write(path + " not found\n")
            sys.exit(1)

    return logbook_dic   


# Dessine la courbe en fonction des donnees contenues dans un logbook
# X : Nombre d'individus evalues
# Y : MSE en test du meilleur individu
def create_curve(logbook, color, label):
    X = np.array(logbook.select("nevals"))
    Y = np.array(logbook.chapters["fitness"].select("min"))

    plt.plot(X, Y, linestyle="-", marker="o", color=color, label=label, markevery=0.1)
    

# Affiche les stats du MSE min en fonction du nb d'eval sous forme de graphe
# Chaque courbe represente une methode de GP differente
# X : Nombre d'individus evalues
# Y : MSE en test du meilleur individu
# Nom de la figure : "Meilleure adequation (erreur MSE) en test en fonction du nombre d'evalutions" 
def create_fig_mse(logbook_dic):
    plt.xlabel("Nombre d'individus evalues")
    plt.ylabel("Adequation en test du meilleur individu (minimisation)")

    axes = plt.gca()
    axes.set_xlim([0,10000])
    
    create_curve(logbook_dic['gpclassic'], "b", "GP classique")
    create_curve(logbook_dic['gpharm'], "r", "GP harm")
    #create_curve(logbook_dic['gpcoef'], "g", "GP coef")

    # Affichage graphique
    plt.legend(loc="best")
    plt.show()

    # Sauvegarde au format PDF
    dataset = logbook_dic['dataset']
    filename = FIG_PATH + "fig_gp_" + dataset + '.pdf'
    plt.savefig(filename)
    print filename + " successfully generated"


def max_entry(logbook_list):
    max = len(logbook_list[0])

    for logbook in logbook_list:
        if len(logbook) > max:
            max = len(logbook)

    return max


# Permet de fusionner des logbook (pour avoir la moyenne des stats sur 5-fold x4)
# Un des problemes ici a ete qu'en fixant une limite d'evolution de la GP en fonction
# du nombre d'evaluation et pas de generation, on se retrouve avec des logbooks qui
# n'ont pas le meme nombre d'entrees
def merge_logbook(logbook_list):
    # On recupere le nombre maximal d'entree parmi les logbooks
    n_entry = max_entry(logbook_list)

    # Le logbook qui contient les entrees fusionnees
    res = tools.Logbook()

    # Les champs a fusionner
    chapters = ["fitness", "size"]
    fields = ["avg", "min", "max", "std"]

    # On realise la fusion proprement dite
    for i in range(n_entry):
        record = {}

        # Moyenne du nombre d'evaluations
        n_mean = 0.0
        nevals_sum = 0.0
        for logbook in logbook_list:
            if i < len(logbook):
                n_mean += 1.0
                nevals_sum += logbook[i]["nevals"]

        nevals = nevals_sum / n_mean
    
        # Moyenne de chaque champ pour chaque chapitre
        for chapter in chapters:
            record[chapter] = {}
            for field in fields:
                # On parcourt les logbooks
                n_mean = 0.0
                nfield_sum = 0.0
                for logbook in logbook_list:
                    if i < len(logbook):
                        n_mean += 1.0
                        nfield_sum += logbook.chapters[chapter][i][field]

                record[chapter][field] = nfield_sum / n_mean

        res.record(gen=i, nevals=nevals, **record)

    return res


################
# Main call    #
################

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python stats.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "load_logbooks(\"" + dataset + "\")"
    logbook_dic = eval(run)

    # Creation de la figure MSE
    create_fig_mse(logbook_dic)


if __name__ == "__main__":
    main()
