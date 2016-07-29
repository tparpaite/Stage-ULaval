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


####################
# Hyperparametres  #
####################

X_LIMIT = 10250
Y_LIMIT = 10000

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
def load_logbooks_gp(dataset):
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


# Retourne un dictionnaire qui contient les logbooks GP d'un dataset
def load_logbooks_stats(dataset):
    logbook_stats_dic = {}
    method_list = ['gpclassic', 'gpharm', 'linear', 'svm', 'mlp']#, 'gpcoef']

    for method in method_list:
        path = LOGBOOK_PATH + "logbook_stats/logbook_stats_" + method + "_" + dataset + ".pickle"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                logbook_stats_dic[method] = pickle.load(f)
        else:
            sys.stderr.write(path + " not found\n")
            sys.exit(1)

    return logbook_stats_dic


# Dessine la courbe en fonction des donnees contenues dans un logbook
# X : Nombre d'individus evalues
# Y : MSE en test du meilleur individu
def create_curve(fig, logbook, color, label):
    X = np.array(logbook.select("nevals"))
    Y = np.array(logbook.chapters["fitness"].select("min"))

    return fig.plot(X, Y, linestyle="-", marker="o", color=color, label=label, markevery=10)


# create_fig_mse
# Affiche les stats du MSE min en fonction du nb d'eval sous forme de graphe
# Chaque courbe represente une methode de GP differente
# X : Nombre d'individus evalues
# Y : MSE en test du meilleur individu
# Nom de la figure : "Meilleure adequation (erreur MSE) en test en fonction du nombre d'evalutions" 
def create_fig_mse(logbook_dic):
    fig, axes = plt.subplots(figsize=(15,6))

    # On ajuste les axes
    axes.set_xlabel("Nombre d'individus evalues")
    axes.set_ylabel("Adequation en test du meilleur individu (minimisation)")
    axes.set_xlim([0,X_LIMIT])
    axes.set_ylim([0,Y_LIMIT])
    
    # Creation des courbes
    curve_gpclassic = create_curve(axes, logbook_dic['gpclassic'], "b", "GP classique")
    curve_gpharm = create_curve(axes, logbook_dic['gpharm'], "r", "GP harm")
    #curve_gpcoef = create_curve(fig, logbook_dic['gpcoef'], "g", "GP coef")

    # Affichage graphique
    curves = curve_gpclassic + curve_gpharm
    labels = [c.get_label() for c in curves]
    axes.legend(curves, labels, loc="best")

    # Sauvegarde au format PDF
    dataset = logbook_dic['dataset']
    filename = FIG_PATH + "fig_gp_" + dataset + '.pdf'
    fig.savefig(filename)
    print filename + " successfully generated"


# create_fig_box
# Affiche les informations sur le mse final pour chaque methode avec un jeu de donnees
# Affiche les informations sur la taille finale d'un invididu (avec les differentes methodes de GP)
# Les informations sont affiches sous forme de BOX (min, max, mediane et quartiles)
def create_fig_box(dataset, logbook_stats_dic, stats_type):
    keys = logbook_stats_dic.keys()

    # On recupere les tableaux souhaites (la condition vient pour le cas ou stats_type vaut size)
    data_array = [logbook_stats_dic[key][stats_type] for key in keys if stats_type in logbook_stats_dic[key]]

    print data_array

    # On creer les boxs
    fig, axes = plt.subplots(figsize=(8,6))
    axes.boxplot(data_array, 0, '')

    axes.set_ylim([0,Y_LIMIT])

    # On affiche la legende correcte sur l'axe des abscisses
    axes.set_xticklabels(keys)

    # Sauvegarde au format PDF
    filename = FIG_PATH + "fig_" + stats_type + "_" + dataset + '.pdf'
    fig.savefig(filename)
    print filename + " successfully generated"


# Permet de fusionner des logbook (pour avoir la moyenne des stats sur 5-fold x4)
# Un des problemes ici a ete qu'en fixant une limite d'evolution de la GP en fonction
# du nombre d'evaluation et pas de generation, on se retrouve avec des logbooks qui
# n'ont pas le meme nombre d'entrees
def merge_logbook(logbook_list):
    # Nombre de logbook
    n_logbook = len(logbook_list)

    # On recupere le nombre d'entree parmi les logbooks (il est egal pour tous les logbooks)
    n_entry = len(logbook_list[0])

    # Le logbook qui contient les entrees fusionnees
    res = tools.Logbook()

    # Les champs a fusionner
    chapters = ["fitness", "size"]
    fields = ["avg", "min", "max", "std"]

    # On realise la fusion proprement dite
    for i in range(n_entry):
        record = {}

        # Nombre d'evaluation courante (la meme pour tous les logbooks)
        nevals = logbook_list[0][i]['nevals']

        # Moyenne de la gen courante
        gen_mean = np.mean([logbook[i]['gen'] for logbook in logbook_list])
    
        # Moyenne de chaque champ pour chaque chapitre
        for chapter in chapters:
            record[chapter] = {}
            for field in fields:
                record[chapter][field] = np.mean([logbook.chapters[chapter][i][field] for logbook in logbook_list])

        res.record(gen=gen_mean, nevals=nevals, **record)

    return res


def stats_run(dataset):
    # Creation de la figure MSE
    run = "load_logbooks_gp(\"" + dataset + "\")"
    logbook_dic = eval(run)
    create_fig_mse(logbook_dic)
    
    # Creation de la figure box pour size, mse en train et en test
    run = "load_logbooks_stats(\"" + dataset + "\")"
    logbook_stats_dic = eval(run)
    create_fig_box(dataset, logbook_stats_dic, 'mse_train')
    create_fig_box(dataset, logbook_stats_dic, 'mse_test')
    create_fig_box(dataset, logbook_stats_dic, 'size')


################
# Main call    #
################

def usage(argv):
    if len(sys.argv) != 2 or (not(sys.argv[1] in load.dataset_list) and sys.argv[1] != '-all'): 
        err_msg = "Usage : python stats.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        err_msg += "Pour generer tous les jeux de donnees utiliser l'option -all\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]

    if dataset == '-all':
        [stats_run(dataset) for dataset in load.dataset_list]
    else:
        stats_run(dataset)

    plt.show()


if __name__ == "__main__":
    main()
