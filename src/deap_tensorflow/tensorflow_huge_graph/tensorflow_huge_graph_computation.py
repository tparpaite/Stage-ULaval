import sys
import numpy as np
import tensorflow as tf
import gp_tensorflow_huge_graph as gpth

####################
# Hyperparametres  #
####################

BATCH_SIZE = 50
# LEARNING_RATE_SAMPLE = [0.1, 0.01, 0.001]
LEARNING_RATE = 0.001
REG_SCALE = 1


##############################################
# Deap - TensorFlow Data structure           #
##############################################

class PopGraph:
    """ On l'utilise pour memoriser les informations liees a une population """

    __slots__ = ('pop_info', 'input', 'output', 'weights_tab', 'learning_rate', 'mse_tab', 'loss_tab', 'train_op_tab')

    def __init__(self, pop_info, input, output, weights_tab, learning_rate, mse_tab, loss_tab):
        self.pop_info = pop_info
        self.input = input
        self.output = output
        self.weights_tab = weights_tab
        self.learning_rate = learning_rate
        self.mse_tab = mse_tab
        self.loss_tab = loss_tab

    def update(self, index_individual, mse, loss):
        self.mse_tab[index_individual] = mse
        self.loss_tab[index_individual] = loss

    def update_train_op(self):
        n_individuals = len(self.pop_info)
        losses_sum = tf.add_n(self.loss_tab)
        losses_avg = tf.div(losses_sum, n_individuals)
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.5, epsilon=1e-10, 
                                                  use_locking=False, name='RMSProp').minimize(losses_avg)
 
        

######################################################
# Definition des fonctions necessaires a TensorFlow  #
######################################################


# On calcule le MSE par rapport a l'ensemble test
def mean_squarred_error(func, optimized_weights, teX, teY):
    sqerrors = 0
    n_elements = len(teX)

    for x, y in zip(teX, teY):
        sqerrors += (y - func(optimized_weights, *x)) ** 2

    return sqerrors / n_elements


# tensorflow_create_individual_subgraph
# Creation du la partie propre a chaque individu dans le huge graphe

def tensorflow_create_individual_subgraph(index_individual, individual, pop_graph):
    # Recuperation des informations sur le graphe principal
    Y = pop_graph.output
    W = pop_graph.weights_tab[index_individual]
    learning_rate = pop_graph.learning_rate

    # Creation du modele (fonction de prediction)
    pred = gpth.primitivetree_to_tensor(index_individual, pop_graph)

    # Calcul du MSE
    mse = tf.reduce_mean(tf.square(pred - Y))

    # Regularisation L2
    reg_l2 = tf.nn.l2_loss(W)
    reg_l2 = reg_l2 * REG_SCALE

    # Definition de la fonction de cout : le MSE auquel on ajoute une regularisation L2
    # C'est-a-dire qu'on penalise les configurations ou les poids sont eleves
    loss = mse + reg_l2
    
    # On utilise la descente du gradient comme fonction d'optimisation
    #train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.5, epsilon=1e-10, 
    #                                     use_locking=False, name='RMSProp').minimize(loss)

    # On met a jour l'objet pop_graph apres avoir cree le modele et la fonction de cout
    pop_graph.update(index_individual, mse, loss)
    

# tensorflow_init
# Creation du graphe TensorFlow correspondant a la population

def tensorflow_init(pop_info, n_inputs):
    n_individuals = len(pop_info)

    # On reinitialise le graphe TensorFlow
    tf.reset_default_graph()

    # Creation des noeuds d'entree et de sortie (communs a tous les individus)
    X = tf.placeholder(tf.float32, [None, n_inputs])
    Y = tf.placeholder(tf.float32, [None, 1])

    # Initialisation aleatoire du tableau contenant les poids pour chaque individu
    weights_tab = [None] * n_individuals

    for i in range(n_individuals):
        n_weights = pop_info[i]['n_weights']
        weights_tab[i] = tf.Variable(tf.random_normal([n_weights]))

    # Creation du placeholder contenant le learning rate
    learning_rate = tf.placeholder(tf.float32)

    # Creation du tableau contenant le mse, loss et train_op pour chaque individu
    mse_tab = [None] * n_individuals
    loss_tab = [None] * n_individuals
 
    # On cree l'objet pop_graph
    pop_graph = PopGraph(pop_info, X, Y, weights_tab, learning_rate, mse_tab, loss_tab)

    # Pour chaque individu on cree le sous-graphe correspondant
    for i in range(n_individuals):
        individual = pop_info[i]['individual']
        tensorflow_create_individual_subgraph(i, individual, pop_graph)

    # On met a jour le train_op dans un noeud commun
    pop_graph.update_train_op()
        
    return pop_graph


def tensorflow_train(pop_info, pop_graph, sess, trX, trY, teX, teY, n_epochs, learning_rate_value):
    # Recuperation des informationss sur le graphe
    X = pop_graph.input
    Y = pop_graph.output
    learning_rate = pop_graph.learning_rate
    train_op = pop_graph.train_op
    # W = pop_graph.weights_tab[index_individual]
    # mse = pop_graph.mse_tab[index_individual]

    # Lien entre le dataset et les noeuds d'entree/sortie
    dictTrain = { X: trX, Y: trY, learning_rate: learning_rate_value }
    dictTest = { X: teX, Y: teY, learning_rate: learning_rate_value }
    
    # ENTRAINEMENT
    for i in range(n_epochs):   
        # Entrainement sur une epoque
        # On divise le dataset en mini-batches (mini-lots)
        for start, end in zip(range(0, len(trX), BATCH_SIZE),
                              range(BATCH_SIZE, len(trX)+1, BATCH_SIZE)):

            # Pour chaque lot, on entraine le reseau et on met a jour les poids
            # On entraine en fait tous les sous-graphes (individus) en parallele
            sess.run(train_op, feed_dict={ X: trX[start:end], Y: trY[start:end], learning_rate: learning_rate_value })


# tensorflow_update_mse
# Cette fonction s'appelle apres avoir realise l'entrainement dans le but de mettre
# a jour pop_info (mse et optimized_weights)
# Retourne le MSE du meilleur individu

def tensorflow_update_mse(pop_info, pop_graph, sess, trX, trY, teX, teY):
    n_individuals = len(pop_info)

    # MSE maximum a l'initialisation
    best_mse = sys.float_info.max

    # On parcourt l'ensembles des individus
    for i in range(n_individuals):
        # Evaluation du MSE sur l'ensemble test
        # MAJ de pop_info

        func = pop_info[i]['func']
        optimized_weights = sess.run(pop_graph.weights_tab[i])
        pop_info[i]['optimized_weights'] = optimized_weights 
        pop_info[i]['mse'] = mean_squarred_error(func, optimized_weights, teX, teY)[0]

        if pop_info[i]['mse'] < best_mse:
            best_mse = pop_info[i]['mse']

    return best_mse


# tensorflow_run
# Cette fonction retourne l'indice du meilleur individu de la population apres train

def tensorflow_run(pop_info, pop_graph, trX, trY, teX, teY, n_epochs):
    n_individuals = len(pop_info)

    # Creation de la session (compilation du graphe)
    sess = tf.Session()

    # Initialisation des variables
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Entraine tous les individus en parallele
    tensorflow_train(pop_info, pop_graph, sess, trX, trY, teX, teY, n_epochs, LEARNING_RATE)

    # Pour chaque individu : on calcule le MSE en test
    # On met a jour pop_info, et on retourne le MSE du meilleur individu
    best_mse = tensorflow_update_mse(pop_info, pop_graph, sess, trX, trY, teX, teY)

    return best_mse 
