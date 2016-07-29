import sys
import numpy as np
import tensorflow as tf
import gp_tensorflow_huge_graph as gpth

####################
# Hyperparametres  #
####################

BATCH_SIZE = 50
LEARNING_RATE_SAMPLE = [0.1, 0.01, 0.001]
REG_SCALE = 1


##############################################
# Deap - TensorFlow Data structure           #
##############################################

class PopGraph:
    """ On l'utilise pour memoriser les informations liees a une population """

    __slots__ = ('pop_info', 'input', 'output', 'weights_tab', 'learning_rate', 'mse_tab', 'loss_tab', 'train_op_tab')

    def __init__(self, pop_info, input, output, weights_tab, learning_rate, mse_tab, loss_tab, train_op_tab):
        self.pop_info = pop_info
        self.input = input
        self.output = output
        self.weights_tab = weights_tab
        self.learning_rate = learning_rate
        self.mse_tab = mse_tab
        self.loss_tab = loss_tab
        self.train_op_tab = train_op_tab

    def update(self, index_individual, mse, loss, train_op):
        self.mse_tab[index_individual] = mse
        self.loss_tab[index_individual] = loss
        self.train_op_tab[index_individual] = train_op
        

######################################################
# Definition des fonctions necessaires a TensorFlow  #
######################################################

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
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.5, epsilon=1e-10, 
                                         use_locking=False, name='RMSProp').minimize(loss)

    # On met a jour l'objet pop_graph apres avoir cree le modele et la fonction de cout
    pop_graph.update(index_individual, mse, loss, train_op)
    

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
    train_op_tab = [None] * n_individuals
 
    # On cree l'objet pop_graph
    pop_graph = PopGraph(pop_info, X, Y, weights_tab, learning_rate, mse_tab, loss_tab, train_op_tab)

    # Pour chaque individu on cree le sous-graphe correspondant
    for i in range(n_individuals):
        individual = pop_info[i]['individual']
        tensorflow_create_individual_subgraph(i, individual, pop_graph)
        
    return pop_graph


# tensorflow_train
# Entrainement du reseau de neurones avec le learning_rate passe en parametre
# Cette fonction retourne les coefficients optimaux ainsi que le MSE apres entrainement

def tensorflow_train_bis(index_individual, pop_info, pop_graph, sess, trX, trY, teX, teY, 
                         n_epochs, learning_rate_value):   
    # Recuperation des informationss sur le graphe
    X = pop_graph.input
    Y = pop_graph.output
    W = pop_graph.weights_tab[index_individual]
    learning_rate = pop_graph.learning_rate
    mse = pop_graph.mse_tab[index_individual]
    train_op = pop_graph.train_op_tab[index_individual]

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
            sess.run(train_op, feed_dict={ X: trX[start:end], Y: trY[start:end], learning_rate: learning_rate_value })

    mse_value = sess.run(mse, feed_dict=dictTest)

    # On retourne les coefficients optimaux ainsi que le MSE apres entrainement
    return sess.run(W), mse_value


# tensorflow_train
# On appelle la fonction tensorflow_train_bis trois fois avec un learning rate different

def tensorflow_train(index_individual, pop_info, pop_graph, sess, trX, trY, teX, teY, n_epochs):
    # Initialisation avec une valeur de mse maximale
    best_weights = { 'weights': None, 'mse': sys.float_info.max, 'learning_rate': None }
    
    # On entraine le reseau trois fois avec un learning rate different
    for learning_rate in LEARNING_RATE_SAMPLE:
        w, mse = tensorflow_train_bis(index_individual, pop_info, pop_graph, sess, 
                                      trX, trY, teX, teY, n_epochs, learning_rate)
        current_weights = { 'weights': w, 'mse': mse, 'learning_rate': learning_rate }

        if current_weights['mse'] < best_weights['mse']:
            best_weights = current_weights.copy()
    
    # On verifie que TensorFlow n'a pas diverge (dans ce cas il retourne NaN pour le mse)
    if best_weights['weights'] == None:
        n_weights = pop_info[index_individual]['weights']
        best_weights['weights'] = sess.run(tf.random_normal([n_weights]))

    # On met a jour pop_info
    pop_info[index_individual]['optimized_weights'] = best_weights['weights']
    pop_info[index_individual]['mse'] = best_weights['mse']
    

# tensorflow_run
# Cette fonction retourne l'indice du meilleur individu de la population apres train

def tensorflow_run(pop_info, pop_graph, trX, trY, teX, teY, n_epochs):
    n_individuals = len(pop_info)

    # Creation de la session (compilation du graphe)
    sess = tf.Session()

    # Initialisation des variables
    init = tf.initialize_all_variables()
    sess.run(init)
    
    # Premiere iteration
    best_index_individual = 0
    tensorflow_train(0, pop_info, pop_graph, trX, trY, teX, teY, n_epochs)

    # Entrainement des sous-graphes
    # On parallelise cette boucle dans l'ideal TODO
    for i in range(1, n_individuals):
        tensorflow_train(i, pop_info, pop_graph, sess, trX, trY, teX, teY, n_epochs)

        if pop_info[i]['mse'] < pop_info[best_index_individual]['mse']:
            best_index_individual = i

    return best_index_individual    
