import sys
import tensorflow as tf
import gp_deap_tensorflow as gpdt

####################
# Hyperparametres  #
####################

BATCH_SIZE = 50
LEARNING_RATE_SAMPLE = [0.1, 0.01, 0.001]
REG_SCALE = 1


##############################################
# Deap - TensorFlow Data structure           #
##############################################

class IndividualTensor:
    """ On l'utilise pour memoriser les informations liees a un individu / tenseur """

    __slots__ = ('individual', 'input', 'output', 'weights', 'learning_rate', 'mse', 'loss', 'train_op')

    def __init__(self, individual, input, output, weights, learning_rate):
        self.individual = individual
        self.input = input
        self.output = output
        self.weights = weights
        self.learning_rate = learning_rate

    def update(self, mse, loss, train_op):
        self.mse = mse
        self.loss = loss
        self.train_op = train_op
        

######################################################
# Definition des fonctions necessaires a TensorFlow  #
######################################################

# tensorflow_init
# Creation du graphe TensorFlow correspondant a l'individu

def tensorflow_init(individual, n_inputs, n_weights, optimized_weights):
    # On reinitialise le graphe TensorFlow
    tf.reset_default_graph()

    # Creation des noeuds d'entree et de sortie
    X = tf.placeholder(tf.float32, [None, n_inputs])
    Y = tf.placeholder(tf.float32, [None, 1])

    # Creation du tableau contenant les poids (coefficients)
    W = tf.Variable(optimized_weights)

    # Creation du placeholder contenant le learning rate
    learning_rate = tf.placeholder(tf.float32)

    # On cree l'objet individual_tensor
    individual_tensor = IndividualTensor(individual, X, Y, W, learning_rate)

    # Creation du modele (fonction de prediction)
    pred = gpdt.primitivetree_to_tensor(individual_tensor)

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

    # On met a jour l'objet individual_tensor apres avoir cree le modele et la fonction de cout
    individual_tensor.update(mse, loss, train_op)

    return individual_tensor


# tensorflow_train
# Entrainement du reseau de neurones avec le learning_rate passe en parametre
# Cette fonction retourne les coefficients optimaux ainsi que le MSE apres entrainement

def tensorflow_train(sess, individual_tensor, trX, trY, teX, teY, n_epochs, learning_rate_value):   
    # Recuperation des informationss sur le graphe
    X = individual_tensor.input
    Y = individual_tensor.output
    W = individual_tensor.weights
    learning_rate = individual_tensor.learning_rate
    mse = individual_tensor.mse
    loss = individual_tensor.loss
    train_op = individual_tensor.train_op

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


# tensorflow_run
# On appelle la fonction tensorflow_train trois fois avec un learning rate different
# Cette fonction retourne les coefficients optimaux

def tensorflow_run(individual_tensor, trX, trY, teX, teY, n_epochs):
    # Creation de la session (compilation du graphe)
    sess = tf.Session()

    # Initialisation des variables
    init = tf.initialize_all_variables()
    sess.run(init)

    ################# TMP #################
    # X = individual_tensor.input
    # Y = individual_tensor.output
    # mse = individual_tensor.mse
    # dictTest = { X: teX, Y: teY }
    # before = sess.run(mse, feed_dict=dictTest)
    ################# TMP #################

    # Initialisation avec une valeur de mse maximale
    best_weights = { 'weights': None, 'mse': sys.float_info.max, 'learning_rate': None }

    # On entraine le reseau trois fois avec un learning rate different
    for learning_rate in LEARNING_RATE_SAMPLE:
        w, mse = tensorflow_train(sess, individual_tensor, trX, trY, teX, teY, n_epochs, learning_rate)
        current_weights = { 'weights': w, 'mse': mse, 'learning_rate': learning_rate }

        if current_weights['mse'] < best_weights['mse']:
            best_weights = current_weights

    ################# TMP #################
    # print "Learning rate : " + str(best_weights['learning_rate']) + " | MSE : Before " + str(before) + " / After " + str(best_weights['mse'])
    ################# TMP #################

    # On retourne les poids optimaux obtenus avec le learning rate adequat
    return best_weights['weights']
