import sys
import time
import os
import pickle
import tensorflow as tf
import numpy as np

sys.path.append('../../')
from datasets import load_utils as load


####################
# Hyperparametres  #
####################

BATCH_SIZE = 50
NUM_EPOCHS = 150


######################################
# Tensorflow Data structure          #
######################################

class MLP:
    """ On l'utilise pour memoriser les informations liees a un MLP """

    __slots__ = ('input', 'output', 'mse', 'loss', 'train_op')

    def __init__(self, input, output, mse, loss, train_op, keep_prob, keep_value):
        self.input = input
        self.output = output
        self.mse = mse
        self.loss = loss
        self.train_op = train_op
        self.keep_prob = keep_prob
        self.keep_value = keep_value


######################################
# Optimisation des hyperparametres   #
######################################

def tensorflow_hyperparameters(dataset, dataX, dataY, kf_array):
    filepath = "./hyperparameters/hypers_tensorflow_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################

    # On recupere le premier pli
    nb_inputs = len(dataX[0])
    kfold = list(kf_array[0])
    tr_index, te_index = kfold[0]
    trX, teX = dataX[tr_index], dataX[te_index]
    trY, teY = dataY[tr_index], dataY[te_index]
    
    # Initialisation avec une valeur de mse maximale
    best_params = { 'mse': sys.float_info.max }

    # Echantillonage de maniere uniforme dans l'espace logarithmique
    # Taux d'aprentissage : [-5, 2]
    learning_rate_log_sample = np.random.uniform(-5, 2, size=100)

    # Coefficient de la regularisation L2 : [-8, 1]
    reg_scale_log_sample = np.random.uniform(-8, 1, size=100)

    # Echantillonage lineaire
    # Probabilite de garder un neurone actif durant DROPOUT : [0, 1]
    keep_value_sample = np.random.uniform(0, 1, size=100)
    
    # Nombre de couche cachees : [1, 3]
    nb_hidden_layer_sample = np.random.randint(1, 4, size=100)
    
    # Nombre de neurones par couche cachee : [5-500]
    nb_hidden1_sample = np.random.randint(5, 501, size = 100)
    nb_hidden2_sample = np.random.randint(5, 501, size = 100)
    nb_hidden3_sample = np.random.randint(5, 501, size = 100)

    # On lance 100 fois l'entrainement du MLP sur le pli train/test avec les echantillons crees
    zip_param = zip(learning_rate_log_sample, reg_scale_log_sample, keep_value_sample,
                    nb_hidden_layer_sample, nb_hidden1_sample, nb_hidden2_sample, nb_hidden3_sample)
    for learning_rate_log, reg_scale_log, keep_value, nb_hidden_layer, nb_hidden1, nb_hidden2, nb_hidden3 in zip_param:    
        # On sort de l'echelle logarithmique
        learning_rate = 10 ** (learning_rate_log)
        reg_scale = 10 ** (reg_scale_log)        

        # On stocke les hyperparametres dans un dictionnaire
        hyperparameters = {
            'learning_rate': learning_rate, 
            'reg_scale': reg_scale,
            'keep_value': keep_value,
            'nb_hidden_layer': nb_hidden_layer, 
            'nb_hidden1': nb_hidden1,
            'nb_hidden2': nb_hidden2,
            'nb_hidden3': nb_hidden3,
            'mse': None
        }
        
        # Creation du MLP et de la session (compilation du graphe)
        mlp = tensorflow_init(hyperparameters, nb_inputs)
        sess = tf.Session()
        
        # Initialisation des variables
        init = tf.initialize_all_variables()
        sess.run(init)
        
        # Entrainement du MLP et evaluation de son MSE apres entrainement
        hyperparameters['mse'] = tensorflow_train(sess, mlp, trX, trY, teX, teY)
        
        # Sauvegarde des hyperparametres s'ils sont meilleurs
        if hyperparameters['mse'] < best_params['mse']:
            best_params = hyperparameters.copy()

    # On sauvegarde les hyperparametres optimaux en dur avec pickle
    with open(filepath, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params


########################################################
# Definition des fonctions necessaires a Tensorflow    #
########################################################

# Lors de la creation du modele (fonction de prediction)
#
# Pour chaque neurone on calcule la combinaison lineaire de ses entrees multipliees
# chacune par un poids, a laquelle on ajoute un biais puis on applique une fonction
# d'activation (dans notre cas sigmoid) au resultat 
#
# Pour chaque neurone
# x : array inputs | w : array weights | z = output | b = biase 
# z = sigmoid(w * x + b)
#
# Pour regulariser (reduire l'overfitting), on applique egalement une fonction DROPOUT 
# a chaque couche cachee (avant la couche output)
# Cela correspond a desactiver aleatoirement des neurones a chaque nouvelle iteration
# On cree pour cela un placeholder qui permet d'indiquer la probabilite qu'un neurone
# soit garde lors de la phase de DROPOUT. Cela nous permet notamment d'activer la 
# fonction DROPOUT pendant la phase d'aprentissage (keep_prob dans [0, 1])
# et de la desactiver pendant la phase de tests (keep_prob a 1)

def tensorflow_create_model(hyperparameters, nb_inputs, X, keep_prob):
    # On recupere les infos sur les hyperparametres optimaux
    nb_hidden_layer = hyperparameters['nb_hidden_layer']
    nb_hidden1 = hyperparameters['nb_hidden1']
    nb_hidden2 = hyperparameters['nb_hidden2']
    nb_hidden3 = hyperparameters['nb_hidden3']

    ###################
    # 1 couche cachee #
    ###################
    if nb_hidden_layer == 1:
        # Creation des dictionnaires permettant de stocker les couches
        weights = { 
            'w1':  tf.Variable(tf.random_normal([nb_inputs, nb_hidden1])),
            'out': tf.Variable(tf.random_normal([nb_hidden1, 1])) 
        }
        biases  = { 
            'b1':  tf.Variable(tf.random_normal([nb_hidden1])),
            'out': tf.Variable(tf.random_normal([1])) 
        }
        
        # Creation du modele (fonction de prediction)
        layer1 = tf.matmul(X, weights['w1']) + biases['b1']
        layer1 = tf.nn.sigmoid(layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob)

        pred = tf.matmul(layer1, weights['out']) + biases['out']
        

    #####################
    # 2 couches cachees #
    #####################
    if nb_hidden_layer == 2:
        # Creation des dictionnaires permettant de stocker les couches
        weights = { 
            'w1':  tf.Variable(tf.random_normal([nb_inputs, nb_hidden1])),
            'w2' : tf.Variable(tf.random_normal([nb_hidden1, nb_hidden2])),
            'out': tf.Variable(tf.random_normal([nb_hidden2, 1]))
        }
        biases  = { 
            'b1':  tf.Variable(tf.random_normal([nb_hidden1])),
            'b2':  tf.Variable(tf.random_normal([nb_hidden2])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        # Creation du modele (fonction de prediction)
        layer1 = tf.matmul(X, weights['w1']) + biases['b1']
        layer1 = tf.nn.sigmoid(layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob)

        layer2 = tf.matmul(layer1, weights['w2']) + biases['b2']
        layer2 = tf.nn.sigmoid(layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob)

        pred = tf.matmul(layer2, weights['out']) + biases['out']
            

    #####################
    # 3 couches cachees #
    #####################
    if nb_hidden_layer == 3:
        # Creation des dictionnaires permettant de stocker les couches
        weights = { 
            'w1':  tf.Variable(tf.random_normal([nb_inputs, nb_hidden1])),
            'w2':  tf.Variable(tf.random_normal([nb_hidden1, nb_hidden2])),
            'w3' : tf.Variable(tf.random_normal([nb_hidden2, nb_hidden3])),
            'out': tf.Variable(tf.random_normal([nb_hidden3, 1]))
        }
        biases  = { 
            'b1':  tf.Variable(tf.random_normal([nb_hidden1])),
            'b2':  tf.Variable(tf.random_normal([nb_hidden2])),
            'b3':  tf.Variable(tf.random_normal([nb_hidden3])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        # Creation du modele (fonction de prediction)

        layer1 = tf.matmul(X, weights['w1']) + biases['b1']
        layer1 = tf.nn.sigmoid(layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob)

        layer2 = tf.matmul(layer1, weights['w2']) + biases['b2']
        layer2 = tf.nn.sigmoid(layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob)

        layer3 = tf.matmul(layer2, weights['w3']) + biases['b3']
        layer3 = tf.nn.sigmoid(layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob)

        pred = tf.matmul(layer3, weights['out']) + biases['out']
    

    # On retourne le modele, c'est-a-dire la fonction de prediction ainsi que les poids et biais
    return pred, weights, biases


def tensorflow_init(hyperparameters, nb_inputs):
    # On reinitialise le graphe TensorFlow
    tf.reset_default_graph()

    # On recupere les infos sur les hyperparametres optimaux
    # On cast le learning_rate (gradient descent sur 32 bits seulement) 
    learning_rate = hyperparameters['learning_rate']
    learning_rate = tf.constant(learning_rate, tf.float32)
    reg_scale = hyperparameters['reg_scale']
    keep_value = hyperparameters['keep_value']

    # Creation des noeuds d'entree et de sortie
    X = tf.placeholder("float", [None, nb_inputs])
    Y = tf.placeholder("float", [None, 1])

    # Creation du placeholder contenant keep_prob pour DROPOUT
    keep_prob = tf.placeholder("float")

    # Creation des couches cachees et du modele (fonction de prediction)
    pred, weights, biases = tensorflow_create_model(hyperparameters, nb_inputs, X, keep_prob)    
    
    # Calcul du MSE
    mse = tf.reduce_mean(tf.square(pred - Y))

    # Regularisation L2
    reg_l2 = 0
    for w, b in zip(weights, biases):
        reg_l2 += tf.nn.l2_loss(weights[w]) + tf.nn.l2_loss(biases[b])

    reg_l2 = reg_l2 * reg_scale

    # Definition de la fonction de cout : le mse auquel on ajoute une regularisation L2 sur les poids
    # C'est-a-dire qu'on penalise les configurations ou les poids sont eleves
    loss = mse + reg_l2

    # On utilise la descente du gradient comme fonction d'optimisation
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.5, epsilon=1e-10,
                                         use_locking=False, name='RMSProp').minimize(loss)

    # Creation de l'objet de stockage du mlp
    mlp = MLP(X, Y, mse, loss, train_op, keep_prob, keep_value)

    return mlp


def tensorflow_train(sess, mlp, trX, trY, teX, teY):
    # Recuperation des informations sur le graphe
    X = mlp.input
    Y = mlp.output
    mse = mlp.mse
    loss = mlp.loss
    train_op = mlp.train_op

    # On active le DROPOUT en train, et on le desactive en test
    # Phase d'aprentissage, keep_prob prend la valeur de l'hyperparametre keep_value
    # Phase de tests, keep_prob prend la valeur 1
    keep_prob = mlp.keep_prob
    keep_value = 1

    # Lien entre le dataset et les noeuds d'entree/sortie
    dictTrain = {X: trX, Y: trY}
    dictTest = {X: teX, Y: teY, keep_prob: 1.0}

    for i in range(NUM_EPOCHS):
        # Calcul de la fonction de cout sur l'ensemble de test
        loss_value = sess.run(loss, feed_dict=dictTest)

        print "[{}]".format(i), " "
        print "Loss : {:.4f}".format(loss_value), " "
    
        # Entrainement a proprement parler
        # On divise le dataset en mini-batches (mini-lots)
        for start, end in zip(range(0, len(trX), BATCH_SIZE),
                          range(BATCH_SIZE, len(trX)+1, BATCH_SIZE)):

            # Pour chaque lot, on entraine le MLP et on met a jour les poids
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], keep_prob: keep_value})

    # On retourne le MSE obtenu a la fin de l'entrainement
    mse = sess.run(mse, feed_dict=dictTest)

    return mse


def tensorflow_run(hyperparameters, dataX, dataY, kf_array):
    nb_inputs = len(dataX[0])

    # Creation du graphe representant le MLP
    mlp = tensorflow_init(hyperparameters, nb_inputs)

    # Creation de la session (compilation du graphe)
    sess = tf.Session()

    # Initialisation des variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # On boucle sur le 5-fold x4 (cross validation)
    mse_sum = 0

    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]
        
            # Entrainement du MLP et evaluation de son MSE
            mse = tensorflow_train(sess, mlp, trX, trY, teX, teY)
            mse_sum += mse
        
    # On retourne le mse moyen
    return mse_sum / 20


###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_mlp.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "load.load_" + dataset + "()"
    dataX, dataY, kf_array = eval(run)

    # Recherche des hyperparametres optimaux (ou chargement si deja calcule)
    hyperparameters = tensorflow_hyperparameters(dataset, dataX, dataY, kf_array)

    # Execution
    begin = time.time()
    mse = tensorflow_run(hyperparameters, dataX, dataY, kf_array)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = dataset + " | MSE : " + str(mse) + " | " + runtime + "\n"
    file = open("logbook_mse_tensorflow.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
