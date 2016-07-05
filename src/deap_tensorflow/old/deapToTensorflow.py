import tensorflow as tf
import numpy as np
import time

from deap import gp
from individualTensor import IndividualTensor

###################################################
# Convertion de DEAP vers TensorFlow              #
###################################################

# Retourne l'index de l'argument, qui represente la colonne correspondante
# dans la matrice d'inputs X

def get_arg_index(arg_name):
    return int(arg_name[3])


# Initialisation

def primitivetree_to_tensor(individual, nb_arg, nb_weights):
    # On reinitialise le graphe TensorFlow
    tf.reset_default_graph()

    # Creation des noeuds inputs et output
    X = tf.placeholder("float", [None, nb_arg])
    Y = tf.placeholder("float", [None, 1])

    # On utilise un dictionnaire pour stocker les differentes colonnes (var) en input
    dict_arg = [nb_arg]
    for i in range(nb_arg):
        dict_arg[i] = tf.slice(X, [0, i], [-1, i+1])        

    # Creation du tableau contenant les poids (coefficients)
    W = tf.Variable(tf.random_normal([nb_weights]))

    # On ajoute de l'information sur l'individu
    individual_tensor = IndividualTensor(individual, X, Y, W, dict_arg)

    retVal, last_id, w_index = primitivetree_to_tensor_bis(individual_tensor, 0, 0)

    # On met a jour le tensor proprement dit
    individual_tensor.tensor = retVal

    return individual_tensor


# primitivetree_to_tensor 
#
# Un individu DEAP represente une expression symbolique
# Pour le modeliser on utilise un arbre de type PrimitiveTree
# Cette fonction sert a convertir l'arbre en question vers un
# graphe (un tenseur) interpretable par TensorFlow
#
# Parametres :
# - (PrimitiveTree) individual, l'individu a convertir
# - (int) index, indique l'avancement de la convertion (recursion)
#
# Retour :
# - (Tensor) retVal, le graphe final (lorsque index == 0)
# - (Tensor) retVal, (int) last_id, le graphe intermediaire et l'id du dernier noeud traite

def primitivetree_to_tensor_bis(individual_tensor, index, w_index):
    individual = individual_tensor.individual
    x = individual[index]

    if x.arity == 2:
        left_tree, next_id, w_index = primitivetree_to_tensor_bis(individual_tensor, index + 1, w_index)
        right_tree, last_id, w_index = primitivetree_to_tensor_bis(individual_tensor, next_id + 1, w_index)
        retVal = create_tensor_node(individual_tensor, x, left_tree, right_tree, w_index)

    elif x.arity == 1:
        under_tree, last_id, w_index = primitivetree_to_tensor_bis(individual_tensor, index + 1, w_index)
        retVal = create_tensor_node(individual_tensor, x, under_tree, None, w_index)

    # x.arity == 0 (leaf)
    else: 
        retVal = create_tensor_node(individual_tensor, x, None, None, w_index)
        w_index += 1
        last_id = index

    return retVal, last_id, w_index


# create_tensor_node
#
# Cette fonction sert a creer un noeud du graphe Tensorflow (tenseur)
# Un noeud peut-etre une operation, auquel cas on indique les parametres
# Un noeud peut aussi etre un argument ou une constante (feuille)
#
#
# Parametres :
# - (Primitive, Terminal, rand101) x
#      | (Primitive) le nom de l'operation
#      | (Terminal) le nom de l'argument
#      | (rand101) la valeur de constante ephemere (coefficient, variable)
# - (Tensor) left_tree, le membre gauche de l'operation
# - (Tensor) right_tree, le membre droit de l'operation
#
# Retour :
# - (Tensor) le noeud correspondant a l'operation ou au terminal

def create_tensor_node(individual_tensor, x, left_tree, right_tree, w_index):
    # x est un noeud interne (operation)
    if isinstance(x, gp.Primitive):
        op = x.name
        
        if op == "add":
            return tf.add(left_tree, right_tree)
        
        elif op == "sub":
            return tf.sub(left_tree, right_tree)

        elif op == "mul":
            return tf.mul(left_tree, right_tree)

        elif op == "max_rectifier":
            return tf.maximum(left_tree, 0)

        elif op == "min_rectifier":
            return tf.minimum(left_tree, 0)

    # x est une feuille
    else:
        value = x.value

        # La feuille correspond a une constante ephemere (var)
        if isinstance(x, gp.rand101):
            W = individual_tensor.weights
            const_eph = tf.Variable(value, dtype="float")
            return tf.mul(W[w_index], const_eph)

        # La feuille correspond a un argument
        else:
            # On recupere les informations
            arg_index = get_arg_index(value)
            X = individual_tensor.input
            W = individual_tensor.weights
            dict_arg = individual_tensor.dict_arg

            # On cree le noeud en choissisant le bon argument (colonne)
            return tf.mul(dict_arg[arg_index], W[w_index])


###################################################
# TensorFlow computation                          #
###################################################

# Hyperparameters 

BATCH_SIZE = 5
LEARNING_RATE = 0.01


def tensorflow_run(individual_tensor, data_train, data_test, nb_epochs):
    # Recuperation des informations
    prediction = individual_tensor.tensor
    X = individual_tensor.input
    Y = individual_tensor.output
    W = individual_tensor.weights

    # Association of loaded data and input/output nodes
    trX = data_train[:, :1]
    trY = data_train[:, -1:]
    teX = data_test[:, :1]
    teY = data_test[:, -1:]

    dictTrain = {X: trX, Y: trY}
    dictTest = {X: teX, Y: teY}


    # Define the loss function (MSE)
    loss = tf.reduce_mean(tf.square(prediction - Y))
    # Use a RMS gradient descent as optimization method
    train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.9, momentum=0.5, epsilon=1e-10, 
                                         use_locking=False, name='RMSProp').minimize(loss)

    # Graph infos
    loss_disp = tf.scalar_summary("MSE (train)", loss)
    w_info = tf.histogram_summary("Weigts", W)
    merged_display = tf.merge_summary([loss_disp, w_info])

    # Graph infos on the test dataset
    loss_test_disp = tf.scalar_summary("MSE (test)", loss)

    # We compile the graph
    sess = tf.Session()

    # Write graph infos to the specified file
    # writer = tf.train.SummaryWriter("/tmp/tflogs_computation", sess.graph, flush_secs=10)

    # We must initialize the values of our variables
    init = tf.initialize_all_variables()
    sess.run(init)

    # Main loop
    for i in range(nb_epochs):
        ## Display informations and plot them in tensorboard 
        # begin = time.time()

        # result = sess.run(merged_display, feed_dict=dictTrain)
        # writer.add_summary(result, i)
        # result = sess.run(loss_test_disp, feed_dict=dictTest)
        # writer.add_summary(result, i)
        # writer.flush()
        
        # print "[{}]".format(i), " "
        # trainPerf = sess.run(loss, feed_dict=dictTrain)
        # testPerf = sess.run(loss, feed_dict=dictTest)
        # print "Train/Test MSE : {:.10f} / {:.10f}".format(trainPerf, testPerf), " "
    
        # This is the actual training
        # We divide the dataset in mini-batches
        for start, end in zip(range(0, len(trX), BATCH_SIZE),
                              range(BATCH_SIZE, len(trX)+1, BATCH_SIZE)):

            # For each batch, we train the network and update its weights
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # print "(done in {:.2f} seconds)".format(time.time() - begin)

    return sess.run(W)



        





