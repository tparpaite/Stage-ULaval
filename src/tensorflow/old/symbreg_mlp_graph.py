import sys
import time
import tensorflow as tf
import numpy as np
import load_utils as load

# Hyperparameters 

BATCH_SIZE = 50
NUM_EPOCHS = 100
NB_NEURON = 800
LEARNING_RATE = 0.001


######################################
# Tensorflow Data structure          #
######################################

class MLP:
    """ On l'utilise pour memoriser les informations liees a un MLP
    """

    __slots__ = ('input', 'output', 'loss', 'train_op', 'graph_train', 'graph_test')

    def __init__(self, input, output, loss, train_op, graph_train, graph_test):
        self.input = input
        self.output = output
        self.loss = loss
        self.train_op = train_op
        self.graph_train = graph_train
        self.graph_test = graph_test


########################################################
# Definition des fonctions necessaires a Tensorflow    #
########################################################

def tensorflow_init(nb_arg):
    # Create input and output nodes
    X = tf.placeholder("float", [None, nb_arg])
    Y = tf.placeholder("float", [None, 1])

    # Create our weights matrix (and provide initialization info)
    w_hidden = tf.Variable(tf.random_normal([nb_arg, NB_NEURON], stddev=0.01))
    w_output = tf.Variable(tf.random_normal([NB_NEURON, 1], stddev=0.01))
    b_hidden = tf.Variable(tf.zeros([NB_NEURON]))

    # Define our model (how do we predict)
    # Hidden layer
    pred = tf.nn.sigmoid(tf.matmul(X, w_hidden) + b_hidden)
    # Output layer
    pred = tf.matmul(pred, w_output)
    
    # Define the loss function (MSE)
    loss = tf.reduce_mean(tf.square(pred - Y))
    # Use a gradient descent as optimization method
    # train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.9, momentum=0.5, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss)
    

    # Graph infos
    loss_disp = tf.scalar_summary("MSE (train)", loss)
    w_disp = tf.histogram_summary("W (hidden layer)", w_hidden)
    w_disp2 = tf.histogram_summary("W (output layer)", w_output)
    graph_train = tf.merge_all_summaries()

    # Graph infos on the test dataset
    graph_test = tf.scalar_summary("MSE (test)", loss)

    mlp = MLP(X, Y, loss, train_op, graph_train, graph_test)

    return mlp


def train_mlp(sess, writer, mlp, trX, trY, teX, teY):
    # Recuperation des informations
    X = mlp.input
    Y = mlp.output
    loss = mlp.loss
    train_op = mlp.train_op
    graph_train = mlp.graph_train
    graph_test = mlp.graph_test

    # Association des donnees chargees et des tensors d'entree/sortie
    dictTrain = {X: trX, Y: trY}
    dictTest = {X: teX, Y: teY}

    for i in range(NUM_EPOCHS):
        # Display informations and plot them in tensorboard 
        begin = time.time()
        result = sess.run(graph_train, feed_dict=dictTrain)
        writer.add_summary(result, i)
        result = sess.run(graph_test, feed_dict=dictTest)
        writer.add_summary(result, i)
        writer.flush()
        
        print "[{}]".format(i), " "
        trainPerf = sess.run(loss, feed_dict=dictTrain)
        testPerf = sess.run(loss, feed_dict=dictTest)
        print "Train/Test MSE : {:.4f} / {:.4f}".format(trainPerf, testPerf), " "
    
        # This is the actual training
        # We divide the dataset in mini-batches
        for start, end in zip(range(0, len(trX), BATCH_SIZE),
                          range(BATCH_SIZE, len(trX)+1, BATCH_SIZE)):

            # For each batch, we train the network and update its weights
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        print "(done in {:.2f} seconds)".format(time.time()-begin)

    return testPerf


def tensorflow_run(dataX, dataY, kfold):
    nb_arg = len(dataX[0])
    mlp = tensorflow_init(nb_arg)

    # We compile the graph
    sess = tf.Session()

    # Write graph infos to the specified file
    writer = tf.train.SummaryWriter("/tmp/tflogs_boston", sess.graph, flush_secs=10)

    # We must initialize the values of our variables
    init = tf.initialize_all_variables()
    sess.run(init)

    mse_sum = 0

    # On boucle sur le 5-fold (cross validation)
    for tr_index, te_index in kfold:
        trX, teX = dataX[tr_index], dataX[te_index]
        trY, teY = dataY[tr_index], dataY[te_index]

        mse = train_mlp(sess, writer, mlp, trX, trY, teX, teY)
        mse_sum += mse

    mse_mean = mse_sum / 5

    return mse_mean

    
###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dict_load): 
        err_msg = "Usage : python symbreg_mlp.py data_name\n"
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
    mse = tensorflow_run(dataX, dataY, kfold)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = arg + " | MSE : " + str(mse) + " | " + runtime + "\n"
    file = open("logbook_mse_tensorflow.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
