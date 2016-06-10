import tensorflow as tf
import numpy as np
import time
from load_utils import loadBoston

# Hyperparameters 

BATCH_SIZE = 50
NUM_EPOCHS = 100
NB_VAR = 1
NB_NEURON = 800
LEARNING_RATE = 0.001


# Load dataset
trX, trY, teX, teY = loadBoston()


def tensorflow_run(prediction, dictTrain, dictTest):
    # Define the loss function (MSE)
    loss = tf.reduce_mean(tf.square(prediction - Y))
    # Use a gradient descent as optimization method
    train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.9, momentum=0.5, epsilon=1e-10, 
                                         use_locking=False, name='RMSProp').minimize(loss)

    # Graph infos
    loss_disp = tf.scalar_summary("MSE (train)", loss)
    merged_display = tf.merge_all_summaries()

    # Graph infos on the test dataset
    loss_test_disp = tf.scalar_summary("MSE (test)", loss)

    # We compile the graph
    sess = tf.Session()

    # Write graph infos to the specified file
    writer = tf.train.SummaryWriter("/tmp/tflogs_computation", sess.graph, flush_secs=10)

    # Main loop
    for i in range(NUM_EPOCHS):
        # Display informations and plot them in tensorboard 
        begin = time.time()
        result = sess.run(merged_display, feed_dict=dictTrain)
        writer.add_summary(result, i)
        result = sess.run(loss_test_disp, feed_dict=dictTest)
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

        print "(done in {:.2f} seconds)".format(time.time()-begin)start_learning()

