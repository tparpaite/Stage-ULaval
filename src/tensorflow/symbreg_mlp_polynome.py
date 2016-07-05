import tensorflow as tf
import numpy as np
import time
from load_utils import loadPolynome

# Hyperparameters 

BATCH_SIZE = 10
NUM_EPOCHS = 500
NB_VAR = 1
NB_NEURON = 4
LEARNING_RATE = 0.01

# Load dataset
trX, trY, teX, teY = loadPolynome()


# Create input and output nodes
X = tf.placeholder("float", [None, NB_VAR])
Y = tf.placeholder("float", [None, 1])

# Create our weights matrix (and provide initialization info)
w_hidden = tf.Variable(tf.random_normal([NB_VAR, NB_NEURON], stddev=0.01))
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
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.9, momentum=0.1, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss)


# Graph infos
loss_disp = tf.scalar_summary("MSE (train)", loss)
w_disp = tf.histogram_summary("W (hidden layer)", w_hidden)
w_disp2 = tf.histogram_summary("W (output layer)", w_output)
merged_display = tf.merge_all_summaries()

# Graph infos on the test dataset
loss_test_disp = tf.scalar_summary("MSE (test)", loss)


# We compile the graph
sess = tf.Session()

# Write graph infos to the specified file
writer = tf.train.SummaryWriter("/tmp/tflogs_airfoil", sess.graph, flush_secs=10)

# We must initialize the values of our variables
init = tf.initialize_all_variables()
sess.run(init)

# Association of loaded data and input/output nodes
dictTrain = {X: trX, Y: trY}
dictTest = {X: teX, Y: teY}


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

    print "(done in {:.2f} seconds)".format(time.time()-begin)
