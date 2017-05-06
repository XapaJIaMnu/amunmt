#!/usr/bin/env python3
import tensorflow as tf
import numpy
import gzip
import bz2
from sklearn.model_selection import train_test_split
from tensorflow.python.client import timeline

class DataBatcher:
    """This is a data iterator that reads in the file and provides the decoder with minibatches"""
    def __init__(self, filename, batch_size):
        self.batch_size = batch_size
        self.train_file = None
        self.fileclosed = False
        if (filename[-3:] == ".gz"):
            self.train_file = gzip.open(filename,'rt')
        elif (filename[-4:] == ".bz2"):
            self.train_file = bz2.open(filename, 'rt')
        else:
            self.train_file = open(filename, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        """Feeds the next batch to the neural network"""
        X_vec = []
        X_wID = []
        Y = []

        current_batch_size = 0
        while current_batch_size < self.batch_size and not self.fileclosed:
            line = self.train_file.readline()
            if line == "": # Check if we reached end of file
                self.train_file.close()
                self.fileclosed = True
                break
            if line.strip() == "":
                continue
            scores, wordIDs, vec = line.strip().split(' ||| ')
            # We could be having multiple scores/wordIDs associated with the same state
            # We should unpack them and transform them in traditional form
            split_scores = scores.split(' ')
            split_wordIDs = wordIDs.split(' ')
            x_vec = [float(x) for x in vec.strip().split(' ')]

            #Loop over the scores and add datapoints
            for i in range(len(split_scores)):
                Y.append(split_scores[i])
                X_wID.append(split_wordIDs[i])
                X_vec.append(x_vec)
            current_batch_size = current_batch_size + 1

        if X_vec != []:
            return (numpy.array(X_vec).astype('float32'), numpy.array(X_wID).astype("int32"), numpy.array(Y).astype('float32'))
        else:
            raise StopIteration


def init_weight(shape, name, previous_weight):
    """Initialize a weight matrix"""
    if previous_weight is not None:
        return tf.Variable(previous_weight, name=name)
    else:
        weight = tf.random_normal(shape, dtype='float32', name=name, stddev=0.1)
    return tf.Variable(weight)

def forwardpass(X, X_ID, w_1, b_1):
    """Forward pass of the NN"""
    # Instead of doing full matrix multiplication, just consider the vocabulary IDs
    # that are in this batch
    selected_embeddings_w_1 = tf.gather(w_1, X_ID)
    selected_embeddings_b_1 = tf.gather(b_1, X_ID)

    # Transpose the matrices to their traditional representation
    selected_embeddings_w_1_t = tf.transpose(selected_embeddings_w_1, name="W1_T")
    selected_embeddings_b_1_t = tf.transpose(selected_embeddings_b_1, name="B1_T")
    
    # Multiply with X with W_1
    multiplication = tf.matmul(X, selected_embeddings_w_1_t, name="X_W1_T")

    # We want the first column here, because every vector of X corresponds to a single
    # vocabID and not each vector with all vocabIDs in the batch
    mult_first_column = tf.gather(tf.transpose(multiplication), 1, name="Y_column_1")

    # add the bias
    bias_addition = tf.add(mult_first_column, selected_embeddings_b_1_t, name="Y_hat")
    return bias_addition


def FFNN_train(data, hidden_layer, vocab, w_1_load=None, b_1_load=None):
    """Defines the feed forward neural network"""
    x_size = hidden_layer # Size of the hidden layer input
    y_size = vocab # Vocab size

    # tf Graph Input
    X = tf.placeholder("float", name="X", shape=[None, x_size])
    X_ID = tf.placeholder("int32", name="X_ID", shape=[None])
    Y = tf.placeholder("float", name="Y", shape=[None])

    # init weights
    w_1 = init_weight((y_size, x_size), 'w1', w_1_load) # Swapped around because tf.gather() is stupid
    b_1 = init_weight((y_size, 1), 'b1', b_1_load) # Swapped around because tf.gather() is stupid

    # Forward pass
    y_hat = forwardpass(X, X_ID, w_1, b_1)

    # Error
    cost = tf.reduce_sum(tf.pow((Y - y_hat), 2))

    # Use adam to optimize and initialize the cost
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    init_op = tf.global_variables_initializer()

    x_vec_train, x_vec_test, x_id_train, x_id_test, y_train, y_test = train_test_split(data[0], data[1], data[2], test_size=0.2, random_state=42)

    with tf.Session() as sess: #config=tf.ConfigProto(log_device_placement=True)
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        sess.run(init_op)

        # Fit all training data
        for epoch in range(100):
            sess.run(train_op, feed_dict={X: x_vec_train, X_ID: x_id_train, Y: y_train})
            #for (x_vec, x_id, y) in zip(x_vec_train, x_id_train, y_train):
            #    sess.run(train_op, feed_dict={X: x_vec.reshape(1,500), X_ID: x_id.reshape(1), Y: y.reshape(1)})#, options=run_options, run_metadata=run_metadata)

            # Create the Timeline object, and write it to a json
            #tl = timeline.Timeline(run_metadata.step_stats)
            #ctf = tl.generate_chrome_trace_format()
            #with open('timeline' + str(epoch) + '.json', 'w') as f:
            #    f.write(ctf)

            # Display logs per epoch step
            c = sess.run(cost, feed_dict={X: x_vec_test, X_ID: x_id_test, Y: y_test})
            print("Epoch:" + str(epoch + 1) + " cost= " + str(c))
        return (w_1.eval(), b_1.eval())

def get_dataset_error(data, w1, b1):
    # Load data
    X, X_ID, Y = tf.constant(data[0]), tf.constant(data[1]), tf.constant(data[2])

    # Load the model
    w_1 = tf.constant(w1)
    b_1 = tf.constant(b1)

    # Compute predictions
    y_hat = forwardpass(X, X_ID, w_1, b_1)
    cost = tf.reduce_sum(tf.pow((Y - y_hat), 2))

    # Init vars
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        c = sess.run(cost)
        print("Total error: " + str(c))

def preprocess(filename):
    """Creates the dataset"""
    train_file = None
    if (filename[-3:] == ".gz"):
        train_file = gzip.open(filename,'rt')
    elif (filename[-4:] == ".bz2"):
        train_file = bz2.open(filename, 'rt')
    else:
        train_file = open(filename, 'r')
    X_vec = []
    X_wID = []
    Y = []
    for line in train_file:
        if line.strip() == "":
            continue
        scores, wordIDs, vec = line.strip().split(' ||| ')
        # We could be having multiple scores/wordIDs associated with the same state
        # We should unpack them and transform them in traditional form
        split_scores = scores.split(' ')
        split_wordIDs = wordIDs.split(' ')
        x_vec = [float(x) for x in vec.strip().split(' ')]

        #Loop over the scores and add datapoints
        for i in range(len(split_scores)):
            Y.append(split_scores[i])
            X_wID.append(split_wordIDs[i])
            X_vec.append(x_vec)

    train_file.close()
    return (numpy.array(X_vec).astype('float32'), numpy.array(X_wID).astype("int32"), numpy.array(Y).astype('float32'))

def train_NN(filename, hidden_layer, vocab):
    data = preprocess(filename)
    return FFNN_train(data, hidden_layer, vocab)
