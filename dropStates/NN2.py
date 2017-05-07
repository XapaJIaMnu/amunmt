#!/usr/bin/env python3
import tensorflow as tf
import numpy
import gzip
import bz2
import sys
from sklearn.model_selection import train_test_split
from tensorflow.python.client import timeline

class DataBatcher:
    """This is a data iterator that reads in the file and provides the decoder with minibatches"""
    def __init__(self, filename, batch_size=1000):
        self.batch_size = batch_size
        self.train_file = None
        self.fileclosed = False

        # For preprocesssed files:
        self.current_idx = 0
        self.preprocessed = False

        # Open the file
        if (filename[-3:] == ".gz"):
            self.train_file = gzip.open(filename,'rt')
        elif (filename[-4:] == ".bz2"):
            self.train_file = bz2.open(filename, 'rt')
        elif (filename[-4:] == ".npy" or filename[-4:] == ".npz"):
            self.train_file = numpy.load(filename)
            self.batch_size = len(self.train_file[0][0])
            self.current_idx = 0
            self.preprocessed = True
        else:
            self.train_file = open(filename, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        """Feeds the next batch to the neural network"""
        if self.preprocessed:
            if self.current_idx < len(self.train_file):
                self.current_idx = self.current_idx + 1
                return self.train_file[self.current_idx - 1]
            else:
                raise StopIteration
        else:
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

    def preprocess_and_save(self, save_location):
        """Requires quite a bit of memory"""
        batches = []
        for minibatch in self:
            batches.append(minibatch)
        self.preprocessed = True
        self.train_file = batches
        numpy.save(save_location, batches)

    def rebatch(self, new_batch_size):
        """This is used in case we load a preprocessed file and we want to rebatch it"""
        if self.preprocessed:
            new_batch_list = []

            X_vec = []
            X_wID = []
            Y = []
            for minibatch in self.train_file:
                for i in range(len(minibatch[0])):
                    if len(X_vec) < new_batch_size:
                        X_vec.append(minibatch[0][i])
                        X_wID.append(minibatch[1][i])
                        Y.append(minibatch[2][i])
                    else:
                        new_batch_list.append((numpy.array(X_vec).astype('float32'), numpy.array(X_wID).astype("int32"), numpy.array(Y).astype('float32')))
                        X_vec = []
                        X_wID = []
                        Y = []
                        X_vec.append(minibatch[0][i])
                        X_wID.append(minibatch[1][i])
                        Y.append(minibatch[2][i])
            self.current_idx = 0
            self.train_file = new_batch_list
        else:
            sys.stderr.write("Rebatch is only available with preprocessed files. Just change the batch size with raw ones.")
            return -1

    def save(self, filename):
        """Saves a preprocessed model. Useful if we want to rebatch and save"""
        if self.preprocessed:
            numpy.save(filename, self.train_file)
        else:
            sys.stderr.write("Save is only available with preprocessed files. Used preprocess_and_save instead.")
            return -1


class FFNN:
    def __init__(self, hidden_layer=500, vocab_size=85000, batch_size=10000, w_1=None, b_1=None, model_filename=None):
        self.batch_size = batch_size

        self.x_size = hidden_layer # Size of the hidden layer input
        self.y_size = vocab_size # Vocab size

        # tf Graph Input
        self.X = tf.placeholder("float", name="X", shape=[None, self.x_size])
        self.X_ID = tf.placeholder("int32", name="X_ID", shape=[None])
        self.Y = tf.placeholder("float", name="Y", shape=[None])

        # init weights
        self.w_1 = None
        self.b_1 = None

        if model_filename is not None:
            self.load_model(model_filename)
        else:
            self.w_1 = self._init_weight((self.y_size, self.x_size), 'w1', w_1) # Swapped around because tf.gather() is stupid
            self.b_1 = self._init_weight((self.y_size, 1), 'b1', b_1) # Swapped around because tf.gather() is stupid

        # Forward pass
        self.y_hat = self.forwardpass()

        # Error
        self.cost = tf.reduce_sum(tf.pow((self.Y - self.y_hat), 2))

        # Use adam to optimize and initialize the cost
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()

        # Init arguments
        self.sess = tf.Session()
        self.sess.run(self.init_op)

    @staticmethod
    def _init_weight(shape, name, previous_weight):
        """Initialize a weight matrix"""
        if previous_weight is not None:
            return tf.Variable(previous_weight, name=name)
        else:
            weight = tf.random_normal(shape, dtype='float32', name=name, stddev=0.1)
        return tf.Variable(weight)

    def forwardpass(self):
        """Forward pass of the NN"""
        # Instead of doing full matrix multiplication, just consider the vocabulary IDs
        # that are in this batch
        selected_embeddings_w_1 = tf.gather(self.w_1, self.X_ID)
        selected_embeddings_b_1 = tf.gather(self.b_1, self.X_ID)

        # Transpose the matrices to their traditional representation
        selected_embeddings_w_1_t = tf.transpose(selected_embeddings_w_1, name="W1_T")
        selected_embeddings_b_1_t = tf.transpose(selected_embeddings_b_1, name="B1_T")
        
        # Multiply with X with W_1
        multiplication = tf.matmul(self.X, selected_embeddings_w_1_t, name="X_W1_T")

        # We want the first column here, because every vector of X corresponds to a single
        # vocabID and not each vector with all vocabIDs in the batch
        mult_first_column = tf.gather(tf.transpose(multiplication), 1, name="Y_column_1")

        # add the bias
        bias_addition = tf.add(mult_first_column, selected_embeddings_b_1_t, name="Y_hat")
        return bias_addition

    def train(self, train_set_files, test_set, max_iterations=10):
        current_generation = 0
        prev_error = None
        while current_generation < max_iterations:
            for filename in train_set_files:
                self._train_file(filename)

            error = self.get_error(test_set)
            print("Error after epoch " + str(current_generation) + ": " + str(error))

            # Stop training when the error starts growing
            if prev_error is None:
                prev_error = error
            elif prev_error > error:
                prev_error = error
            else:
                break
            current_generation = current_generation + 1

    def _train_file(self, filename):
        """Does one iteration over a file"""
        batches = DataBatcher(filename, self.batch_size)
        counter = 0
        for minibatch in batches:
            counter = counter + 1
            if counter % 100 == 0:
                print("On batch: " + str(counter) + "...")
            self._train_minibatch(minibatch)

    def get_error(self, filename):
        """Get the error of a set."""
        error = 0
        batches = DataBatcher(filename, self.batch_size)
        for minibatch in batches:
            error = error + self._get_error(minibatch)
        return error

    def _train_minibatch(self, minibatch):
        [x_vec, x_id, y_train] = minibatch
        self.sess.run(self.train_op, feed_dict={self.X: x_vec, self.X_ID: x_id, self.Y: y_train})

    def _get_error(self, minibatch):
        [x_vec, x_id, y_train] = minibatch
        return self.sess.run(self.cost, feed_dict={self.X: x_vec, self.X_ID: x_id, self.Y: y_train})

    def get_model(self):
        return (self.w_1.eval(session=self.sess), self.b_1.eval(session=self.sess))

    def save_model(self, filename):
        (w1, b1) = self.get_model()
        numpy.savez(filename, **{'w_1':w1, 'b_1':b1})

    def load_model(self, filename):
        model = numpy.load(filename)
        w_1 = model['w_1']
        b_1 = model['b_1']

        self.y_size, self.x_size = w_1.shape
        self.w_1 = self._init_weight((self.y_size, self.x_size), 'w_1', w_1)
        self.b_1 = self._init_weight((self.y_size, 1), 'b_1', b_1)

def update_nematus_model(nematus_model, our_model, save_location):
    """This updates the existing nematus model with our updated w_4 and b_1"""
    nematus = dict(numpy.load(nematus_model))

    # Make our matrices be the same shape as the nematus one.
    # The user must ensure the dimensions match
    w_1 = our_model[0]
    b_1 = our_model[1]
    w_1 = w_1.transpose()
    b_1 = b_1.reshape(b_1.shape[0],)

    # Update the file
    nematus['ff_logit_W'] = w_1
    nematus['ff_logit_b'] = b_1

    # save to the updated model
    numpy.savez(save_location, **nematus)


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
