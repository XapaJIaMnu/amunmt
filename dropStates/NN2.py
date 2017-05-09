#!/usr/bin/env python3
"""Neural network for learning BLEU scores"""
from collections import namedtuple
import gzip
import bz2
import sys
import tensorflow as tf
import numpy

class DataBatcher:
    """This is a data iterator that reads in the file and provides the decoder with minibatches"""
    def __init__(self, filename, batch_size=1000, scale=1):
        self.batch_size = batch_size
        self.train_file = None
        self.fileclosed = False
        self.scale = scale #Scaling is only available when producing the model

        # For preprocesssed files:
        self.current_idx = 0
        self.preprocessed = False

        # Open the file
        if filename[-3:] == ".gz":
            self.train_file = gzip.open(filename, 'rt')
        elif filename[-4:] == ".bz2":
            self.train_file = bz2.open(filename, 'rt')
        elif filename[-4:] == ".npy" or filename[-4:] == ".npz":
            self.train_file = numpy.load(filename)
            self.batch_size = len(self.train_file[0][0])
            self.current_idx = 0
            self.preprocessed = True
        else:
            self.train_file = open(filename, 'r')

    def __iter__(self):
        return self

    def next(self): # python2 compatibility
        return self.__next__()

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
                    Y.append(float(split_scores[i])*self.scale) # We might want to scale the input to true percentages
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
    """Actual neural network to train the BLEU score multiple regression"""
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
        self.cost = tf.reduce_mean(tf.squared_difference(self.y_hat, self.Y))

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

            error = self.get_mean_error(test_set)
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

    def get_mean_error(self, filename):
        """Get the error of a set."""
        error = 0
        batches = DataBatcher(filename, self.batch_size)
        counter = 0
        for minibatch in batches:
            counter = counter + 1
            error = error + self._get_error(minibatch)
        return error/counter

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

class LookupSimulation:
    """We'll use this class to simulate the calculation that the model makes so we can see if the
    predicted score is the highest for this item that we are interested"""
    def __init__(self, model, queryFile=None, batch_size=1):
        self.w_1 = model[0].transpose()
        self.b_1 = model[1].reshape(model[1].shape[0],)

        self.inputFile = None
        if queryFile is not None:
            self.inputFile = DataBatcher(queryFile, batch_size)

    def getItem(self):
        if self.inputFile is not None:
            return self.inputFile.__next__()
        return -1

    def queryNext(self):
        """Queries the next item in the input file"""
        if self.inputFile is not None:
            (vec, vocabId, bleu) = self.getItem()
            bleu_vec = self.query(vec)
            return self.analyze(bleu_vec, vocabId, bleu)
        return -1

    def query(self, datavec):
        """Does the query"""
        return numpy.matmul(datavec,self.w_1) + self.b_1

    def analyze(self, bleu_vec, true_point, true_score):
        """Shows where the true point is in the distribution"""
        vocab_with_id = []
        for j in range(len(bleu_vec)):
            this_vec = []
            for i in range(len(bleu_vec.transpose())):
                this_vec.append((bleu_vec[j, i], i))
            this_vec.sort()
            this_vec.reverse()
            vocab_with_id.append(this_vec)

        retlist = []
        Analysis = namedtuple('Analysis', ('i', 'bleu', 'true_bleu', 'best_bleu'))
        for j in range(len(vocab_with_id)):
            sorted_bleu_vec = vocab_with_id[j]
            for i in range(len(sorted_bleu_vec)):
                if true_point[j] == sorted_bleu_vec[i][1]:
                    # Returns which consecutive score, the bleu score, the true_score and the best score
                    retlist.append(Analysis(i, sorted_bleu_vec[i], true_score[j], sorted_bleu_vec[0]))
        return retlist


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