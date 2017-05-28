#!/usr/bin/env python
"""Preprocessor for states used to train the NN"""
import gzip
import bz2
import sys
import os
import numpy
if sys.version_info > (3, 0):
    import pickle
else:
    import cPickle as pickle # python2 compatibility

from random import randrange, uniform

def gen_rand_words(vocab_size, forbidden, num_rand, low_bleu):
    """Generates random words with mock BLEU scores"""
    ret_list = []
    bleu_scorz = []

    while len(ret_list) < num_rand:
        newnum = randrange(0, vocab_size)
        if (newnum not in forbidden) and (newnum not in ret_list):
            bleu = uniform(0, low_bleu)
            bleu_scorz.append(bleu)
            ret_list.append(newnum)
    return list(zip(ret_list, bleu_scorz))

class DataBatcher:
    """This is a data iterator that reads in the file and provides the decoder with minibatches"""
    def __init__(self, filename, batch_size=1000, scale=1, hallucinate=False, hallucinate_factor=10, vocab_size=10000):
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
            if sys.version_info > (3, 0):
                self.train_file = bz2.open(filename, 'rt')
            else:
                self.train_file = bz2.BZ2File(filename, 'r') # python2 compatilibity
        elif filename[-4:] == ".npy" or filename[-4:] == ".npz":
            self.train_file = numpy.load(filename)
            self.batch_size = len(self.train_file[0][0])
            self.current_idx = 0
            self.preprocessed = True
        elif filename[-4:] == ".pkl":
            self.train_file = pickle.load(open(filename, 'rb'))
            self.current_idx = 0
            self.preprocessed = True
        else:
            self.train_file = open(filename, 'r')

        self.hallucinate = hallucinate
        self.num_hallucinate = hallucinate_factor
        self.vocab_size = vocab_size

    def __iter__(self):
        return self

    def next(self):
        """Iterator method for python2 compatibility"""
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

                if self.hallucinate:
                    extra_data = gen_rand_words(self.vocab_size, X_wID, self.num_hallucinate, min(Y))
                    current_batch_size = current_batch_size + len(extra_data)
                    for item in extra_data:
                        X_vec.append(x_vec)
                        X_wID.append(item[0])
                        Y.append(item[1])

            if X_vec != []:
                return (numpy.array(X_vec).astype('float32'), numpy.array(X_wID).astype("int32"), numpy.array(Y).astype('float32'))
            else:
                raise StopIteration

    def preprocess_and_split_and_save(self, save_location, threshold=1000000):
        """We should split large files anyways."""
        # Create the save_location if it's not present
        if not os.path.isdir(save_location):
            if os.path.exists(save_location):
                print("Provided save location is a file: " + save_location + ". Need a directory or path to directory to be created")
                exit()
            else:
                os.makedirs(save_location)
        # Do the computation
        appendstr = 0
        totallegth = 0
        batches = []
        for minibatch in self:
            batches.append(minibatch)
            totallegth = totallegth + len(minibatch[0])
            if totallegth > threshold:
                with open(save_location + '/' + str(appendstr) + ".pkl", 'wb') as infile:
                    pickle.dump(batches, infile)
                appendstr = appendstr + 1
                totallegth = 0
                batches = []
                sys.stderr.write("\rPreprocessed batch %s." % str(appendstr*threshold))
        if batches != []:
            with open(save_location + '/' + str(appendstr) + ".pkl", 'wb') as infile:
                pickle.dump(batches, infile)
        sys.stderr.write("\rPreprocessed %s batches in total.\n" % str((appendstr + 1)*threshold))

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

if __name__ == '__main__':
    """Preprocesses a bz2 or gzip or txt file into batches ready for training"""
    if len(sys.argv) < 5:
        print("Usage: " + sys.argv[0] + " filename, batch_size, scale_factor, output_directory")
    else:
        batch_preprocessor = DataBatcher(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        batch_preprocessor.preprocess_and_split_and_save(sys.argv[4])
