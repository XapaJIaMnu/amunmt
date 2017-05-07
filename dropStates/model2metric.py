#!/usr/bin/env python2
"""This script is used to replace the final model scores with an arbitrary
metric. Initially that arbitrary metric is just going to be BLEU"""
from sys import argv
import sys
from metrics.sentence_bleu import SentenceBleuScorer

def complex_map_lookup(probmap, probability):
    """This function iterates to the map and returns the entry closest to probability.
    This is because of the floating point issues we've been having"""
    prob = float(probability)
    lowest_diff = 100
    best = None
    for item in probmap.keys():
        diff = abs(float(item) - prob)
        if diff < lowest_diff:
            best = item
            lowest_diff = diff
    return best

def score_sent(n_best_list, reference_sent):
    """Scores a sentence with a BLEU score"""
    bleu_scorer = SentenceBleuScorer('n=4') # @TODO is that a normal ngram length default for BLEU?
                                            # @TODO maybe do a couple different Ns and average them
    bleu_scorer.set_reference(reference_sent.split(' '))
    retmap = {} # Map of the form normalized_score -> bleu_score
    for line in n_best_list:
        [sentID, sent, total_score, normalized_score] = line
        bleu_score = bleu_scorer.score(sent.split(' '))
        retmap[normalized_score] = bleu_score
    return retmap

def update_finished(new_scores, dropStates_location, cur_sent_id):
    """Updates a finished file"""
    finished_filename = dropStates_location + '/' + "finished_" + str(cur_sent_id) + ".txt"
    finished_file = open(finished_filename, 'r')
    outfile_txt = ""
    # Update scores
    for line in finished_file:
        [context, score] = line.strip().split(' ||| ')
        rounded_str = str(round(float(score), 6))
        if rounded_str not in new_scores.keys():
            rounded_str = str(float(rounded_str) - 0.000001) # Python and C round differently sometimes. Fuck me
            if rounded_str not in new_scores.keys():
                # Reducing by one doesn't always work, so we try increasing by 1 (we do 2 to compensate for the previous change)
                rounded_str = str(float(rounded_str) + 0.000002)
        if rounded_str not in new_scores.keys(): # Sometimes we just can't guess the floating number ;/
            rounded_str = complex_map_lookup(new_scores, score)
        new_score = str(new_scores[rounded_str])
        outfile_txt = outfile_txt + context + ' ||| ' + new_score+ '\n'
    finished_file.close()
    # Overwrite the file
    finished_file = open(finished_filename, 'w')
    finished_file.write(outfile_txt)
    finished_file.close()
    

def update_scores(n_best_file, dropStates_location, reference_sents_file):
    """This will update the finished files with BLEU scores instead of
    model scores"""
    nbest = open(n_best_file, 'r')
    reference_sents = open(reference_sents_file, 'r')
    
    n_best_list = []
    cur_sent_id = 0
    for line in nbest:
        [sentID, sent, total_score, normalized_score] = line.strip().split(' ||| ')
        if int(sentID) == cur_sent_id:
            n_best_list.append([sentID, sent, total_score, normalized_score])
        else:
            #Get a reference and process the files
            reference = reference_sents.readline().strip()
            new_scores = score_sent(n_best_list, reference)
            update_finished(new_scores, dropStates_location, cur_sent_id)
            cur_sent_id = cur_sent_id + 1
            n_best_list = []
            n_best_list.append([sentID, sent, total_score, normalized_score])
    nbest.close()
    reference_sents.close()
            
if __name__ == "__main__":
    if len(argv) != 4:
        print("Usage: " + argv[0] + " n_best_file dropStates_location reference_sents_file")
        exit()
    update_scores(argv[1], argv[2], argv[3])
