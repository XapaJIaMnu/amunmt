#!/usr/bin/env python

import sys

if len(sys.argv) < 2:
    print "Usage: ", sys.argv[0], "numOfSentences"
    exit()

for num in xrange(int(sys.argv[1])):
    finished = open("".join(['finished_', str(num), '.txt']), 'r')

    # find all the possible prefixes that were finished, with their best score
    prefixes = {}
    for line in finished:
        line = line.strip()
        words, score = line.split(' ||| ')
        score = float(score)
        words = words.split(',')
        for i in xrange(1,len(words)):
            prefix = ",".join(words[0:i])
            if ( prefix not in prefixes ) or ( score > prefixes[prefix] ):
                prefixes[prefix] = score

    hypotheses = open("".join(['hypotheses_', str(num), '.txt']), 'r')
    states = open("".join(['states_', str(num), '.txt']), 'r')

    for line in hypotheses:
        line = line.strip()
        state = states.readline().strip()
        if line in prefixes:
            print " ||| ".join([str(prefixes[line]), state])
