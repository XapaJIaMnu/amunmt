#!/usr/bin/env python

import sys

finished = open('finished.txt', 'r')

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

hypotheses = open('hypotheses.txt', 'r')
states = open('states.txt', 'r')

for line in hypotheses:
    line = line.strip()
    state = states.readline().strip()
    if line in prefixes:
        print " ||| ".join([str(prefixes[line]), state])
