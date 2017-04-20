#!/usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "numOfSentences")
    exit()

for num in range(int(sys.argv[1])):
    finished = open("".join(['finished_', str(num), '.txt']), 'r')

    # find all the possible prefixes that were finished, with their best score
    prefixes = {}
    for line in finished:
        line = line.strip()
        words, score = line.split(' ||| ')
        score = float(score)
        words = words.split(',')
        for i in range(1,len(words) + 1):
            prefix = ",".join(words[0:i])
            if ( prefix not in prefixes ) or ( score > prefixes[prefix] ):
                prefixes[prefix] = score

    states = open("".join(['states_', str(num), '.txt']), 'r')

    for line in states:
        [words, state] = line.split(' ||| ')
        if words in prefixes:
            print(" ||| ".join([str(prefixes[words]), words.split(',')[-1], state]))
    states.close()
    finished.close()
