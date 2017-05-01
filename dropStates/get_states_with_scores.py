#!/usr/bin/python3
import sys
from hashlib import md5

if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], "numOfSentences")
    exit()

for num in range(int(sys.argv[1])):
    finished = open("".join(['finished_', str(num), '.txt']), 'r')
    #Do a primitive progress bar
    sys.stderr.write("\rProcessing file %d out of %s" % (num, sys.argv[1]))

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

    common_states = {}
    hashstates = {}
    for line in states:
        if line.strip() == "":
            continue
        [words, state] = line.split(' ||| ')

        #Hash the state and store it
        hashstate = md5(str.encode(state)).hexdigest()
        if hashstate not in hashstates:
            hashstates[hashstate] = state

        #Record which words correspond to which hashes
        if hashstate not in common_states:
            common_states[hashstate] = [words]
        else:
            common_states[hashstate].append(words)

    #Prepare the states for dumping
    for hashstate in hashstates:
        words = common_states[hashstate]
        scores = []
        splitted_words = ""
        for word in words:
            if word in prefixes:
                scores.append(prefixes[word])
                splitted_words = splitted_words + word.split(',')[-1] + " "
        splitted_words = splitted_words[:-1]

        scores_str = ""
        for score in scores:
            scores_str = scores_str + str(score) + " "
        scores_str = scores_str[:-1]
        if splitted_words == "":
            continue
        print(" ||| ".join([scores_str, splitted_words, hashstates[hashstate]]))

    states.close()
    finished.close()
sys.stderr.write("\rFinished processing %s files." % sys.argv[1])
