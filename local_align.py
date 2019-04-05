# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:42:58 2019

@author: sylvi
"""

'''Protein identification by local alignment using Smith-Waterman algorithm.'''

import random 
import pandas as pd 
import numpy as np 

# Open sequence database
def read_data(filename):
    dataset = pd.read_csv(filename)
    dataset['sequence'] = dataset['sequence'].str.replace('\n', '')
    return dataset

transloc_time = 10
std_dev = 10

# Generate reference signal for each protein. Assume each residue's translocation time is 10 ms. 
def create_reference(dataset):
    num_protein, column = dataset.shape
    reference_set = []
    for p in range(num_protein):
        sequence = dataset.iloc[p, 1]
        length_seq = len(sequence)
        full_signal = []
        for l in range(length_seq): 
            if sequence[l] == 'K': 
                full_signal.extend(np.ones(transloc_time))
            else: 
                full_signal.extend(np.zeros(transloc_time))
        reference_set.append(full_signal)
        reference_set.append(full_signal[::-1])
    return reference_set

# Generate random protein test signal segments (1000 datapts). Each residue's translocation time is 10 +/- 10 ms. 
# Lysine label efficiency is 85 percent. 
def create_testset(dataset, testset_size):
    rows, column = dataset.shape
    X = []
    Y = []
    for b in range(testset_size):
        num = random.randint(0,rows-1)
        sequence = dataset.iloc[num,1]
        if num%2 == 0: 
            sequence = sequence[::-1]
        length_seq = len(sequence)
        signal = []
        for l in range(length_seq): 
            aa_time = int(round(np.random.normal(transloc_time, std_dev)))
            if aa_time < 0:
                aa_time = 0
            if sequence[l] == 'K': 
                label = random.randint(1,100)
                if label < 86:
                    signal.extend(np.ones(aa_time))
                else:
                    signal.extend(np.zeros(aa_time))
            else: 
                signal.extend(np.zeros(aa_time))
        segment_pos = random.randint(0, len(signal)-1000)
        signal_segment = signal[segment_pos:segment_pos+1000]
        X.append(signal_segment)
        Y.append(num)
    return X, Y

match_score = 1
mismatch_penalty = -1
gap_penalty = -1

# Smith-Waterman score algorithm
# Create a 2D matrix of scores of possible local alignments between test signal segment and reference signal. 
# First row and first column represent gaps. Penalize -1 for mismatches and gaps, and reward +1 for matches.
# Score depends on highest evaulated score from top, left, and diagonal neighbor.
def smith_waterman(test_signal, ref_signal):
    length_test = len(test_signal)
    length_ref = len(ref_signal)
    F = np.zeros(shape = (length_test, length_ref))
    max_score = 0
    max_pos = None
    for i in range(length_test):
        for j in range(length_ref):  
            if (i == 0) or (j == 0): 
                F[i,j] = 0
            else:
                if test_signal[i-1] == ref_signal[j-1]:
                    diagonal = F[i-1, j-1] + match_score
                else:
                    diagonal = F[i-1, j-1] + mismatch_penalty
                up = F[i-1, j] + gap_penalty
                left = F[i, j-1] + gap_penalty
                F[i, j] = max(diagonal, up, left, 0)
            if F[i,j] > max_score: 
                max_score = F[i,j] 
                max_pos = (i,j)
    return F, max_pos

# Determine best alignment by tracing the highest scoring path through the matrix. 
# Start at the max score position and move towards top left corner until segment ends. 
def traceback(F, max_pos):
    x ,y = max_pos
    total_score = 0
    while x>0 and y>0:
        score = F[x, y]
        scorediag = F[x-1, y-1]
        scoreup = F[x, y-1]
        scoreleft = F[x-1,y]
        if score == scorediag + match_score or score == scorediag + mismatch_penalty:
#                    align_test.append(test_signal[k])
#                    align_ref.append(reference_signal[l])
            x = x-1
            y = y-1
        elif score == scoreup + gap_penalty: 
#                    align_test.append('-')
#                    align_ref.append(reference_signal[l])
            y = y-1
        elif score == scoreleft + gap_penalty:
#                    align_test.append(test_signal[k])
#                    align_ref.append('-')
            x = x-1
            total_score = total_score + score
        else: 
            break
    return total_score

# Evaluate whether global alignment can identify a protein signal.
# Use Smith-Waterman algorithm to determine score for best local alignment between a test signal segment and reference signal.
# Highest alignment score determines best matching protein. Accuracy is calculated from # of correctly identified protein/# of test signals.
def evaluate(X_test, Y_test, reference_set):
    num_references = len(reference_set)
    num_examples = len(Y_test)
    predicted = []
    for n in range(num_examples):
        test_signal = X_test[n]
        allscores = []
        for r in range(num_references):
            ref_signal = reference_set[r]
            score_matrix, position = smith_waterman(test_signal, ref_signal, 1, -1, -1)
            best_score = traceback(score_matrix, position)
            allscores.append(best_score)
        prediction = np.floor(allscores.idx(max(allscores))/2)
        predicted.append(prediction)
    accuracy = (np.array(Y_test) == np.array(predicted)).mean()
    print('Accuracy is ', accuracy)


dataset = read_data('ten protein sequence.csv')
reference_set = create_reference(dataset)
X_test, Y_test = create_testset(dataset, 100)
evaluate(X_test, Y_test, reference_set)