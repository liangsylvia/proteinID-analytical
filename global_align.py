# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:40:05 2019

@author: sylvi
"""

'''Protein identification by global alignment using Needleman-Wunsch algorithm.'''

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
        signal = []
        for l in range(length_seq): 
            if sequence[l] == 'K': 
                signal.extend(np.ones(transloc_time))
            else: 
                signal.extend(np.zeros(transloc_time))
        reference_set.append(signal)
        reference_set.append(signal[::-1])
    return reference_set 

# Generate random protein test signals. Each residue's translocation time is 10 +/- 10 ms. 
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
        X.append(signal)
        Y.append(num)
    return X, Y

mismatch_penalty = -1
gap_penalty = -1
match_score = 1

# Needleman-Wunsch score algorithm
# Create a 2D matrix of scores of possible alignments between test signal and reference signal. 
# First row and first column represent gaps. Penalize -1 for mismatches and gaps, and reward +1 for matches.
# Score depends on highest evaulated score from top, left, and diagonal neighbor.
def needleman_wunsch(test_signal, ref_signal):
    k = len(test_signal)
    l = len(ref_signal)
    F = np.zeros(shape = (k+1, l+1))
    for i in range(k):
        for j in range(l):  
            if (i == 0) or (j == 0): 
                F[i,j] = gap_penalty*(i+j)
            else:
                if test_signal[i-1] == ref_signal[j-1]:
                    diagonal = F[i-1, j-1] + match_score
                else:
                    diagonal = F[i-1, j-1] + mismatch_penalty
                up = F[i-1, j] + gap_penalty
                left = F[i, j-1] + gap_penalty
                F[i, j] = max(diagonal, up, left)
    return F

# Determine best alignment by tracing the highest scoring path through the matrix. 
# Start at the bottom right corner and move towards top left corner. 
def traceback(F):
    row, column = F.shape
    k = row-1
    l = column-1
    total_score = 0
    while k>0 and l>0:
        score = F[k, l]
        scorediag = F[k-1, l-1]
        scoreup = F[k, l-1]
        scoreleft = F[k-1,l]
        if score == scorediag + match_score or score == scorediag + mismatch_penalty:
#                    align_test.append(test_signal[k])
#                    align_ref.append(reference_signal[l])
            k = k-1
            l = l-1
        elif score == scoreup + gap_penalty: 
#                    align_test.append('-')
#                    align_ref.append(reference_signal[l])
            l = l-1
        elif score == scoreleft + gap_penalty:
#                    align_test.append(test_signal[k])
#                    align_ref.append('-')
            k = k-1
        total_score = total_score + score
    return total_score

# Evaluate whether global alignment can identify a protein signal.
# Use Needleman-Wunsch algorithm to determine score for best global alignment between a test signal and reference signal.
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
            score_matrix = needleman_wunsch(test_signal, ref_signal)
            best_score = traceback(score_matrix)
            allscores.append(best_score)
        prediction = np.floor(allscores.idx(max(allscores))/2)
        predicted.append(prediction)
    accuracy = (np.array(Y_test) == np.array(predicted)).mean()
    print('Accuracy is ', accuracy)

dataset = read_data('ten protein sequence.csv')
reference_set = create_reference(dataset)
X_test, Y_test = create_testset(dataset, 100)
evaluate(X_test, Y_test, reference_set)
        
                    
                    
            
            
            
            