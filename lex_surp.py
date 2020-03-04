#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:59:32 2018

@author: andreagerardo
"""

import csv
import kenlm
import pickle
import numpy as np
import os
from stop_words import get_stop_words




# EMPTY VARIABLES
lex_surp = []
sem_surp = []
sem_surp_noFW =[]
vocab = []






def ngram_prob(triplette):
    
    tmp_score = list(trigram.full_scores(triplette))#,bos=False, eos=False))

    return 10**tmp_score[2][0]





# NEW UNSEEN TEXT
passage = 'gianna_words_noApos'#input('Name of the passage to analyze:  ')
text = os.path.join('input', passage +'.txt')

f=open(text, "r", encoding='utf-8-sig')
all_words=[row[0].lower() for row in csv.reader(f, delimiter = '\n')]
print('TEXT LOADED!')






# NGRAM
ngram_path = os.path.join('input','paisa_3ord.lm')
#ngram_path = os.path.join('input','paisa_5ord.lm')
print('NGRAM MODELS LOADED!')

trigram = kenlm.Model(ngram_path)
print('TRIGRAM MODEL LOADED!')



####################SEMANTIC SURPRISAL#################################
for idx, word in enumerate(all_words):
    # CONSIDERING THAT THE FIRST THREE WORDS CANNOT HAVE A HISTORY WE CAN APPLY
    # ONLY THE TRIGRAM PROBABLITY (FOR THE FIRST AND SECOND WORD WE WILL APPLY
    # UNIGRAM OR BIGRAM?)
    print("WORD: " + word)
    print("INDEX:" + str(idx))

    #the first 3 words are paricualr cases
    if (idx < 3): 
        if (idx == 0): 
            tmp_score = list(trigram.full_scores(word, bos=True,eos=False))
            lex_surp.append([idx,word,10**tmp_score[0][0]])

        if (idx == 1):
            tmp_score = list(trigram.full_scores(all_words[idx - 1] + ' ' + word, bos=True,eos=False))
            lex_surp.append([idx,word,10**tmp_score[1][0]]) 

        if (idx == 2):
            tmp_score = list(trigram.full_scores(all_words[idx - 2] + ' ' + all_words[idx - 1] + ' ' + word, bos= True,eos=False))
            lex_surp.append([idx,word,10**tmp_score[2][0]])

    else:
            

        triplette = all_words[idx-2] + ' ' + all_words[idx - 1] + ' '+ word
        
        print('TRIGRAM PROBABILITY ESTIMATION.......')
        ngram_comp = ngram_prob(triplette)
        
        lex_surp.append([idx, word, ngram_comp]) 






f = open(os.path.join('output','lexical','lex_surprisal_'+str(passage)+'.txt'), 'wb')
pickle.dump(lex_surp,f)




print('DONE!')

