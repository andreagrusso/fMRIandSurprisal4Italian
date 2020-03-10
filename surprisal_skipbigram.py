#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:59:32 2018

@author: andreagerardo
"""

import kenlm
import pickle
import numpy as np
import os
import csv
from stop_words import get_stop_words
import time

############################################################################
# EMPTY VARIABLES
history_dict_words = {}
ngram_comp = {}
sem_surprisal = []
vocab = []


def skip_bigram(c_word,all_words,idx,freq):

    sb_vec = []
    hist_count = 1 #this index counts the word in the history. It has to be no more than 4

    min_freq = min([freq[word] for word in freq.keys()])

    if idx>5:
        for word in all_words[idx-2:0:-1]:#skipping the word already use for trigram 
            if (word not in itstops) and (hist_count<5):
                tmp_score = list(trigram.full_scores(word + ' ' + c_word, bos=False,eos=False))
                if word in list(freq):
                    sb_vec.append(10**tmp_score[1][0]/freq[word]) #bigram
                else:
                    sb_vec.append(10**tmp_score[1][0]/min_freq) #bigram
    else:
        for word in all_words[idx-2:0:-1]:#skipping the word already use for trigram 
            if (word not in itstops):
                tmp_score = list(trigram.full_scores(word + ' ' + c_word, bos=False,eos=False))
                if word in list(freq):
                    sb_vec.append(10**tmp_score[1][0]/freq[word]) #bigram
                else:
                    sb_vec.append(10**tmp_score[1][0]/min_freq) #bigram
  
    if len(sb_vec) != 0:
        sb_factor = (1/len(sb_vec))*sum(np.asarray(sb_vec)) 
    else:
        sb_factor = 0

    return sb_factor


##################################################################################
os.system('clear')



################## WORD2VEC MODEL ################################################

#frequency
freq = pickle.load(open(os.path.join('input','paisa_freq.pkl'),'rb'))
print("Frequency loaded!")


# NEW UNSEEN TEXT
passage = 'gianna_words_noApos'
text = os.path.join('input', passage +'.txt')
f=open(text, "r", encoding='utf-8')
all_words=[row[0].lower() for row in csv.reader(f, delimiter = '\n')]
print('TEXT LOADED!')

# STOPWORDS
itstops = get_stop_words('it')

f = open(os.path.join('input','word_vectors_vocab1000_105K.pkl'), 'rb')
word_vectors = pickle.load(f)
print('WORD VECTORS LOADED!')

# NGRAM
ngram_path = os.path.join('input','paisa_3ord.lm')
trigram = kenlm.Model(ngram_path)
print('TRIGRAM MODEL LOADED!')

#################################################################
freq_vec = [freq[word]/222153649 for word in freq]#not in itstops]
#freq_vec_norm = np.asarray(freq_vec)/sum(np.asarray(freq_vec))



######################################################################
#OUTPUT VARIABLES

cs = [] #cosine similarity all words
cs_noFwords = [] #cosine similarty no function words
surp = [] # surprisal (3-gram)
surp_sb = [] #skip-bigram factor
 

####################SEMANTIC SURPRISAL#################################

for idx,word in enumerate(all_words):

    print("word: "+ word)
         

    print("Estimating skip-bigram factor....")
    surp_sb.append([idx,skip_bigram(word,all_words,idx,freq)])


f2 = open(os.path.join('output','lexical','skip_bigram_nosent_outlocalcontext_'+str(passage)+'_v02.txt'), 'wb')
pickle.dump(surp_sb,f2)

