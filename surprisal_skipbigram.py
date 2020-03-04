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
import gensim 
from scipy.special import softmax
import time
from sklearn.metrics.pairwise import cosine_similarity

############################################################################
# EMPTY VARIABLES
input_dir = '/home/arusso/vector_creation/new_semsurp'
history_dict_words = {}
ngram_comp = {}
sem_surprisal = []
vocab = []

def history_creation(all_words,idx,itstops,model):

    history_vec = np.zeros([1,300])
    hist_count = 1 #this index counts the word in the history. It has to be no more than 4
    if idx>5:
        for word in all_words[idx-2:0:-1]:#skipping the word already use for trigram 
            if (word not in itstops) and (hist_count<5) and (word in model.wv.vocab):
                history_vec = history_vec + model.wv[word]
                print(word)
                hist_count += 1
    else:
        for word in all_words[idx-2:0:-1]:#skipping the word already use for trigram 
            if (word not in itstops) and (word in model.wv.vocab):
                history_vec = history_vec + model.wv[word]

    history_vec = history_vec.reshape(1,300)

    return history_vec



def skip_bigram(c_word,all_words,idx,itstops,model,freq):

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
gen_dir = '/home/arusso/vector_creation/word2vec'
print('LOADING MODEL.........')
model = gensim.models.Word2Vec.load(os.path.join(gen_dir,'word2vec.model'))
print('Word2Vec model loaded!')



#frequency
freq = pickle.load(open(os.path.join(input_dir,'input','paisa_freq.pkl'),'rb'))
print("Frequency loaded!")


# NEW UNSEEN TEXT
passage = 'gianna_words_noApos'
text = os.path.join(input_dir,'input', passage +'.txt')
f=open(text, "r", encoding='utf-8')
all_words=[row[0].lower() for row in csv.reader(f, delimiter = '\n')]
print('TEXT LOADED!')

# STOPWORDS
itstops = get_stop_words('it')
itstops.append("c'è")
itstops.append("c'era")
itstops.append("c'erano")
print('STOPWORDS LOADED!')

# NGRAM
ngram_path = os.path.join(input_dir,'input','paisa_3ord.lm')
trigram = kenlm.Model(ngram_path)
print('TRIGRAM MODEL LOADED!')

#################################################################
freq_vec = [freq[word]/222153649 for word in list(model.wv.vocab) if word]#not in itstops]
freq_vec_norm = np.asarray(freq_vec)/sum(np.asarray(freq_vec))

word_vectors = np.asarray([model[word] for word in model.wv.vocab])# if word not in itstops])
word_name = [word for word in model.wv.vocab]#if word not in itstops]
word_vectors_norm = np.linalg.norm(word_vectors, axis = 1)

######################################################################
#OUTPUT VARIABLES

cs = [] #cosine similarity all words
cs_noFwords = [] #cosine similarty no function words
surp = [] # surprisal (3-gram)
surp_sb = [] #skip-bigram factor
 

####################SEMANTIC SURPRISAL#################################

for idx,word in enumerate(all_words):

#    if (word in model.wv.vocab):
    print("word: "+ word)
         

    print("Estimating skip-bigram factor....")
    surp_sb.append([idx,skip_bigram(word,all_words,idx,itstops,model,freq)])

#    else:
#        cs.append([word,0])
#        cs_noFwords.append([word,0])
#        surp_sb.append([word,0])


    
#for idx,word in enumerate(all_words):
#
#    if (idx < 3): 
#        if (idx == 0):
#            tmp_score = list(trigram.full_scores(word, bos=False,eos=False))
#            surp.append([idx,word,10**tmp_score[0][0]])
#        if (idx == 1):
#            tmp_score = list(trigram.full_scores(all_words[idx - 1] + ' ' + word, bos=True,eos=False))
#            surp.append([idx,word,10**tmp_score[1][0]]) 
#        if (idx == 2):
#            tmp_score = list(trigram.full_scores(all_words[idx - 2] + ' ' + all_words[idx - 1] + ' ' + word, bos= True,eos=False))
#            surp.append([idx,word,10**tmp_score[2][0]])
#
#        # WHEN WE ARE AT 8th WORD WE CAN ESTIMATE THE SEMANTIC SURPRISAL
#    else:
#        # WE NEED ONLY 3-GRAM PROBABILITY FOR STOPWORDS
#        ngram  = all_words[idx-3] + ' ' +all_words[idx-2] + ' ' + all_words[idx - 1] + ' '+ word
#        if idx == len(all_words):
#            tmp_score = list(trigram.full_scores(ngram, bos= False,eos=True))
#            surp.append([idx,word,10**tmp_score[2][0]])
#        else:
#            tmp_score = list(trigram.full_scores(ngram, bos= False,eos=False))
#            surp.append([idx,word,10**tmp_score[2][0]])


            

#f1 = open(os.path.join(input_dir,'output','lexical','lex_surprisal_3gram_nosent_'+str(passage)+'.txt'), 'wb')
#pickle.dump(surp,f1)
f2 = open(os.path.join(input_dir,'output','lexical','skip_bigram_nosent_outlocalcontext_'+str(passage)+'_v02.txt'), 'wb')
pickle.dump(surp_sb,f2)
#f3 = open(os.path.join(input_dir,'output','sem_similarity','cs_allwords_outside_local_context_'+str(passage)+'.txt'), 'wb')
#pickle.dump(cs,f3)
#f4 = open(os.path.join(input_dir,'output','sem_similarity','cs_zeroFwords_outside_local_context_'+str(passage)+'.txt'), 'wb')
#pickle.dump(cs_noFwords,f4)
