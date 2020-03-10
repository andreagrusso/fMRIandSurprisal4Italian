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
vocab = []


#second version of the history with the sum of the vectors
def history_vec_sum(word, idx):

    history_vec_notnorm = np.zeros([1,  1000])

    if word not in word_vectors:

        word_vec = np.zeros((1, 1000)) #for history with sum
        print('MISSING WORD!',word)

    else:
        word_vec = word_vectors[word]

    hist_count = 1 #this index counts the word in the history. It has to be no more than 4
    
    if idx>5:

        for word in all_words[idx-2:0:-1]:#skipping the word already use for trigram 
            
            if (word not in itstops) and (hist_count<5):
                history_vec_notnorm = history_vec_notnorm + word_vec
                print(word)
                hist_count += 1

    else:

        for word in all_words[idx-2:0:-1]:

            if (word not in itstops):
                history_vec_notnorm = history_vec_notnorm + word_vec

    hist_norm_factor = np.sum(history_vec_notnorm * context_w_freq)
    history_vec = np.divide(history_vec_notnorm, hist_norm_factor)
    return history_vec


def ngram_prob(triplette):
    
    tmp_score = list(trigram.full_scores(triplette,bos=False, eos=False))

    return 10**tmp_score[2][0]


def sem_similarity_estimation(word,history_vec_word):

    word_vec = np.reshape(np.asarray(word_vectors[word]),[1,1000])

    # finally we can estimate the second component
    semantic_similarity_unscaled = history_vec_word*word_vec
    semantic_similarity = np.sum(semantic_similarity_unscaled*context_w_freq)

    return semantic_similarity

def sem_similarity_estimation4content_w(content_word,history_vec_word,idx):

    word_vec = np.reshape(np.asarray(word_vectors[content_word]),[1,1000])
    
    semantic_similarity_unscaled = history_vec_word*word_vec
    semantic_similarity = np.sum(semantic_similarity_unscaled*context_w_freq)

    context_trigram = all_words[idx - 2] + ' ' + all_words[idx - 1]+ ' '+ content_word
    new_semantic_similarity = ngram_prob(context_trigram)*semantic_similarity

    return new_semantic_similarity


def norm_factor_estimation(idx,history_vec_word):

    den_factor = 0
    num_factor = 0

    den_factor = sum([sem_similarity_estimation4content_w(content_word,history_vec_word,idx)
        for content_word in word_vectors.keys() if content_word not in itstops])
    print(den_factor)

    num_factor = sum([ngram_prob(all_words[idx - 2] + ' ' + all_words[idx - 1]+ ' ' + content_word) 
        for content_word in word_vectors.keys() if content_word not in itstops])
    print(num_factor)

    norm_fact = num_factor / den_factor
    
    return norm_fact

     




os.system('clear')





# NEW UNSEEN TEXT
passage = 'gianna_words_noApos'#input('Name of the passage to analyze:  ')
text = os.path.join('input', passage +'.txt')

f=open(text, "r", encoding='utf-8-sig')
all_words=[row[0].lower() for row in csv.reader(f, delimiter = '\n')]
print('TEXT LOADED!')



# WORD VECTORS
f = open(os.path.join('input','word_vectors_vocab1000_105K.pkl'), 'rb')
word_vectors = pickle.load(f)
print('WORD VECTORS LOADED!')

# STOPWORDS
itstops = get_stop_words('it')
itstops.append("c'è")
itstops.append("c'era")
itstops.append("c'erano")
itstops.append("l'")
itstops.append("'")
itstops.append("dell'")
itstops.append("nell'")
itstops.append("un'")
itstops.append("quell'")
itstops.append("po'")
print('STOPWORDS LOADED!')


# NGRAM
ngram_path = os.path.join('input','paisa_3ord.lm')
#ngram_path = os.path.join(input_dir, 'input','paisa_5ord.lm')
print('NGRAM MODELS LOADED!')

trigram = kenlm.Model(ngram_path)
print('TRIGRAM MODEL LOADED!')

# UNIGRAM
f = open(os.path.join('input','paisa_freq.pkl'), 'rb')
unigram_freq = pickle.load(f)
print('UNIGRAM FREQUENCIES LOADED!')

# CONTEXT WORDS
# load context word estiamted with GLOVE
f = open(os.path.join('input','vocab1000.txt'), 'r')
csvreader = csv.reader(f, delimiter=' ')
for row in csvreader:
    vocab.append(row)
print('CONTEXT WORDS LOADED')

context_w_freq = [unigram_freq[tmp[0]] for tmp in vocab]
context_w = [tmp[0] for tmp in vocab]
context_w_freq = np.reshape(np.asarray(context_w_freq),[1,1000])

print('CONTEXT WORD NAME AND FREQUENCIES LOADED')

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
            sem_surp.append([idx,word,10**tmp_score[0][0]])
            lex_surp.append([idx,word,10**tmp_score[0][0]])

        if (idx == 1):
            tmp_score = list(trigram.full_scores(all_words[idx - 1] + ' ' + word, bos=True,eos=False))
            sem_surp.append([idx,word,10**tmp_score[1][0]])
            lex_surp.append([idx,word,10**tmp_score[1][0]]) 

        if (idx == 2):
            tmp_score = list(trigram.full_scores(all_words[idx - 2] + ' ' + all_words[idx - 1] + ' ' + word, bos= True,eos=False))
            sem_surp.append([idx,word,10**tmp_score[2][0]])
            lex_surp.append([idx,word,10**tmp_score[2][0]])

    else:

        if (word in itstops) or (word not in list(word_vectors.keys())):# WE NEED ONLY 3-GRAM PROBABILITY FOR STOPWORDS
            
            sem_surp.append([idx,word,ngram_prob(all_words[idx - 2] + ' ' + all_words[idx - 1]+ ' ' + word)])
            lex_surp.append([idx,word,ngram_prob(all_words[idx - 2] + ' ' + all_words[idx - 1]+ ' ' + word)])    

        else:

            if word not in word_vectors:

                word_vectors[word] = np.zeros((1, 1000)) #for history with sum
                print('MISSING WORD!',word)


            triplette = all_words[idx-2] + ' ' + all_words[idx - 1] + ' '+ word
            #print("TRIGRAM CONSIDERED: " + triplette)
                # WE NEED 3 COMPONENTS FOR CONTENT WORDS:
                # 1) 3-GRAM COMPONENT OF THE WORD
                # 2) SEMANTIC SIMILARITY COMPONENT
                # 3) NORMALIZATION FACTOR OVER ALL THE WORDS OF THE CORPUS

            # 1) 3-gram component
            print('TRIGRAM PROBABILITY ESTIMATION.......')
            ngram_comp = ngram_prob(triplette)
            #print("TRIGRAM PROBABILITY ESTIMATED!")

            # 2) semantic similarity component
            # history vector creation
            #print('HISTORY VECTOR CREATION.......')
            history_vec_word = history_vec_sum(word, idx)#,all_words,word_vectors,context_w_freq,itstops)
            history_vec_word = np.reshape(np.asarray(history_vec_word),[1,1000])
            #print("HISTORY VECTOR ESTIMATED!")

            #print('SEMANTIC SIMILARITY ESTIMATION........')
            semantic_similarity = sem_similarity_estimation(word,history_vec_word)#,context_w_freq)
            #print("SEMANTIC SIMILARITY BETWEEN THE CONSIDERED WORD AND ITS HISTORY ESTIMATED!")

            # 3) normalization factor for content word
            #print('NORMALIZATION..........')
            normalization_fact = norm_factor_estimation(idx,history_vec_word)
            #print("NORMALIZATION DONE!")

            # SEMANTIC SURPRISAL
            surprisal_val = ngram_comp * normalization_fact * semantic_similarity
            lex_ngram_surp = ngram_prob(triplette)
            sem_surp.append([idx, word,surprisal_val])
            lex_surp.append([idx, word, lex_ngram_surp])

            print("SEMANTIC SURPRISAL OF THE WORD:" + str(surprisal_val))
            print("LEXICAL SURPRISAL OF THE WORD:" + str(lex_ngram_surp))




f = open(os.path.join('output','semantic','sem_surprisal_'+str(passage)+'.txt'), 'wb')
pickle.dump(sem_surp,f)

f = open(os.path.join('output','lexical','lex_surprisal_'+str(passage)+'.txt'), 'wb')
pickle.dump(lex_surp,f)

print('DONE!')

