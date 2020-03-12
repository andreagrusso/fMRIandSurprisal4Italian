# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Andrea gerardo Russo
PhD candidate in Neuroscience
BrainLab, University of Salerno, Fisciano (Sa), Italy
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from stop_words import get_stop_words
from scipy.stats import zscore
import math

##APostrophes index
ap_idx = [264,418,635,667,690,851,947,963,968,980,1053,1118,1304,1358,
          1406,1546,1558,1596,1667,1776,1784,1813]
ap_idx= list(np.asarray(ap_idx) - 1) 


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





##INPUT DIRECTORY
input_dir ='output'

lex_surp_file = 'lex_surprisal_gianna.txt'
sem_surp_file = 'sem_surprisal_gianna.txt'
surp_sb_t = 'skip_bigram_gianna.txt'


words =[]
sem_surp = []
lex_surp = []
surp_sb = []
index =[]


#surprisal opening#############################################################

f=open(os.path.join(input_dir,lex_surp_file),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    if idx not in ap_idx:
        index.append(val[0])
        words.append(val[1])
        lex_surp.append(val[2])

f=open(os.path.join(input_dir,sem_surp_file),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    if idx not in ap_idx:
        sem_surp.append(val[2])


        
f=open(os.path.join(input_dir,surp_sb_t),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    if idx not in ap_idx:
        surp_sb.append(val[1])

print('Files loaded!')



idx_content_words = [idx for idx,word in enumerate(words) if word not in itstops]



#SwS, LS and SkipBigramLS
data = np.empty((len(sem_surp),3))
probdata = np.empty((len(sem_surp),3))

sem_surp = np.asarray(sem_surp)
lex_surp = np.asarray(lex_surp)
surp_sb = np.asarray(surp_sb)





#vector of factor interpolation
k_value = np.linspace(0,1,101)

skip_bg_effect = np.zeros([len(words),len(k_value)])
log_skip_bg_effect = np.zeros([len(words),len(k_value)])

#estiating the skipnigram interpolation
for idx,kval in enumerate(k_value):
    skip_bg_effect[:,idx] = kval*lex_surp + (1-kval)*surp_sb
    tmp_log = -np.log10(skip_bg_effect[:,idx])
    tmp_log[tmp_log == np.inf] = np.max(tmp_log[tmp_log != np.inf])
    log_skip_bg_effect[:,idx] = tmp_log
    
        
#log_skip_bg_effect[log_skip_bg_effect == np.inf] = np.max(log_skip_bg_effect[log_skip_bg_effect != np.inf])
    


#average over the content words
surp_skip = list(np.average(skip_bg_effect[idx_content_words,:],axis=0))   
log_surp_skip = list(np.average(log_skip_bg_effect[idx_content_words,:],axis=0))   

#maximise probability --> minimize surprisal
#k_max = k_value[(surp_skip.index(max(surp_skip)))]

k_min = k_value[(log_surp_skip.index(min(log_surp_skip)))]

print("Minimum interpolation value: " + str(k_min))
###############################################################################

print("Calculting the best interpolated surprisal model")
best_surp = log_skip_bg_effect[:,log_surp_skip.index(min(log_surp_skip))]
#best_surp = k_min*np.array(lex_surp) + (1-k_min)*np.array(surp_sb) 

data[:,0] = -np.log10(sem_surp)
data[:,1] = -np.log10(lex_surp)
data[:,2] = best_surp

probdata[:,0] = sem_surp
probdata[:,1] = lex_surp
probdata[:,2] = k_min*np.array(lex_surp) + (1-k_min)*np.array(surp_sb)



plt.figure()
plt.plot(zscore(data))
plt.legend(['Sem Surp','Lex Surp','SkipBigram'])
plt.title('Surprisal values')

    
plt.figure()
plt.xlabel('Lambda')
plt.ylabel('Surprisal')
plt.plot(k_value,np.average(log_skip_bg_effect[idx_content_words,:],axis=0))
plt.title("Skip-bigram interpolation effect")
plt.savefig(os.path.join(input_dir,'Skip-bigram_interpolation.png'),dpi=600)



all_means = np.mean(data,axis=0)


print('Mean value SemSurp: ' + str(all_means[0]))
print('Mean value LexSurp: ' + str(all_means[1]))
print('Mean value SkipBrigramurp: ' + str(all_means[2]))

output_name = 'surprisal_data_' + datetime.today().strftime('%Y-%m-%d') +'.txt'
np.savetxt(os.path.join(input_dir,output_name),data)

output_name = 'surprisal_data_prob' + datetime.today().strftime('%Y-%m-%d') +'.txt'
np.savetxt(os.path.join(input_dir,output_name),probdata)
