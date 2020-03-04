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
from stop_words import get_stop_words
from scipy.stats import zscore


##STOPWORDS
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
print('STOPWORDS LOADED!')


##INPUT DIRECTORY
input_dir ='output'

lex_surp_file = 'lex_surprisal_gianna.txt'
sem_surp_file = 'sem_surprisal_gianna.txt'
surp_sb_t = 'skip_bigram_gianna.txt '


words =[]
sem_surp = []
lex_surp = []
surp_sb = []
index =[]


#surprisal opening#############################################################

f=open(os.path.join(input_dir,lex_surp_file),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    index.append(val[0])
    words.append(val[1])
    lex_surp.append(val[2])

f=open(os.path.join(input_dir,sem_surp_file),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    sem_surp.append(val[-1])


        
f=open(os.path.join(input_dir,surp_sb_t),'rb')
tmp = pickle.load(f)
for idx,val in enumerate(tmp):
    surp_sb.append(val[1])

print('Files loaded!')

#SwS, LS and SkipBigramLS
data = np.empty((len(sem_surp),3))

sem_surp = np.asarray(sem_surp)
lex_surp = np.asarray(lex_surp)
surp_sb = np.asarray(surp_sb)





#vector of factor interpolation
k_value = np.linspace(0,1,101)

skip_bg_effect = np.zeros([len(words),len(k_value)])
for idw, w in enumerate(words):
    for idx,kval in enumerate(k_value):
        if (lex_surp[idw]!=0) and (surp_sb[idw]!=0):
            tmp_val = kval*lex_surp[idw] + (1-kval)*surp_sb[idw]
            skip_bg_effect[idw,idx] = -np.log10(tmp_val)




surp_skip = list(np.mean(skip_bg_effect,0))   

k_min = k_value[(surp_skip.index(min(surp_skip)))]
print("Minimum interpolation value: " + str(k_min))
###############################################################################

print("Calculting the best interpolated surprisal model")
best_surp = k_min*np.asanyarray(lex_surp) + (1-k_min)*np.asarray(surp_sb) 

data[:,0] = sem_surp
data[:,1] = lex_surp
data[:,2] = surp_sb





plt.figure()
plt.plot(zscore(data))
plt.legend(['Sem Surp','Lex Surp','SkipBigram'])
plt.title('Surprisal values')

    
plt.figure()
plt.xlabel('Lambda')
plt.ylabel('Surprisal')
plt.plot(k_value,surp_skip)
plt.title("Skip-bigram interpolation effect")
plt.savefig(os.path.join(input_dir,'Skip-bigram_interpolation.png'),dpi=600)



all_means = np.mean(data,axis=0)
mean_ss = np.mean(-np.log10(data[:,0]))
mean_ls = np.mean(-np.log10(data[:,1]))

print('Mean value SemSurp: ' + str(-np.log10(all_means[0])))
print('Mean value LexSurp: ' + str(-np.log10(all_means[1])))

output_name = 'surprisal_data_SwS_LS_SBSurp.txt'
np.savetxt(os.path.join(input_dir,output_name),data)