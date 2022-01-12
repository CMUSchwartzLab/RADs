# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:00:31 2022

@author: Leo
"""

import pickle
import sys
import os 
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import utils
from sklearn.metrics import r2_score
import sRAD_v3 as sRAD
import time


#%%
#1. load the simulated data
#0. define the variable:
b = 0.0 # noise level on bulk data
s = 0.0 # noise level on singe cell data
n = 1   # bulk sample number
lam = 0.1   # penalty value

with open('../simulated_data/0.0_0.0_1.pickle', 'rb') as handle:
                data = pickle.load(handle)
result_dir = '../results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
   
# 1. get the true data
B = data['B']
C = data['trueC']
S = data['S']
F = data['trueF']
mu = data['mu']
K = C.shape[1]
k = S.shape[1]
y = K - k
m = B.shape[0]

# 2. initialize variables:
C1_init = S
C2_init = np.zeros((m, y))
C_init = np.hstack((C1_init, C2_init))
F_init = sRAD._quad_prog_BCmu2F(B, C_init) # solve F as initialization
mu_init = sRAD._linear_reg_mu(C1_init, S)

#%%
# 3. solve the deconvolution and save the result
# this may take a while, take a cup of coffee...
START_TIME = time.time()
inferC, inferF, infermu, list_err = sRAD._rna_coordescent(B, C1_init, C2_init, F_init, S, mu_init, lam)

results = {'inferC': inferC, 'inferF': inferF, 'infermu':infermu, 'k':k, 
           'listErr':list_err,}
with open('../result_%s_%s_%s_%s.pickle' % (b, s, n, lam), 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

END_TIME = time.time()
print('Finished the deconvolution in %0.2f s' % (END_TIME-START_TIME))
#%%

# 4. plot the result
b = 0.0
s = 0.0
n = 1
lam = 0.1

sim_data, result = utils.load_result(b, s, n, lam)
utils.eval_plot(sim_data, result)