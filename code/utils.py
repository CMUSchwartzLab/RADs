""" utils functions to support RAD and analysis
"""

import numpy as np
import pickle
from itertools import permutations
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns




def mask_mse(B, M, C, F):

    Y = np.absolute( np.multiply(M, B-np.dot(C, F)) )

    res = 1.0*np.sum(np.multiply(Y, Y))/np.sum(M)

    return res


def get_square_distance(v1, v2):
  
    v = v1 - v2
    square_distance = np.dot(v, v)
    
    return square_distance


def get_sum_square_distance(C, Cgt, indexCgt):
  
    sum_square_distance = 0
    for i, j in enumerate(indexCgt):
      v1 = C[:,i]
      v2 = Cgt[:,j]
      square_distance = get_square_distance(v1, v2)
      sum_square_distance += square_distance
      
    return 1.0*sum_square_distance/len(indexCgt)


def get_min_ssd_index(C, Cgt, IndexCgt):
  
    min_ssd = float("inf")
    min_indexCgt = None
    for indexCgt in IndexCgt:
      sum_square_distance = get_sum_square_distance(C, Cgt, indexCgt)
      if sum_square_distance < min_ssd:
        min_indexCgt = indexCgt
        min_ssd = sum_square_distance
      
    return min_indexCgt, min_ssd

def alignResult(C, F, Cgt, k):
    '''
    align inferC and inferF regarding ground truth
    '''
    IndexCgt = list(permutations(range(0, k)))

    min_indexCgt, min_ssd = get_min_ssd_index(C, Cgt, IndexCgt)

    # aligned predicted C and F matrices
    Cp = np.zeros(C.shape)
    Fp = np.zeros(F.shape)
    for i, j in enumerate(min_indexCgt):
      Cp[:,j] = C[:,i]
      Fp[j,:] = F[i,:]

    return Cp, Fp

def load_result(b, s, n, lam):
    simulated_path = '../simulated_data/%s_%s_%s.pickle' % (b, s, n)
    results_path = '../results/result_%s_%s_%s_%s.pickle' % (b, s, n, lam)
    with open(results_path, 'rb') as handle:
        result = pickle.load(handle)
        handle.close()
    with open(simulated_path, 'rb') as handle:
        sim_data = pickle.load(handle)
        handle.close()

    return sim_data, result

def eval_plot(truedata, inferdata, plotfig=True):
    # 1. assess the overall data
    Cgt = truedata['trueC']
    Fgt = truedata['trueF']
        
    Cp = inferdata['inferC']
    Fp = inferdata['inferF']
    
    r2f = r2_score(Fgt.reshape(-1),Fp.reshape(-1))
    d = Fgt.reshape(-1)-Fp.reshape(-1)
    mse = np.dot(d,d)/len(d)
    l1_loss = np.sum(np.abs(Cgt.reshape(-1) - Cp.reshape(-1)))/np.sum(Cgt.reshape(-1))
    
    r2c = r2_score(Cgt.reshape(-1),Cp.reshape(-1))
    # 2. assess the known and unknown data
    x, y = np.log2(Cgt.reshape(-1)+1),np.log2(Cp.reshape(-1)+1)
    if plotfig:
        fig = plt.figure(figsize=(10,5), dpi=300)
        ax = plt.subplot(1,2,1)
    
        plt.plot([0,1],[0,1],"--", label="$R_F^2$=%.3f, MSE=%.3f"%(r2f, mse),color="gray")
    
        plt.scatter(Fgt[0],Fp[0], label="Population 1")
        plt.scatter(Fgt[1],Fp[1], label="Population 2")
        plt.scatter(Fgt[2],Fp[2], label="Population 3")
        plt.scatter(Fgt[3],Fp[3], label="Population 4")
        plt.scatter(Fgt[4],Fp[4], label="Population 5")
        plt.scatter(Fgt[5],Fp[5], label="Population 6")
        
    
        # Hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    
        plt.legend(frameon=False)
    
        plt.xlabel("Ground truth abundance ($F$)")
        plt.ylabel("Estimated abundance ($\hat{F}$)")
    
        ax = plt.subplot(1,2,2)
    
        plt.plot([0, 15], [0, 15], "--", color="gray",label=r"$R_C^2$=%.3f, $L_1$ loss=%.3f"%(r2c,l1_loss))
    
        plt.scatter(x, y, s=1,  edgecolor=['blue'], alpha=0.2)
    
        # hide the right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
    
        plt.legend(frameon=False)
        plt.xlabel("Ground truth expression ($\log_2 C$)")
        plt.ylabel("Estimated expression ($\log_2 \hat{C}$)")
    
    return r2f, mse, r2c, l1_loss

