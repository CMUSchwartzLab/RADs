import pickle
import numpy as np
import utils
import sRAD_v3 as sRAD
import RAD
import os
import tqdm
import time
import sys
import pdb

test_parameters = {'b_noise': [float(sys.argv[1])], #std_co # true C + noise --> noise B
                   #'dropout_rate':[0, 0.2, 0.4, 0.6, 0.8],
                   's_noise':[0.0, 0.1], # true C vs S
                   'sample_nums':[int(sys.argv[2])],
                   'lambda': [0.0]}

date = '122721-5'
c = 0

START_TIME = time.time()
for b_noise in test_parameters['b_noise']:
    for s_noise in test_parameters['s_noise']:
        for n in tqdm.tqdm(test_parameters['sample_nums']):
            sim_dir = '../data/%s/%s_%s_%s' % (date, b_noise, s_noise, n)
            with open('%s/%s_%s_%s.pickle' % (sim_dir, b_noise, s_noise, n), 'rb') as handle:
                data = pickle.load(handle)
            handle.close()
            result_dir = sim_dir + '/results'
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

            for lam in test_parameters['lambda']:
                # 2.1 run RAD w/0 any prior when lambda==0, using lam=flag for marker
                if lam == 0:
                    print('b_noise=%s, s_noise=%s, sample_n=%s on pure RAD'%(b_noise, s_noise, n))
                    # k = RAD.estimate_number(B, max_comp=4, n_splits=2)
                    print('%s cell component...' % K)
                    #pdb.set_trace()
                    inferC, inferF, list_err = RAD.estimate_clones(B, K, verbose=False)
                    #inferC, inferF = utils.alignResult(inferC, inferF, C, K)
                    inferC1, inferF1 = utils.alignResult(inferC[:,:k], inferF[:k,:], C[:,:k], k)
                    inferC2, inferF2 = utils.alignResult(inferC[:,k:], inferF[k:,:], C[:,k:], y)
                    inferC = np.hstack((inferC1, inferC2))
                    inferF = np.vstack((inferF1, inferF2))

                    infermu = sRAD._linear_reg_mu(inferC[:, :k], S)

                    C_diff = np.sqrt(((C-inferC)**2).mean())
                    C_diff_norm = C_diff / np.sqrt((C**2).mean())

                    C1_diff = np.sqrt(((C[:,:k]-inferC1)**2).mean())
                    C1_diff_norm = C1_diff / np.sqrt((C[:,:k]**2).mean())

                    C2_diff = np.sqrt(((C[:,k:]-inferC2)**2).mean())
                    C2_diff_norm = C2_diff / np.sqrt((C[:,k:]**2).mean())

                    F_diff = np.sqrt(((F-inferF)**2).mean())
                    F_diff_norm = F_diff / np.sqrt((F**2).mean())

                    F1_diff = np.sqrt(((F[:k,:]-inferF1)**2).mean())
                    F1_diff_norm = F1_diff / np.sqrt((F[:k,:]**2).mean())

                    F2_diff = np.sqrt(((F[k:,:]-inferF2)**2).mean())
                    F2_diff_norm = F2_diff / np.sqrt((F[k:,:]**2).mean())

                    mu_diff = np.abs(mu-infermu)
                    mu_diff_norm = mu_diff/mu

                    results = {'inferC': inferC, 'inferF': inferF, 'infermu':infermu, 'k':k, 
                            'listErr':list_err, 'diffC': C_diff, 'diffF': F_diff, 'diffmu':mu_diff,
                            'diffCnorm': C_diff_norm, 'diffFnorm': F_diff_norm, 'diffmunorm':mu_diff_norm,
                            'diffC1norm': C1_diff_norm, 'diffC2norm':C2_diff_norm, 
                            'diffF1norm': F1_diff_norm, 'diffF2norm': F2_diff_norm}

                    with open('%s/result_%s_%s_%s_%s.pickle' % (result_dir, b_noise, s_noise, n, 'flag'), 'wb') as handle:
                        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.close()
                #break # only run for flag
                c+=1
                print('b_noise=%s, s_noise=%s, sample_n=%s,lam=%s'%(b_noise, s_noise, n, lam))
                inferC, inferF, infermu, list_err = sRAD._rna_coordescent(B, C1_init, C2_init, F_init, S, mu_init, lam)
                #inferC, inferF = utils.alignResult(inferC, inferF, C, K)
                inferC1, inferF1 = utils.alignResult(inferC[:,:k], inferF[:k,:], C[:,:k], k)
                inferC2, inferF2 = utils.alignResult(inferC[:,k:], inferF[k:,:], C[:,k:], y)
                inferC = np.hstack((inferC1, inferC2))
                inferF = np.vstack((inferF1, inferF2))

                C_diff = np.sqrt(((C-inferC)**2).mean())
                C_diff_norm = C_diff / np.sqrt((C**2).mean())

                C1_diff = np.sqrt(((C[:,:k]-inferC1)**2).mean())
                C1_diff_norm = C1_diff / np.sqrt((C[:,:k]**2).mean())

                C2_diff = np.sqrt(((C[:,k:]-inferC2)**2).mean())
                C2_diff_norm = C2_diff / np.sqrt((C[:,k:]**2).mean())

                F_diff = np.sqrt(((F-inferF)**2).mean())
                F_diff_norm = F_diff / np.sqrt((F**2).mean())

                F1_diff = np.sqrt(((F[:k,:]-inferF1)**2).mean())
                F1_diff_norm = F1_diff / np.sqrt((F[:k,:]**2).mean())

                F2_diff = np.sqrt(((F[k:,:]-inferF2)**2).mean())
                F2_diff_norm = F2_diff / np.sqrt((F[k:,:]**2).mean())

                mu_diff = np.abs(mu-infermu)
                mu_diff_norm = mu_diff/mu
                
                results = {'inferC': inferC, 'inferF': inferF, 'infermu':infermu, 'k':k, 
                            'listErr':list_err, 'diffC': C_diff, 'diffF': F_diff, 'diffmu':mu_diff,
                            'diffCnorm': C_diff_norm, 'diffFnorm': F_diff_norm, 'diffmunorm':mu_diff_norm,
                            'diffC1norm': C1_diff_norm, 'diffC2norm':C2_diff_norm, 
                            'diffF1norm': F1_diff_norm, 'diffF2norm': F2_diff_norm}

                with open('%s/result_%s_%s_%s_%s.pickle' % (result_dir, b_noise, s_noise, n, lam), 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()

END_TIME = time.time()
print('Finished %s deconvolution in %0.2f s' % (c, END_TIME-START_TIME))
