#!/usr/bin/env python
# coding=utf-8


import cvxopt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pdb
from tqdm import tqdm
import time

''''
min_{C, F, \mu}  ||B-CF||_Fr^2 + \lamda ||C-\mu S||_Fr^2
    s.t. C_{il} >= 0
         F_{lj} >= 0
         \sum^k{l=1} F_{lj} = 1
'''
def _cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    '''
    solve the standard QP program with cvxopt:
        min_x 1/2 x^T P x + q^T x
            s.t. Gx <= h
                 Ax = b
                 
    *check the standard form for cvxopt:
        https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    '''
    P = 0.5 * (P + P.T) # this is to make sure that P is symetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)] # P and q are required
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options["show_progress"] =False # not show output to the screen
    sol = cvxopt.solvers.qp(*args)
    if "optimal" not in sol["status"]:
        return None
    return np.array(sol["x"]).reshape((P.shape[1],))



def _quad_prog_BCmu2F(B, C):
    """
    phase 1: solve F when fixing B. 
         min_F ||B-CF||_Fr^2
            s.t. F_{lj}>=0, l=1,..., k; j=1,...,n
                 \sum^k_{l=1} F_{lj}=1, j=1,...,n
    <==> min_f 1/2 f^TC^TCf -b'^TCf
      ==>   Q = C^TC, q^T = -(b'^TC)
            f is column of F, b' is column of b
    """
    # 1. define the dimension of the matrix
    # B=m*n, C=m*k, F=k*n
    num_gene = B.shape[0]       #row of B, m: number of gene            
    num_sample = B.shape[1]     #col of B, n: number of bulk sample
    num_cell = C.shape[1]       #col of C, k: number of cell type/population
    
    # 2. Gx<=h: inequality constraints of f 
    # 2.1 lower bound for f, f >=0 <==> -f <= 0
    Gl = np.diag([-1.0] * num_cell)
    hl = np.zeros(num_cell).reshape((num_cell,))
    
    # 2.2 upper bound for f, f <= 1
    Gu = np.diag([1.0] * num_cell)
    hu = np.ones(num_cell).reshape((num_cell,))
    
    # 2.3 combine 2.1 and 2.2 to the form of Gx <= h
    G = np.vstack((Gl, Gu))
    h = np.hstack((hl, hu))
    
    # 3. Ax=b, equality constraints of f, \sum f = 1
    A = np.ones(num_cell).reshape((1,num_cell))
    b = np.ones(1).reshape((1,))
    #pdb.set_trace()   
    # 4. solve the QP of F column by column
    # 1/2 f^TC^TCf -b'^TCf ==> Q=C^TC, q^T=-b'^TC
    F = []
    for j in range(num_sample):
        P = np.dot(C.T, C)
        q = -np.dot(B[:,j].T, C).T
        
        fj = _cvxopt_solve_qp(P, q, G, h, A, b)
        
        if fj is None: 
            return None
        
        F.append(fj)
    F = np.vstack(F).T
    
    return F        
    
def _quad_prog_BFmu2C1(B, F, S, C2, mu, lam, max_val=2**19):
    """
    phase 2.1: solve C1 when fix F and mu
        min_C1 ||B-C1F||_{Fr}^2 + \lambda||C1-\mu S||_{Fr}^2
            s.t. C1_{il}>=0, i=1,...,m; l=1,...,k
                 C1_{il}<=max_val -- expression can be any value smaller than max value
                 C1_{ij}=0, i=1,...,m; j=k+1,...,K 
    <==> min_c1 1/2 c1^T(FF^T+\lambda I)c1 -(Fb+\lambda\mu s)^Tc1
      ==> Q = (FF^T+\lambda I), q^T = -(Fb+\lambda\mu s)^T
            c1 is transpose of row of C1, b is transpose of row of B
            s is transpose of row of S
        K: total number of cell types
        k: cell types in single-cell metastasis.
        k <= K
    """
    # 1. define the dimension of the matrix
    # B=m*n, C1=m*K, F=K*n
    num_gene = B.shape[0]       #row of B, m: number of gene            
    num_sample = B.shape[1]     #col of B, n: number of bulk sample
    y = C2.shape[1]             #number of unknown cell type/population
    K = F.shape[0]              #number of total cell type/population
    k = K-y                     #numbr of known cell type found in single-cell metastasis
    F1 = F[:k,:]              #we only optimize on the first k cell type
    F2 = F[k:,:]              #fraction of unknown cell types
    #pdb.set_trace()
    num_cell = F1.shape[0]      #row of F, K: number of cell type/population
    B = B - np.dot(C2, F2)      #substract the effect from C2

    # 2. Gx<=h, inequaility of c
    # 2.1 lower bound for c, c >=0 <==> -c <= 0
    Gl = np.diag([-1.0] * num_cell)
    hl = np.zeros(num_cell).reshape((num_cell,))
    
    # 2.2 uppder bound for c, c<=max_val
    Gu = np.diag([1.0] * num_cell)
    hu = max_val*np.ones(num_cell).reshape((num_cell,))
    
    # 2.3 combine 2.1 and 2.2 to the form of Gx <= h
    G = np.vstack((Gl, Gu))
    h = np.hstack((hl, hu))
    #pdb.set_trace()
    # 3. we have partial equality constraints for c1
    # but we just optimize on the first k cell types


    # 4. solve the QP of C row by row
    # 1/2 c^T(FF^T+\lambda I)c -(Fb+\lambda\mu s)^Tc
    #  ==> P=(FF^T+\lambda I), q^T=-(Fb+\lambda\mu s)^T
    C1 = []
    for i in range(num_gene):
        P = np.dot(F1,F1.T) + lam*np.identity(num_cell)
        q = -np.dot(F1, B[i,:].T) - lam*mu*S[i,:].T
        from tqdm import tqdm

        c1i = _cvxopt_solve_qp(P, q, G, h)

        if c1i is None:
            return None
        
        C1.append(c1i)
    C1 = np.vstack(C1)

    return C1

def _quad_prog_BFmu2C2(B, F, S, C1, mu, lam, max_val=2**19):
    """
    phase 2.2: solve C2 when fix F
        min_C2 ||B-C2F||_{Fr}^2
            s.t. C2_{il}>=0, i=1,...,m; l=1,...,k
                 C2_{il}<=max_val -- expression can be any value smaller than max value
                 C2_{ij}=0, i=1,...,m; j=0,...,k
    <==> min_c2 1/2 c2^T(FF^T)c2
      ==> Q = FF^T, q^T = -(Fb)^T
            c2 is transpose of row of C2, b is transpose of row of B
        K: total number of cell types
        k: cell types in single-cell metastasis.
        k <= K
    """
    # 1. define the dimension of the matrix
    # B=m*n, C2=m*y, F=K*n
    num_gene = B.shape[0]       #row of B, m: number of gene            
    num_sample = B.shape[1]     #col of B, n: number of bulk sample
    k = C1.shape[1]             #number of known cell type/population from mestasis
    K = F.shape[0]              #number of total cell type/population
    y = K-k                     #numbr of known unknown cell type 
    F1 = F[:k,:]                #we only optimize on the first k cell type
    F2 = F[k:,:]                #fraction of unknown cell types
    num_cell = F2.shape[0]      #row of F2, y: number of unknown cell type/population
    B = B - np.dot(C1, F1)      #substract the effect from C1
    if y:
        # 2. Gx<=h, inequaility of c
        # 2.1 lower bound for c, c >=0 <==> -c <= 0
        Gl = np.diag([-1.0] * num_cell)
        hl = np.zeros(num_cell).reshape((num_cell,))
        
        # 2.2 uppder bound for c, c<=max_val
        Gu = np.diag([1.0] * num_cell)
        hu = max_val*np.ones(num_cell).reshape((num_cell,))
        
        # 2.3 combine 2.1 and 2.2 to the form of Gx <= h
        G = np.vstack((Gl, Gu))
        h = np.hstack((hl, hu))
        #pdb.set_trace()
        # 3. we have partial equality constraints for c1
        # but we just optimize on the first k cell types


        # 4. solve the QP of C row by row
        # 1/2 c^T(FF^T+\lambda I)c -(Fb+\lambda\mu s)^Tc
        #  ==> P=(FF^T+\lambda I), q^T=-(Fb+\lambda\mu s)^T
        C2 = []
        for i in range(num_gene):
            P = np.dot(F2,F2.T)
            q = -np.dot(F2, B[i,:].T)
            from tqdm import tqdm

            c2i = _cvxopt_solve_qp(P, q, G, h)

            if c2i is None:
                return None
            
            C2.append(c2i)
        C2 = np.vstack(C2)
        return C2
    else:
        return None


def _linear_reg_mu(C1, S):
    """
    phase 3: solve mu when C and S are known
         min_mu ||C-\mu S||_{Fr}^2
    <==> min RSS(\mu) = \sum_i^m(C[i:]-\mu S[i:])^2=\sum^m_{i=1}\sum^k_{l=1}(C[i,l]-\mu S[i,l])^2
    Since \mu is scalar, we can just obtain the derivative of RSS(\mu) regarding \mu and set it to 0 
    """
    # 1. obtain the derivative regarding \mu
    # RSSD = 2\sum^m_{i=1}\sum^k_{l=1}(C[i,l]*S[i,l]-\mu S[i,l])
    # let \sum^m_{i=1}\sum^k_{l=1}(C[i,l]*S[i,l]-\mu S[i,l]} = 0 
    # ==> mu = (\sum^m_{i=1}\sum^k_{l=1}C[i,l]*S[i,l]) / (\sum^m_{i=1}\sum^k_{l=1}S[i,l])
    mu = np.multiply(C1, S).sum()/(S**2).sum()
    return mu

def _rna_coordescent(B, C1, C2, F, S, mu, lam, max_iter=1000, tol=1e-8):
    C = np.hstack((C1, C2))
    error_at_init = ((B-np.dot(C,F))**2).mean()
    list_err = [error_at_init]
    error = float('inf')
    previous_error = error_at_init

    for i in range(max_iter):
        # 1. Fix C and mu, solve F
        F_tmp = _quad_prog_BCmu2F(B, C)
        if F_tmp is not None:
            F = F_tmp
        error = ((B-np.dot(C,F))**2).mean()
        #list_err.append(error)

        # 2. Fix F and mu, solve C
        C1_tmp = _quad_prog_BFmu2C1(B, F, S, C2, mu, lam)
        C2_tmp = _quad_prog_BFmu2C2(B, F, S, C1, mu, lam)
        if C1_tmp is not None:
            C1 = C1_tmp
        if C2_tmp is not None:
            C2 = C2_tmp
        # if (C1_tmp is not None) and (C2_tmp is not None): 
        C = np.hstack((C1, C2))
        #pdb.set_trace()
        error = ((B-np.dot(C,F))**2).mean()
        list_err.append(error)

        # 3. Fix C solve mu
        mu = _linear_reg_mu(C1, S)
        # error = ((B-np.dot(C,F))**2).mean()
        # list_err.append(error)
        
        # 4. finish one round optimization and check the errpr 
        #print("Current error: ", abs(previous_error - error))
        if i >= 500 and abs(previous_error - error) < tol:
            break
        # update the error
        previous_error = error

    print("Iteration: %s, Last Error: %s " %(i+1 , error))
    print('--------------------------------')
    
    return C, F, mu, list_err