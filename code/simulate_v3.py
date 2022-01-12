import pandas as pd
import numpy as np

import umap
from sklearn import cluster
from scipy.optimize import curve_fit

def _gen_noisy_replicates(origin, n_replicates):

    ''' Generate noisy replicates of the original expressions
      noisy exp ~ Gaussian(mean_exp, (std_exp/std_division)**2)
    Args:
    
    - origin: (n samples x m genes) np array
    - n_replicates: int, the number of replicates to generate
    Returns:
    - noisy_replicates: (n_replicates cells x m genes) np array
    '''
    # temporarily set zeros to NaN to exclude dropouts from mean/std calculations
    tmp = origin.replace(0, np.nan)
    mean = tmp.mean(axis=0).fillna(0)
    std = tmp.std(axis=0).fillna(0)
    noisy_replicates = np.random.normal(mean, std, (n_replicates, tmp.shape[1]))
    noisy_replicates[noisy_replicates < 0] = 0
    return noisy_replicates

def _dropout (cells, p_dropout):
    
    ''' Set a target percent of expression values to 0
    Args:
    - cells: (n cells x m genes) np array
    - p_dropout: 0-1 float, the target dropout rate
    Returns:
    - cells_dropout: (n cells x m genes) np array with dropouts
    '''
    n_dropped = int(p_dropout * cells.size)
    flatten_exp = cells.flatten()

    # lower exp vals are more likely dropped 
    dropped_weight = 1 / (0.01+flatten_exp)
    dropped_weight /= dropped_weight.sum() # make probabilities sum to 1
    dropped_idx = np.random.choice(np.arange(flatten_exp.shape[0]),
                                  size=n_dropped, 
                                  p=dropped_weight, replace=False)
    flatten_exp[dropped_idx] = 0
    return flatten_exp.reshape(cells.shape)

def _get_GEP(real_cells):
    
    ''' Generate the gene expression profile from real sc data
    
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    Returns:
    - GEP: (k cell types x (m genes + 1 celltype name)) pandas DataFrame
    - unique_celltypes: list of strs, the name of unique cell types
    '''
    
    unique_celltypes = real_cells.iloc[:, -1].unique()
    GEP = pd.DataFrame()
    for celltype in unique_celltypes:
        type_mask = (real_cells['cellType']==celltype)
        type_members = real_cells[type_mask]
        type_GEP = type_members.mean(axis=0).to_frame().T
        type_GEP['cellType'] = celltype
        GEP = pd.concat([type_GEP, GEP])
    GEP.iloc[:, :-1] = GEP.iloc[:, :-1].div(GEP.iloc[:, :-1].sum(axis=1), axis=0)*1000000
    GEP = GEP.set_index(['cellType'])
    return GEP

def _michaelis_mentens(exp, K_M):
    ''' the Michaelis-Mentens equation 
    for inferring dropout probabilities'''
    p_dropout = 1 - (exp / (K_M + exp))
    return p_dropout

def _estimate_KM(real_cells, GEP, cell_types):
    ''' Estimate the K_M constant for each cell type separately
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - GEP: (k cell types x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the name of desired cell types
    Returns:
    - fitted_KM: list of floats, the estimated K_M constant
    '''

    fitted_KM = []
    for celltype in cell_types:
        members = real_cells[real_cells['cellType']==celltype].iloc[:, :-1]
        n_dropped_per_gene = (members == 0).astype(int).sum().values
        n_cells = real_cells.shape[0]
        x = GEP[GEP.index==celltype].to_numpy().astype(np.float64)[0]
        y = n_dropped_per_gene / n_cells
        parameters, covariance = curve_fit(_michaelis_mentens, x, y)
        fitted_KM.append(parameters[0])
    return fitted_KM

def _correct_GEP(GEP, cell_types, fitted_KM):
    
    ''' Correct the GEP using dropout rate inferred from fitted KM
    
    Args:
    - GEP: (k cell types x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the name of desired cell types
    - fitted_KM: list of floats, the estimated K_M constant corresponding 
                  to each celltype
                  
    Returns:
    - corrected_GEP: (k cell types x (m genes + 1 celltype name)) pandas DataFrame
    '''
    corrected_GEP = GEP.copy()
    for (celltype, i) in zip(cell_types, list(range(len(cell_types)))):
        K_M = fitted_KM[i]
        p_dropout = _michaelis_mentens(GEP[GEP.index==celltype], K_M)
        p_dropout[p_dropout < 0] = 0
        corrected_GEP[corrected_GEP.index==celltype] = GEP[GEP.index==celltype] / (1 - p_dropout)
    corrected_GEP = corrected_GEP.fillna(0)
    return corrected_GEP

def gen_sim_sc(real_cells, cell_types, selected_gene_idx,
               n_replicates=1000, p_dropout=0.5):

    '''Generate simulated single cell data with real data
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the names of celltypes to be simulated
    - selected_gene_idx: (n_genes x 1) np array, the idx of selected genes
    - n_replicates: int, the number of replicates to generate
    - p_dropout: 0-1 float, the target dropout rate
    Returns:
    - sim_sc: ((n_replicates*n_celltypes) cells x n_genes genes ) 
              np array
    - selected_gene_idx: (n_genes x 1) np array, the idx of selected genes
    '''
    
    sim_sc = []
    for cell_type in cell_types:
        type_real_cells = real_cells[real_cells['cellType'] == cell_type].iloc[:, selected_gene_idx]
        # generate noisy replicates of the original real data
        noisy_replicates = _gen_noisy_replicates(type_real_cells,
                                                 n_replicates)
        sim_sc.append(noisy_replicates)
    sim_sc = np.vstack(sim_sc)
    # perform random dropout weighted by expression value
    sim_sc = _dropout(sim_sc, p_dropout)
    return sim_sc

def gen_S(sim_sc, k):

    ''' Generate S term from single cell data by spectral clustering in umap space
    Args:
    - sim_sc: (n cells x m genes) np array
    - k: int, the number of desired components / cell communities
    Returns:
    - S: (k components x m genes) np array
    '''

    reducer = umap.UMAP()
    umap_sc = reducer.fit_transform(sim_sc)
    spectral = cluster.SpectralClustering(n_clusters=k)
    labels = spectral.fit_predict(umap_sc)
    S = []
    for component in np.unique(labels):
        component_mask = (labels==component)
        component_members = sim_sc[component_mask, :]
        S.append(np.mean(component_members, axis=0).reshape(1, -1))
    S = np.vstack(S)
    return S
    
def gen_S_from_GEP(real_cells, cell_types, selected_gene_idx, s_noise):
    
    ''' Generate noisy S directly from GEP
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the name of desired cell types
    - selected_gene_idx: (n_genes x 1) np array, the idx of selected genes
    - s_noise: 0-1 float, the coefficient of the real std
    
    Returns:
    - S: (k components x m genes) np array
    '''
    
    GEP = _get_GEP(real_cells).loc[cell_types].iloc[:, selected_gene_idx]
    log_GEP = np.log2(GEP+0.01)
    
    S = log_GEP.copy() # create a deep copy for the tissue GEP
    
    if log_GEP.shape[0] != 1:
        std = log_GEP.std(axis=0)
    else:
        GEP_for_std = _get_GEP(real_cells).iloc[:, selected_gene_idx]
        log_GEP_for_std = np.log2(GEP_for_std+0.01)
        std = log_GEP_for_std.std(axis=0)
    
    # replace each gene j's expression value by x~N(v, std_j)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            exp = np.random.normal(S.iloc[i, j], std[j] * s_noise)
            S.iloc[i, j] = exp
    S = 2 ** S
    return S

def gen_F(n_comp=3, n_samp=400):
    
    """ Generate fraction matrix F
    """

    # https://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
    F = np.random.uniform(size=(n_comp, n_samp))
    F = -np.log(F)
    F_sum = np.sum(F, axis=0)
    F = F/F_sum

    return F

def gen_F_from_sc(real_cells, cell_types, n_sample=400, f_noise=0.01, y=1, p=0.1):
    
    """ Generate fraction matrix F
    
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: array-like of strs, the name of desired cell types
    - n_sample: int, number of samples to generate
    - f_noise: 0-1 float, the coefficient of the real group-wise std for adding noise to each sample
    
    Returns:
    F: (n cell types x n_samples) fraction matrix
    """
    # the fraction vector for metastasis samples are estimated from the single cell data
    metastasis_base_F = real_cells.iloc[:, -1].value_counts(normalize=True)[cell_types].to_numpy().reshape(-1, 1)
    
    # the fraction vector for primary samples are pertubed fractions from metasis
    group_noise = 0.5
    std = metastasis_base_F.std()
    primary_base_F = metastasis_base_F.copy()
    for i in range(metastasis_base_F.shape[0]):
        primary_base_F[i] = np.random.normal(primary_base_F[i], std*group_noise)
    F = np.repeat(primary_base_F, repeats=n_sample, axis=1)
    F[F < 0] = 0
    
    # add noise between samples
    log_F = np.log2(F+0.0000001)
    std = log_F.std(axis=0)
    for i in range(log_F.shape[0]):
        for j in range(log_F.shape[1]):
            exp = np.random.normal(log_F[i, j], std[j]*f_noise)
            log_F[i, j] = exp
    F = 2 ** log_F
    F[F < 0] = 0
    
    if y > 0:
        # set the last y cell types as unknown (replace with values f_i ~ Uniform(0, p])
        F[-y:, :] = np.random.uniform(low=0, high=p, size=(y, F.shape[1]))
            
    # normalize the column/sample sum of the fraction matrix to 1
    F_sum = np.sum(F, axis=0).reshape(1, -1)
    F = F/F_sum
    
    return F, metastasis_base_F



def gen_bulk(real_cells, S, cell_types, selected_gene_idx,
         std_co=0.2, correct_C=True, n_replicates=400):

    '''Generate bulk RNA samples from GEP
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the name of desired cell types
    - selected_gene_idx: (n_genes x 1) np array, the idx of selected genes
    - std_co: float, the coefficient of real data's standard 
              deviation, a higher value introduces more noise
    - correct_C: bool, if apply dropout correction on C
    - n_replicates: int, the number of replicates to generate
    Returns:
    - B: (n_replicates x (n cell types x n_genes)) bulk RNA seq samples
    - C: (n_replicates x (n cell types x n_genes)) cell expression profile
    - F: (n cell types x n_replicates) fraction matrix
    '''

    GEP = _get_GEP(real_cells)
    if correct_C:
        fitted_KM = _estimate_KM(real_cells, GEP, cell_types)
        GEP = _correct_GEP(GEP, cell_types, fitted_KM)

    # modification 1: assume that primary and mestatsis samples have large
    # inter-group variations in their Fs, but the internal variation is low
    n_components = len(cell_types)
    F, baseF = gen_F_from_sc(real_cells, cell_types, n_sample=n_replicates, y=0)

    GEP = GEP.loc[cell_types].iloc[:, selected_gene_idx]
    log_GEP = np.log2(GEP+0.01)
    B = []
    C = []
    mu = []
    for i_sample in range(n_replicates):
        c_i = log_GEP.copy() # create a deep copy for the tissue GEP
        if c_i.shape[0] != 1:
            std = log_GEP.std(axis=0)
        else:
            GEP_for_std = _get_GEP(real_cells).iloc[:, selected_gene_idx]
            log_GEP_for_std = np.log2(GEP_for_std+0.01)
            std = log_GEP_for_std.std(axis=0)
        # replace each gene j's expression value by x~N(v, std_j)
        for i in range(c_i.shape[0]):
            for j in range(c_i.shape[1]):
                exp = np.random.normal(c_i.iloc[i, j], std[j] * std_co)
                c_i.iloc[i, j] = exp
        c_i = 2 ** c_i
        c_i[c_i<0] = 0
        mu.append(np.sum(c_i, axis=0) / np.sum(S, axis=0))

        f_i = F[:, i_sample].reshape(-1, 1) # n_comp x n_sample=1 composition

        # convolute bulk sample by B = F * C
        B.append(pd.DataFrame(np.dot(f_i.T, c_i), columns=GEP.columns)) # n_sample=1 x n_gene bulk data
        C.append(c_i)

        B = pd.concat(B)
        C = np.array(C)
        mu = np.array(mu)
        return mu, B, C, F, baseF

def gen_bulk_with_unknown(real_cells, S, cell_types, selected_gene_idx,
                         std_co=0.2, correct_C=True, n_replicates=400, 
                          y=1, p=0.1):

    '''Generate bulk RNA samples from GEP
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the name of desired cell types
    - selected_gene_idx: (n_genes x 1) np array, the idx of selected genes
    - std_co: float, the coefficient of real data's standard 
              deviation, a higher value introduces more noise
    - correct_C: bool, if apply dropout correction on C
    - n_replicates: int, the number of replicates to generate
    Returns:
    - B: (n_replicates x (n cell types x n_genes)) bulk RNA seq samples
    - C: (n_replicates x (n cell types x n_genes)) cell expression profile
    - F: (n cell types x n_replicates) fraction matrix
    '''

    GEP = _get_GEP(real_cells)
    if correct_C:
        fitted_KM = _estimate_KM(real_cells, GEP, cell_types)
        GEP = _correct_GEP(GEP, cell_types, fitted_KM)

    # generate F (F1, F2)
    n_components = len(cell_types)
    F, baseF = gen_F_from_sc(real_cells, cell_types, n_sample=n_replicates, y=y, p=p)
    F1 = F[:-y, :]
    F2 = F[-y:, :]
    
    # generate C1, B1 from sc data of known cell types and F1
    known_celltypes = cell_types[:-y]
    GEP1 = GEP.loc[known_celltypes].iloc[:, selected_gene_idx]
    log_GEP1 = np.log2(GEP1+0.01)
    B1 = []
    C1 = []
    mu = []
    for i_sample in range(n_replicates):
        c_i = log_GEP1.copy() # create a deep copy for the tissue GEP
        if c_i.shape[0] != 1:
            std = log_GEP1.std(axis=0)
        else:
            GEP_for_std = _get_GEP(real_cells).iloc[:, selected_gene_idx]
            log_GEP_for_std = np.log2(GEP_for_std+0.01)
            std = log_GEP_for_std.std(axis=0)
        # replace each gene j's expression value by x~N(v, std_j)
        for i in range(c_i.shape[0]):
            for j in range(c_i.shape[1]):
                exp = np.random.normal(c_i.iloc[i, j], std[j] * std_co)
                c_i.iloc[i, j] = exp
        c_i = 2 ** c_i
        c_i[c_i<0] = 0
        mu.append(np.sum(c_i, axis=0) / np.sum(S.iloc[:-y, :], axis=0))
        f_i = F1[:, i_sample].reshape(-1, 1) # n_comp x n_sample=1 composition
        # convolute bulk sample by B = F * C
        B1.append(pd.DataFrame(np.dot(f_i.T, c_i), columns=GEP1.columns)) # n_sample=1 x n_gene bulk data
        C1.append(c_i)
    B1 = pd.concat(B1)
    C1 = np.array(C1)
    mu = np.array(mu)
    
    # generate C2, B2 from sc data of unknown cell types and F2 (random sample f_i~Uniform(0, p])
    unknown_celltypes = cell_types[-y:]
    GEP2 = GEP.loc[unknown_celltypes].iloc[:, selected_gene_idx]
    log_GEP2 = np.log2(GEP2+0.01)
    B2 = []
    C2 = []
    for i_sample in range(n_replicates):
        c_i = log_GEP2.copy() # create a deep copy for the tissue GEP
        if c_i.shape[0] != 1:
            std = log_GEP2.std(axis=0)
        else:
            GEP_for_std = _get_GEP(real_cells).iloc[:, selected_gene_idx]
            log_GEP_for_std = np.log2(GEP_for_std+0.01)
            std = log_GEP_for_std.std(axis=0)
        # replace each gene j's expression value by x~N(v, std_j)
        for i in range(c_i.shape[0]):
            for j in range(c_i.shape[1]):
                exp = np.random.normal(c_i.iloc[i, j], std[j] * std_co)
                c_i.iloc[i, j] = exp
        c_i = 2 ** c_i
        c_i[c_i<0] = 0
        f_i = F2[:, i_sample].reshape(-1, 1) # n_comp x n_sample=1 composition
        # convolute bulk sample by B = F * C
        B2.append(pd.DataFrame(np.dot(f_i.T, c_i), columns=GEP2.columns)) # n_sample=1 x n_gene bulk data
        C2.append(c_i)
    B2 = pd.concat(B2)
    C2 = np.array(C2)
    
    B = B1 + B2
    return mu, B, B1, B2, C1, C2, F1, F2, baseF
    
def sim_data(real_cells,
             cell_types=['Mast', 'Endothelial', 'Macrophage'], 
             y=1,
             p=0.1,
             s_noise=0.2, 
             n_bulk=400, n_genes=5000, 
             std_co=0.2, 
             correct_C=True):
  
    ''' Simulate data from real scRNA-seq data
    Args:
    - real_cells: (n cells x (m genes + 1 celltype name)) pandas DataFrame
    - cell_types: list of strs, the names of celltypes to be simulated
    - n_sc: int, the number of single cells to generate for each cell type
    - n_bulk: int, the number of bulk RNAseq samples to generate
    - n_genes: int, the number of genes to simulate for
    - std_co: float, the coefficient of real data's standard 
              deviation, a higher value introduces more noise
    - correct_C: bool, if apply dropout correction on C
    - p_dropout: 0-1 float, the target dropout rate
    Returns:
    - S: (k components x m genes) np array
    - B: (n_replicates x (n cell types x n_genes)) bulk RNA seq samples
    - gtC: (n_replicates x (n cell types x n_genes)) cell expression profile
    - gtF: (n cell types x n_replicates) fraction matrix
    '''

    k = len(cell_types)

    selected_gene_idx = np.random.choice(np.arange(real_cells.shape[1]-1), 
                                         size=n_genes,
                                         replace=False)
    
    S = gen_S_from_GEP(real_cells, 
                       cell_types, selected_gene_idx,
                       s_noise=s_noise)
    
    if y > 0:
        mu, B, B1, B2, gtC1, gtC2, gtF1, gtF2, baseF = gen_bulk_with_unknown(real_cells, S,
                                                               cell_types, selected_gene_idx,
                                                               std_co=std_co,
                                                               correct_C=correct_C,
                                                               n_replicates=n_bulk,
                                                               y=y, p=p)
        S = S.iloc[:-y, :]
        return mu, S, B, B1, B2, gtC1, gtC2, gtF1, gtF2, baseF
    
    else:
        mu, B, gtC, gtF, baseF = gen_bulk(real_cells, S,
                                           cell_types, selected_gene_idx,
                                           std_co=std_co,
                                           correct_C=correct_C,
                                           n_replicates=n_bulk)
        return mu, S, B, gtC, gtF, baseF