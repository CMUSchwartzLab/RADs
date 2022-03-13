import numpy as np
from numpy.lib.shape_base import split
from numpy.random.mtrand import dirichlet
import pandas as pd
import simulate
import pickle
import sRAD
import tqdm
import time
import os
import sys
import pdb

def get_C(cell, cellType):
    '''
    retrieve single cell data based on assigned cell type
    '''
    filter_c = pd.DataFrame([], columns=cell.columns)
    for c in cellType:
        temp = cell[cell['cellType']==c]
        filter_c = filter_c.append(temp)
    return filter_c

def data_split(C, cell_type):
    '''
    based on each cell type
    split cell to 5:1 as train and test
    '''
    # 1. pick up the desired cell type
    C = C.loc[C['cellType'].isin(cell_type)].reset_index(drop=True)
    train_C = pd.DataFrame([], columns=C.columns)
    test_C = pd.DataFrame([], columns=C.columns)

    # 2. split each celltype to train:test = 5:1
    for ct in set(C['cellType']):
        index = list(C.loc[C['cellType']==ct].index)
        np.random.shuffle(index)
        split_n = int(len(index)//6) * 5
        train_index = index[:split_n]
        test_index = index[split_n:]
        train_C = train_C.append(C.iloc[train_index])
        test_C = test_C.append(C.iloc[test_index])
    #pdb.set_trace()
    return train_C.reset_index(drop=True), test_C.reset_index(drop=True)

def main(date):
    real_bulk = pd.read_csv('../data/cleaned_bulk.csv')
    cell = pd.read_csv('../data/cell.csv')
    de_gene = pd.read_csv('../data/de_genes.csv')
    cell_types=['Mast', 'Endothelial', 'Macrophage']
    lams = [0, 0.01, 0.1, 1, 10, 100, 1000]
    bulk_gene = set(real_bulk.columns)
    cell_gene = set(cell.columns)
    common_gene = list(bulk_gene.intersection(cell_gene))

    #filter B and C with common genes
    B = real_bulk[common_gene]
    C = cell[common_gene + ['cellType']]
    train_C, test_C = data_split(C, cell_types)
    k = len(cell_types)

    cell_gene = train_C.columns.to_list()[0:-1]
    selected_gene_idx = []
    for gene in de_gene.columns.to_list():
        try:
            index = cell_gene.index(gene)
            selected_gene_idx.append(index)
        except:
            continue
    # selected_gene_idx = np.random.choice(np.arange(C.shape[1]-1), 
    #                                         size=5000, replace=False)
    selected_gene = B.columns[selected_gene_idx].to_list()

    S = simulate.gen_S_from_GEP(train_C, 
                       cell_types, selected_gene_idx,
                       s_noise=0)

    # get final C and B based on selected cellTypes and genes
    C = get_C(train_C, cellType=cell_types)
    C_mean = train_C.groupby('cellType').mean()

    B = B.iloc[:, selected_gene_idx]#.to_numpy().T
    C = C.iloc[:, selected_gene_idx]#.to_numpy().T
    C_mean = C_mean.iloc[:, selected_gene_idx]#.to_numpy().T

    assert (C_mean.columns == B.columns).all()
    assert (S.columns == B.columns).all()
    assert (C_mean.columns == S.columns).all()
    assert (train_C.columns == test_C.columns).all()

    #reindex C_mean
    C_mean = C_mean.reindex(S.index)

    data = {'S': S}
    data['trainC'] = train_C.iloc[:, selected_gene_idx]
    data['trainC']['cellType'] = train_C['cellType']
    data['testC'] = test_C.iloc[:, selected_gene_idx]
    data['testC']['cellType'] = test_C['cellType']

    #run scRAD
    B = B.to_numpy().T
    S = S.to_numpy().T
    C_mean = C_mean.to_numpy().T

    C_init = S
    F_init = sRAD._quad_prog_BCmu2F(B, C_init) # solve F as initialization
    mu_init = sRAD._linear_reg_mu(C_mean, S)

    for lam in lams:
        inferC, inferF, infermu, list_err = sRAD._rna_coordescent(B, C_init, F_init, S, mu_init, lam)
    
        data['inferC'] = inferC
        data['inferF'] = inferF
        data['infermu'] = infermu

        result_dir = '../results/real/%s' % date
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        with open('%s/result_%s.pickle' % (result_dir, lam), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

if __name__ == '__main__':
    START_TIME = time.time()
    date = sys.argv[1]
    main(date)
    END_TIME = time.time()
    print('Finished real data deconvolution in %0.2f s' % (END_TIME-START_TIME))



