# RADs

## Introduction
RADs stands for Robust and Accurate Deconvolution with Single-cell data. It is an improved method based on our previous publication and take advantage of the single-cell RNA-seq information to infer the cell type profiles in bulk tumor sample(s).

The high level idea of to combine bulk and single-cell RNA-seq data is as shown as below:

![img](fig/pa.png)

In a mathematic way, we would like: Given a non-negative bulk RNA expression matrix B \in R_+^{m x n}, where each row i is a gene, each column j is a tumor sample, our goal is to infer an expression profile matrix C \in R_+^{m x k}, where each column l is a cell community, and a fraction matrix F \in R_+^{k x n}, such that:

  B ~= C F. 

  In the meanwhile, we use single-cell data from metastases as reference and allow unknown cell type(s) that only exist in primary tumor. The overall problem is shown as:

  ![img](fig/math.png)

  C_1 is the known cell types in metastatic tumor, C_2 is the unknown cell type only in primary tumor, (C_1|C_2) means horizonal stack of these two gene expression in matrix manner

Compared to other method, RADs has at least two advantages:
- it can work on small number of tumor samples, which is usually difficult or impossible for other methods
- it can infer additional information about the primary tumor, while other methods that use reference can only infer information (e.g. cell types) in the reference used

## Requirement
The problem was stated as a Quadratic Programming and solved using [CVXOPT](https://cvxopt.org/) in Python, you may install the package first by using 
```conda install -c conda-forge cvxopt``` or ```pip install cvxopt```

## Tutorial
While the real data is not able to share at this moment, we provided a simulated data based on the real data as well as  jupyter notebook `tutorial.ipynb` in the `code` directory for users to better understand the tool