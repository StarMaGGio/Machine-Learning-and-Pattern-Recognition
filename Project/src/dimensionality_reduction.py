import numpy as np
import scipy
from src.utils import computeCovariance
# -*- coding: utf-8 -*-

def trainPCAmodel(D, m):
    # covariance matrix
    C = computeCovariance(D)[0]
    # eigenvalues and eigenvectors
    s, U = np.linalg.eigh(C)
    # m leading eigenvectors
    P = U[:, ::-1][:, 0:m]
    return P # return the PCA matrix

def trainLDAmodel(D, L, m):
    D1 = D[:, L == 0] # Columns corresponding to the class 0
    D2 = D[:, L == 1] # Columns corresponding to the class 1
    # S_B
    mu = D.mean(1).reshape((D.shape[0], 1))
    mu1 = D1.mean(1).reshape((D1.shape[0], 1))
    mu2 = D2.mean(1).reshape((D2.shape[0], 1))
    S_B = ((mu1-mu)@(mu1-mu).T*D1.shape[1]+\
           (mu2-mu)@(mu2-mu).T*D2.shape[1])/float(D.shape[1])
    # S_W
    C1 = computeCovariance(D1)
    C2 = computeCovariance(D2)
    S_W = ((C1*D1.shape[1])+(C2*D2.shape[1]))/float(D.shape[1])
    # Solve generalized eigenvalue problem
    s, U = scipy.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]
    return W # return the LDA matrix