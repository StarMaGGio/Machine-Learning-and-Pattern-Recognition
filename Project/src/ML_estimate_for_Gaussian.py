# -*- coding: utf-8 -*-
import numpy as np

def logpdf_GAU_ND_singleSample(x, mu, C):
    C_inv = np.linalg.inv(C)
    M = x.shape[0]
    return -0.5*M*np.log(2*np.pi) - 0.5*np.linalg.slogdet(C)[1] - 0.5*((x-mu).T @ C_inv @ (x-mu)).ravel()

def logpdf_GAU_ND(X, mu, C):
    num_samples = X.shape[1]
    log_densities = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(num_samples)]
    return np.array(log_densities).ravel()