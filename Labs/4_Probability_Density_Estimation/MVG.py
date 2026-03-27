# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def vrow(x):
    return x.reshape((1, x.size))

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND_singleSample(x, mu, C):
    C_inv = np.linalg.inv(C)
    M = x.shape[0]
    return -0.5*M*np.log(2*np.pi) - 0.5*np.linalg.slogdet(C)[1] - 0.5*((x-mu).T @ C_inv @ (x-mu)).ravel()

def logpdf_GAU_ND_slow(X, mu, C):
    num_samples = X.shape[1]
    log_densities = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(num_samples)]
    return np.array(log_densities).ravel()

def logpdf_GAU_ND_fast(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((X-mu) * (P @ (X-mu))).sum(0)

def loglikelihood(X, mu, C):
    return logpdf_GAU_ND_slow(X, mu, C).sum()

if __name__ == "__main__":
    
    # Plot density for mu = 1 and C = 2
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    mu = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    multivariate_gaussian_density = np.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu, C))
    #plt.plot(XPlot.ravel(), multivariate_gaussian_density)
    
    # Check slow pdf function results with Solution/llGAU.npy
    pdfSol = np.load('Solution/llGAU.npy')
    pdfRes = logpdf_GAU_ND_fast(vrow(XPlot), mu, C)
    print("Max difference single dimension case: ", np.abs(pdfSol - pdfRes).max())
    
    # Check multi-dimensional case with Solution/XND.npy
    XND = np.load('Solution/XND.npy')
    muND = np.load('Solution/muND.npy')
    CND = np.load('Solution/CND.npy')
    
    pdfSol = np.load('Solution/llND.npy')
    pdfRes = logpdf_GAU_ND_slow(XND, muND, CND)
    print("Max difference multi dimension case: ", np.abs(pdfSol - pdfRes).max())
    print()
    
    # Maximum Likelihood Estimate for XND dataset
    mu_ML, C_ML = computeCovariance(XND)
    print(mu_ML)
    print()
    print(C_ML)
    ll = loglikelihood(XND, mu_ML, C_ML)
    print(ll)
    print()
    
    # Maximum Likelihood Estimates and Density Plot for X1D dataset
    X1D = np.load('Solution/X1D.npy')
    mu_ML, C_ML = computeCovariance(X1D)
    print(mu_ML)
    print()
    print(C_ML)
    
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_slow(vrow(XPlot), mu_ML, C_ML)))
    plt.show()
    
    print(loglikelihood(X1D, mu_ML, C_ML))
    
    # Other values
    print(loglikelihood(X1D, np.array([[1.0]]), np.array([[2.0]])))
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_slow(vrow(XPlot), np.array([[0.0]]), np.array([[1.0]]))))
    plt.show()