from src.ML_estimate_for_Gaussian import logpdf_GAU_ND
import numpy as np
import scipy
from src.utils import computeCovariance, vcol

# Computes the log-density of a GMM for a set of samples contained in matrix X
# gmm = [(w1, mu1, C1), (x2, mu2, C2), ...]
def logpdf_GMM(X, gmm):
    M = len(gmm)
    S = np.zeros((M, X.shape[1])) # S shape (M, N). S[g, ;] is the log-density of the g-th Gaussian component for all samples in X
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    # We use the log-sum-exp trick to compute the log-density of the GMM
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens # Output is a vector (N,) where each element is the log-density of the GMM for the corresponding sample in X
    
# Algorithm to estimate the parameters of a GMM that maximize the likelihood for a training set X
def GMM_EM_estimation(X, gmm_init, psi=None, eps_ll=1e-6):
    M = len(gmm_init)
    gmm = gmm_init.copy()
    S = np.zeros((M, X.shape[1]))

    currentLoglikelihood = None

    while True:
        # Step 1 (E-Step): compute posterior prob. for each component of GMM for each sample
        for g in range(M):
            w, mu, C = gmm[g]
            S[g, :] = logpdf_GAU_ND(X, mu, C) + np.log(w) # Joint density of g-th component and X
            
        logSMarginal = scipy.special.logsumexp(S, axis=0) # Marginal densities
        newLoglikelihood = logSMarginal.mean()
        
        # Stopping criterion: we can check the log-likelihood of the data given the model parameters and stop when it does not increase significantly anymore
        if currentLoglikelihood is not None and np.abs(newLoglikelihood - currentLoglikelihood) < eps_ll:
            break
            
        currentLoglikelihood = newLoglikelihood
        
        logSPost = S - logSMarginal
        SPost = np.exp(logSPost)

        # Step 2 (M-Step): Update model parameters
        for g in range(M):
            Z_g = SPost[g, :].sum() # Normalization factor for g-th component
            F_g = vcol((X * SPost[g, :]).sum(axis=1)) # Weighted sum of samples for g-th component
            S_g = (X * SPost[g, :]) @ X.T # Weighted sum of squared samples for g-th component

            new_mu_g = F_g / Z_g
            new_C_g = S_g / Z_g - new_mu_g @ new_mu_g.T
            new_w_g = Z_g / X.shape[1]
            
            # Regularize covariance matrix if psi is not None
            if psi is not None:
                U, s, Vh = np.linalg.svd(new_C_g)
                s[s < psi] = psi
                new_C_g = U @ (vcol(s) * U.T)

            gmm[g] = (new_w_g, new_mu_g, new_C_g)

    return gmm

# Function to split a Gaussian component into two components with mean shift alpha along the direction of maximum variance (used in LBG algorithm)
def LBG_split(gmm_in, alpha):
    gmm_out = []
    for w, mu, C in gmm_in:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmm_out.append((w/2, mu - d, C))
        gmm_out.append((w/2, mu + d, C))
    return gmm_out

def train_GMM_LBG_EM(X, numComponents, alpha=0.1, psi=0.01):
    C, mu = computeCovariance(X)
    
    # Regularize covariance matrix if psi is not None
    if psi is not None:
        U, s, Vh = np.linalg.svd(C)
        s[s < psi] = psi
        C = U @ (vcol(s) * U.T)
        
    gmm = [(1.0, mu, C)]
    while len(gmm) < numComponents:
        gmm = LBG_split(gmm, alpha) # Split each component into two components with mean shift alpha along the direction of maximum variance
        gmm = GMM_EM_estimation(X, gmm, psi=psi) # Estimate the parameters of the GMM with EM algorithm after splitting the components
    return gmm