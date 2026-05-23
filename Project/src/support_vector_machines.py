import numpy as np
import scipy
from src.utils import vcol, vrow

def train_dual_SVM_linear(DTR, LTR, C, K):
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K]) # Append a row of elements all = K
    ZTR = LTR * 2.0 - 1.0 # Convert labels to -1/+1
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR) # H_i,j = z_i*z_j*x_i^T*x_j where x = DTR_EXT and z = ZTR
    
    # Dual objective and gradient
    def fOpt(alpha):
        Halpha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Halpha).ravel() - alpha.sum() # L^D(alpha) = -J^D(alpha)
        grad = Halpha.ravel() - np.ones(alpha.size)
        return loss, grad
    
    # Search the minimazer of the loss function
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=np.nan, pgtol=1e-5)
    
    # Directly compute primal objective to check duality gap
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel() # Scores
        return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()
    
    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b from w_hat
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w^Tx + b * K
    
    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    print('SVM - K %f - C %f - primal loss %e - dual loss %e - duality gap %e' % (K, C, primalLoss, dualLoss[0], primalLoss - dualLoss[0]))
  
    return w, b

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):
    ZTR = LTR * 2.0 - 1.0 # Convert labels to -1/+1
    K = kernelFunc(DTR, DTR) + eps # Replace DTR dot product with Kernel Function
    H = vcol(ZTR) * vrow(ZTR) * K
    
    # Dual objective and gradient
    def fOpt(alpha):
        Halpha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Halpha).ravel() - alpha.sum() # L^D(alpha) = -J^D(alpha)
        grad = Halpha.ravel() - np.ones(alpha.size)
        return loss, grad
    
    # Search the minimazer of the loss function
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=np.nan, pgtol=1e-5)
    
    def primalLoss(alpha):
        Halpha = H @ vcol(alpha)
        return 0.5 * (vrow(alpha) @ Halpha) + C * np.maximum(0, 1 - Halpha).sum()

    primalLoss, dualLoss = primalLoss(alphaStar), -fOpt(alphaStar)[0][0]
    print('SVM (Kernel) - C %f - primal loss %e - dual loss %e - duality gap %e' % (C, primalLoss, dualLoss, primalLoss - dualLoss))
    
    # Compute scores for samples in DTE
    def fScore(DTE):
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)
    
    return fScore # Directly return the function to compute scores of test samples matrix