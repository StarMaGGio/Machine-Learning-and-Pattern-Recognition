import numpy as np
import scipy
import scipy.special as special
import sklearn.datasets

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']    
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def compute_confusion_matrix(predictions, true_labels):
    n = len(np.unique(true_labels))
    conf_matrix = np.zeros((n, n), np.int32)
    
    for p,t in zip(predictions, true_labels):
        conf_matrix[p][t] += 1
        
    return conf_matrix

def compute_bayes_risk(pi, Cfn, Cfp, cm):
    Pfn = cm[0][1]/(cm[0][1]+cm[1][1])
    Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
    
    return pi*Cfn*Pfn + (1-pi)*Cfp*Pfp

def compute_normalized_DCF(pi, Cfn, Cfp, cm):
    DCFu = compute_bayes_risk(pi, Cfn, Cfp, cm)
    
    return DCFu/min(pi*Cfn, (1-pi)*Cfp)

def compute_normalized_minDCF(evalset_llr_binary, evalset_labels_binary, pi, Cfn, Cfp):
    minDCF = 1000
    predicted_labels_binary = np.zeros(evalset_labels_binary.shape[0], dtype=np.int32)
    # for each "threshold" in the llr list
    for t in np.unique(evalset_llr_binary):
        # Compute predicted labels using the score as a threshold
        predicted_labels_binary[evalset_llr_binary > t] = 1
        predicted_labels_binary[evalset_llr_binary <= t] = 0
        
        # Compute confusion matrix
        conf = compute_confusion_matrix(predicted_labels_binary, evalset_labels_binary)
        
        # Compute normalized DCF
        currentDCF = compute_normalized_DCF(pi, Cfn, Cfp, conf)
    
        minDCF = min(minDCF, currentDCF)
    
    return minDCF

##############
# Linear SVM #
##############
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

##############
# Kernel SVM #
##############
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

####################
# Kernel Functions #
####################
def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c)**degree
    return polyKernelFunc

def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)
    return rbfKernelFunc

if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            w, b = train_dual_SVM_linear(DTR, LTR, C, K)    # Train SVM model -> Return model parameters
            SVAL = (vrow(w) @ DVAL + b).ravel()             # Compute scores
            PVAL = (SVAL > 0) * 1                           # Compute predictions
            err = (PVAL != LVAL).sum() / float(LVAL.size)   # Copute predictions error
            print('Error rate: %.1f' % (err*100))
            print('minDCF - pT = 0.5: %.4f' % compute_normalized_minDCF(SVAL, LVAL, 0.5, 1.0, 1.0))
            print('actDCF - pT = 0.5: %.4f' % compute_normalized_DCF(0.5, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL)))
            print()
            
    for kernelFunc in [polyKernel(2, 0), polyKernel(2, 1), rbfKernel(1.0), rbfKernel(10.0)]:
        for eps in [0.0, 1.0]:
            fScore = train_dual_SVM_kernel(DTR, LTR, 1.0, kernelFunc, eps)
            SVAL = fScore(DVAL)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL).sum() / float(LVAL.size)
            print('Error rate: %.1f' % (err*100))
            print('minDCF - pT = 0.5: %.4f' % compute_normalized_minDCF(SVAL, LVAL, 0.5, 1.0, 1.0))
            print('actDCF - pT = 0.5: %.4f' % compute_normalized_DCF(0.5, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL)))
            print()