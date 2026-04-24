import numpy as np
import scipy.special

def vrow(x):
    return x.reshape((1, x.size))

def vcol(x):
    return x.reshape((x.size, 1))

def load(fileName):
    dataMatrix = []
    labels = []
    hLabels = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
        }
    with open(fileName) as f:
        for line in f:
            x_j = np.array([float(i) for i in line.split(",")[0:-1]])
            x_j = x_j.reshape((x_j.size, 1))
            dataMatrix.append(x_j)
            
            l_j = line.split(",")[-1].strip()
            labels.append(hLabels[l_j])
    return (np.hstack(dataMatrix), np.array(labels))

def split_db_2to1(D, L, seed = 0):
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

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((X-mu) * (P @ (X-mu))).sum(0)

if __name__ == "__main__":
    # Load and split data for the binary task (keep only classes 1 and 2)
    DIris, LIris = load("iris.csv")
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR),(DVAL, LVAL) = split_db_2to1(D, L)
    
    DTR1 = DTR[:, LTR == 1] # Columns of the training dataset corresponding to class 1
    DTR2 = DTR[:, LTR == 2] # Columns of the training dataset corresponding to class 2
    
    # Evaluate model parameters (MVG)
    mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
    mu2 = DTR2.mean(1).reshape((DTR2.shape[0], 1))
    C1 = computeCovariance(DTR1)
    C2 = computeCovariance(DTR2)
    
    # Compute log-likelihood ratios
    LLRs = logpdf_GAU_ND(DVAL, mu2, C2) - logpdf_GAU_ND(DVAL, mu1, C1)
    LLRs_sol = np.load("Solution/llr_MVG.npy")
    
    # Compute predictions
    t = 0
    
    PVAL = np.zeros(DVAL.shape[1], np.int32)
    PVAL[ LLRs < t ] = 1
    PVAL[ LLRs >= t ] = 2
    
    n_correct_predictions = np.array([PVAL == LVAL]).sum()
    acc = n_correct_predictions / PVAL.shape[0]
    err = 1 - acc
    print("MVG error rate: ", err)
    
    
    # Evaluate model parameters (Tied Gaussian model)
    C1t = C1 * np.identity(C1.shape[0])
    C2t = C2 * np.identity(C2.shape[0])
    
    # Compute log-likelihoods ratios
    LLRs_t = logpdf_GAU_ND(DVAL, mu2, C2t) - logpdf_GAU_ND(DVAL, mu1, C1t)
    
    # Compute predictions
    t = 0
    
    PVAL = np.zeros(DVAL.shape[1], np.int32)
    PVAL[ LLRs_t < t ] = 1
    PVAL[ LLRs_t >= t ] = 2
    
    n_correct_predictions = np.array([PVAL == LVAL]).sum()
    acc = n_correct_predictions / PVAL.shape[0]
    err = 1 - acc
    print("Tied error rate: ", err)
    