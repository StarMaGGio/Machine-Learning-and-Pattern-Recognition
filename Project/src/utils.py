import numpy as np

def vrow(x):
    return x.reshape((1, x.size))

def vcol(x):
    return x.reshape((x.size, 1))

def loadData(fileName):
    dataMatrix = []
    labels = []
    
    with open(fileName) as f:
        for line in f:
            features = np.array([float(i) for i in line.split(",")[0:-1]])
            columnFeatures = features.reshape(features.size, 1)
            dataMatrix.append(columnFeatures)
            
            label = int(line.split(",")[-1])
            labels.append(label)
            
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

def computeMean(D):
    return D.mean(1).reshape((D.shape[0], 1))

def computeCovariance(D):
    mu = computeMean(D)
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C

def computeCorrelationMatrix(C):
    C / ( vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5 ))
    
def compute_confusion_matrix(predictions, true_labels):
    n = len(np.unique(true_labels))
    conf_matrix = np.zeros((n, n), np.int32)
    
    for p,t in zip(predictions, true_labels):
        conf_matrix[p][t] += 1
        
    return conf_matrix

def quadratic_expansion(X):
    xxT_all = X[:, None, :] * X[None, :, :]
    vec_xxT_all = xxT_all.reshape(-1, X.shape[1])
    phi_X = np.vstack([vec_xxT_all, X])
    return phi_X

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