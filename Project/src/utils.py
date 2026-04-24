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

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C, mu