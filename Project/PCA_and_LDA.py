# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy

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

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C

def trainPCAmodel(D, m):
    # covariance matrix
    C = computeCovariance(D)
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

def histsPlot(D, L, nDimensions = 6):
    hFea = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        }
    
    D0 = D[:, L==0] # Fake class
    D1 = D[:, L==1] # Genuine class
    
    for idxFea in range(nDimensions):
        plt.figure()
        plt.hist(D0[idxFea, :], bins=10, density=True, alpha=0.4, label='Fake', color="red")
        plt.hist(D1[idxFea, :], bins=10, density=True, alpha=0.4, label='Genuine', color="green")
        plt.xlabel(hFea[idxFea])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    D, L = loadData("trainData.txt")
    # Plot histograms for the features of the initial dataset
    #histsPlot(D, L)
    
    # ANALYZE EFFECTS OF PCA ON THE FEATURES
    P = trainPCAmodel(D, 6)
    DP = np.dot(P.T, D)
    # Plot histograms for the 6 PCA directions
    #histsPlot(DP, L)
    
    # ANALYZE EFFECTS OF LDA
    W = trainLDAmodel(D, L, 1)
    DW = np.dot(W.T, D)
    # Plot histogram
    #histsPlot(DW, L, 1)
    
    # APPLY LDA FOR CLASSIFICATION
    # Divide the dataset in training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Compute and apply LDA matrix to training and validation sets
    W = trainLDAmodel(DTR, LTR, 1)
    DTRW = np.dot(W.T, DTR)
    DVALW = np.dot(W.T, DVAL)
    #histsPlot(DVALW, LVAL, 1)
    # Compute threshold for the classification
    threshold = (DTRW[0, LTR==0].mean() + DTRW[0, LTR==1].mean()) / 2.0
    print(f"{threshold:.5f}")
    
    # Classify DVAL with LDA
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVALW[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
    PVAL[DVALW[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold
    # Compute LDA prediction error rate
    nDisagreeLDA = (LVAL != PVAL).sum()
    error_rate = nDisagreeLDA/len(PVAL)
    print(f"LDA-only error rate: {error_rate:.5f}")