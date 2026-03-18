import numpy as np
import matplotlib.pyplot as plt
import scipy

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

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C

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

def trainPCAmodel(D, m):
    # DATASET MEAN
    mu = D.mean(1).reshape((D.shape[0], 1))
    
    DC = D - mu
    
    # COVARIANCE MATRIX
    C = DC @ DC.T / float(D.shape[1])
    
    # EIGENVALUES and EIGENVECTORS
    s, U = np.linalg.eigh(C)
    
    # Retrieve m leading eigenvectors
    P = U[:, ::-1][:, 0:m]
    
    return P # return the PCA matrix

def trainLDAmodel(D, L):
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
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
    W = U[:, ::-1][:, 0:2]
    
    return W # return the LDA matrix

def histPlot(D, L, title):
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
    plt.figure()
    plt.hist(D1[0, :], bins=5, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[0, :], bins=5, density=True, alpha=0.4, label="Virginica", color="green")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def scatterPlot(D, L, title):
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
    plt.figure()
    plt.scatter(D1[0, :], D1[1, :], label="Versizolor", color="orange")
    plt.scatter(D2[0, :], D2[1, :], label="Virginica", color="green")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DIris, LIris = load("iris.csv")
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
# --- APPLYING ONLY THE LDA ---

    # Train Linear Discriminant Analysis model (LDA)
    W = trainLDAmodel(DTR, LTR)

    # Project Training samples
    DTRW = np.dot(W.T, DTR)
    #histPlot(DTRW, LTR, "Model training set (DTR, LTR)")
    
    # Project Validation samples
    DVALW = np.dot(W.T, DVAL)
    #histPlot(DVALW, LVAL, "Validation set (DVAL, LVAL)")
    
    #scatterPlot(DVAL, LVAL, "original")
    #scatterPlot(DVALW, LVAL, "projected")
    
    # Compute threshold for the classification
    threshold = (DTRW[0, LTR==1].mean() + DTRW[0, LTR==2].mean()) / 2.0
    
    # Classify DVAL with LDA
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVALW[0] >= threshold] = 2 # Predict class 2 for elements greater than the threshold
    PVAL[DVALW[0] < threshold] = 1 # Predict class 1 for elements lower than the threshold
    
    nDisagreeLDA = (LVAL != PVAL).sum()
    print("LDA-only mispredictions: ")
    print(nDisagreeLDA)
    print()
    
# --- APPLYING ONLY THE PCA ---
    
    # Train Principal Component Analysis model (PCA)
    P = -trainPCAmodel(DTR, m=2)
    
    # Project Training samples
    DTRP = np.dot(P.T, DTR)
    #histPlot(DTRP, LTR, "PCA-only Model training set (DTR, LTR)")
    
    # Project Validation samples
    DVALP = np.dot(P.T, DVAL)
    #histPlot(DVALP, LVAL, "PCA-only Validation set (DVAL, LVAL)")
    
    # Compute threshold for the classification
    threshold = (DTRP[0, LTR==1].mean() + DTRP[0, LTR==2].mean()) / 2.0
    
    # Classify DVALP with PCA
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVALP[0] >= threshold] = 2 # Predict class 2 for elements greater than the threshold
    PVAL[DVALP[0] < threshold] = 1 # Predict class 1 for elements lower than the threshold
    
    nDisagreePCA = (LVAL != PVAL).sum()
    print("PCA-only mispredictions: ")
    print(nDisagreePCA)
    print()
    
# --- APPLYING PCA AND LDA ---

    # Estimate PCA on initial DTR
    P = trainPCAmodel(DTR, m=2)
    
    # Apply PCA matrix on DTR
    DTR_pca = np.dot(P.T, DTR)
    
    # Apply PCA matrix on initial DVAL
    DVAL_pca = np.dot(P.T, DVAL)
    
    # Estimate LDA on DTR_pca
    W = trainLDAmodel(DTR_pca, LTR)
    
    # Apply LDA matrix on DTR_pca
    DTR_lda = np.dot(W.T, DTR_pca)
    
    # Apply LDA matrix on DVAL_pca
    DVAL_lda = np.dot(W.T, DVAL_pca)
    histPlot(DVAL_lda, LVAL, "result")
    
    # Estimate threshold from DTR_lda
    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0
    
    # Classify DVAL_lda with PCA and LDA
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2 # Predict class 2 for elements greater than the threshold
    PVAL[DVAL_lda[0] < threshold] = 1 # Predict class 1 for elements lower than the threshold
    
    nDisagreePCA_LDA = (LVAL != PVAL).sum()
    print("PCA-LDA mispredictions: ")
    print(nDisagreePCA_LDA)
    
    
    
    
    
    
    