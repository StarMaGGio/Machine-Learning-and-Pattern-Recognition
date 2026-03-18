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

def computePCA(D, m):
    # DATASET MEAN
    mu = D.mean(1).reshape((D.shape[0], 1))
    
    DC = D - mu
    
    # COVARIANCE MATRIX
    C = DC @ DC.T / float(D.shape[1])
    
    # EIGENVALUES and EIGENVECTORS
    s, U = np.linalg.eigh(C)
    
    # Retrieve m=2 leading eigenvectors
    P = U[:, ::-1][:, 0:m]
    
    # Apply projection to the Dataset
    DP = np.dot(P.T, D)
    
    return DP

def scatterPlot(D, L, title):
    D0 = D[:, L == 0] # Columns corresponding to the class 0
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
    plt.figure()
    plt.scatter(D0[0, :], D0[1, :], label="Setosa", color="blue")
    plt.scatter(D1[0, :], D1[1, :], label="Versizolor", color="orange")
    plt.scatter(D2[0, :], D2[1, :], label="Virginica", color="green")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C
    
def computeLDA(D, L):
    D0 = D[:, L == 0] # Columns corresponding to the class 0
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
    # Between class covariance matrix
    mu = D.mean(1).reshape((D.shape[0], 1))
    mu0 = D0.mean(1).reshape((D0.shape[0], 1))
    mu1 = D1.mean(1).reshape((D1.shape[0], 1))
    mu2 = D2.mean(1).reshape((D2.shape[0], 1))
    S_B = ((mu0-mu)@(mu0-mu).T*D0.shape[1]+\
            (mu1-mu)@(mu1-mu).T*D1.shape[1]+\
            (mu2-mu)@(mu2-mu).T*D2.shape[1])/float(D.shape[1])
    
    # Within class covariance matrix
    C0 = computeCovariance(D0)
    C1 = computeCovariance(D1)
    C2 = computeCovariance(D2)
    S_W = ((C0*D0.shape[1])+(C1*D1.shape[1])+(C2*D2.shape[1]))/float(D.shape[1])

    return S_B, S_W

def histPlot(D, title):
    D0 = D[:, L == 0] # Columns corresponding to the class 0
    D1 = D[:, L == 1] # Columns corresponding to the class 1
    D2 = D[:, L == 2] # Columns corresponding to the class 2
    
    # PLOT HIST FOR SEPAL LENGTH FEATURE
    plt.figure()
    plt.hist(D0[0, :], bins=10, density=True, alpha=0.4, label="Setosa", color="blue")
    plt.hist(D1[0, :], bins=10, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[0, :], bins=10, density=True, alpha=0.4, label="Virginica", color="green")
    
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    D, L = load("iris.csv")
    
    # PRINCIPAL COMPONENT ANALYSIS
    DP = computePCA(D, 2)
    
    scatterPlot(DP, L, "Principal Component Analysis")
    
    # LINEAR DISCRIMINANT ANALYSIS
    S_B, S_W = computeLDA(D, L)
    
    print("S_B:")
    print(S_B)
    print()
    print("S_W:")
    print(S_W)
    
    # Solve generalized eigenvalue problem
    s, U = scipy.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:2]
    print("LDA matrix")
    print(W)
    
    # Apply LDA to the dataset
    DW = np.dot(W.T, D)
    
    scatterPlot(DW, L, "Linear Discriminant Analysis")
    
    histPlot(DP, "PCA 1st direction")
    histPlot(DW, "LDA 1st direction")