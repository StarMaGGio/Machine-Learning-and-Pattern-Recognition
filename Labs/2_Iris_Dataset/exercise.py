# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

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
        
def histsPlot(D0, D1, D2):
    # PLOT HIST FOR SEPAL LENGTH FEATURE
    plt.figure()
    plt.hist(D0[0, :], bins=10, density=True, alpha=0.4, label="Setosa", color="blue")
    plt.hist(D1[0, :], bins=10, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[0, :], bins=10, density=True, alpha=0.4, label="Virginica", color="green")
    
    plt.xlabel("Sepal length")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # PLOT HIST FOR SEPAL WIDTH FEATURE
    plt.figure()
    plt.hist(D0[1, :], bins=10, density=True, alpha=0.4, label="Setosa", color="blue")
    plt.hist(D1[1, :], bins=10, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[1, :], bins=10, density=True, alpha=0.4, label="Virginica", color="green")
    
    plt.xlabel("Sepal width")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # PLOT HIST FOR PETAL LENGTH
    plt.figure()
    plt.hist(D0[2, :], bins=10, density=True, alpha=0.4, label="Setosa", color="blue")
    plt.hist(D1[2, :], bins=10, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[2, :], bins=10, density=True, alpha=0.4, label="Virginica", color="green")
    
    plt.xlabel("Petal length")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # PLOT HIST FOR PETAL WIDTH
    plt.figure()
    plt.hist(D0[3, :], bins=10, density=True, alpha=0.4, label="Setosa", color="blue")
    plt.hist(D1[3, :], bins=10, density=True, alpha=0.4, label="Versicolor", color="orange")
    plt.hist(D2[3, :], bins=10, density=True, alpha=0.4, label="Virginica", color="green")
    
    plt.xlabel("Petal width")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def scattersPlot(D0, D1, D2):
    # SCATTER PLOTS
    numFeature = {
        0: "Sepal length",
        1: "Sepal width",
        2: "Petal length",
        3: "Petal width"
        }
    
    for idxFea1 in range(4):
        for idxFea2 in range(4):
            if idxFea1 == idxFea2: continue
            plt.figure()
            plt.scatter(D0[idxFea1, :], D0[idxFea2, :], label="Setosa", color="blue")
            plt.scatter(D1[idxFea1, :], D1[idxFea2, :], label="Versicolor", color="orange")
            plt.scatter(D2[idxFea1, :], D2[idxFea2, :], label="Virginica", color="green")
            
            plt.xlabel(numFeature[idxFea1])
            plt.ylabel(numFeature[idxFea2])
            plt.legend()
            plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    # LOAD OF THE DATASET
    dataMatrix, labels = load('iris.csv')
    
    # EXTRACTION OF THE PARTS CORRESPONDING TO THE DIFFERENT CLASSES
    M0 = (labels == 0)
    D0 = dataMatrix[:, M0] # Columns corresponding to the class 0
    
    M1 = (labels == 1)
    D1 = dataMatrix[:, M1] # Columns corresponding to the class 1
    
    M2 = (labels == 2)
    D2 = dataMatrix[:, M2] # Columns corresponding to the class 2
    
    histsPlot(D0, D1, D2)
        
    # DATASET MEAN
    mu = dataMatrix.mean(1).reshape((dataMatrix.shape[0], 1))
    
    print("Mean: ")
    print(mu)
    print()
    
    centeredDataMatrix = dataMatrix - mu
    
    # COVARIANCE MATRIX
    C = centeredDataMatrix @ centeredDataMatrix.T / float(dataMatrix.shape[1])
    print("Covariance matrix:")
    print(C)
    print()
    
    variance = dataMatrix.var(1)
    standard_deviation = dataMatrix.std(1)
    print("Variance: ", variance)
    print("Std. dev.: ", standard_deviation)
    print()
    
    numClass = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
        }

    for cls in range(3):
        print('Class', numClass[cls])
        DCls = dataMatrix[:, labels==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean: ')
        print(mu)
        C = (DCls - mu) @ (DCls - mu).T / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance', var)
        print('Std. dev.:', std)
        print()
    

    
    
    