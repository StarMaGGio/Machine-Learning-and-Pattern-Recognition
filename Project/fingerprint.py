# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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

def histsPlot(D, L):
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
    
    for idxFea in range(6):
        plt.figure()
        plt.hist(D0[idxFea, :], bins=10, density=True, alpha=0.4, label='Fake', color="red")
        plt.hist(D1[idxFea, :], bins=10, density=True, alpha=0.4, label='Genuine', color="green")
        plt.xlabel(hFea[idxFea])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
    plt.show()
    
def scattersPlot(D, L):
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
    
    for idxFea1 in range(6):
        for idxFea2 in range(6):
            if idxFea1 == idxFea2: continue
            plt.figure()
            plt.scatter(D0[idxFea1, :], D0[idxFea2, :], alpha=0.5, label="Fake", color="red")
            plt.scatter(D1[idxFea1, :], D1[idxFea2, :], alpha=0.5, label="Genuine", color="green")
            plt.xlabel(hFea[idxFea1])
            plt.ylabel(hFea[idxFea2])
            plt.legend()
            plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    D, L = loadData("trainData.txt")
    
    histsPlot(D, L)
    
    scattersPlot(D, L)