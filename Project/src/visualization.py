# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from src.utils import computeCovariance, vrow
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND

def histsPlot(D, L, title, nDimensions = 6):
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
        plt.title(title)
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
        
# Plot distibution density on top of the normalized histogram for all the features of the dataset
def plot_distribution_density(D, L):
    # For each class, for each feature, compute ML estimate and plot the distibution density on top of the normalized histogram
    XPlot = np.linspace(-4, 4, 1000)
    for c in range(2):
        D_c = D[:, L==c]
        mu_class_ML = D_c.mean(1).reshape((D_c.shape[0], 1))
        for i in range(6):
            mu_class_fea_ML = mu_class_ML[i]
            D_class_fea = D_c[i, :].reshape(1, -1)
            C_class_fea_ML = computeCovariance(D_class_fea)[0]
            
            plt.figure()
            plt.hist(D_c[i], bins=50, density=True)
            logpdf = logpdf_GAU_ND(vrow(XPlot), mu_class_fea_ML, C_class_fea_ML).sum()
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_class_fea_ML, C_class_fea_ML)))
            plt.title(f"Gaussian Distribution of Feature {i+1} - Class {c}")
            plt.show()