# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src.utils import loadData, split_db_2to1, computeCovariance, vrow
from src.visualization import histsPlot
from src.dimensionality_reduction import trainPCAmodel, trainLDAmodel
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND

# Function to preprocess the dataset with Principal Component and Linear Discrimination Analysis
def PCA_LDA_effects_and_classification_analysis(D, L):
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
    
    # ------ PCA + LDA ------
    m = 1
    # Estimate PCA on initial DTR
    P = -trainPCAmodel(DTR, m)
    # Apply PCA on DTR and DVAL
    DTR_pca = np.dot(P.T, DTR)
    DVAL_pca = np.dot(P.T, DVAL)
    histsPlot(DTR_pca, LTR, "DTR_pca", m)
    #histsPlot(DVAL_pca, LVAL, "DVAL_pca", m)
    # Estimate LDA on DTR_pca
    W = trainLDAmodel(DTR_pca, LTR, 1)
    # Apply LDA on DTR_pca and DVAL_pca
    DTR_lda = np.dot(W.T, DTR_pca)
    histsPlot(DTR_lda, LTR, "DTR_lda", 1)
    DVAL_lda = np.dot(W.T, DVAL_pca)
    #histsPlot(DVAL_lda, LVAL, 1)
    # Estimate threshold from DTR_lda
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
    print(f"{threshold:.5f}")
    # Classify DVAL_lda with estimated threshold
    PVAL2 = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL2[DVAL_lda[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
    PVAL2[DVAL_lda[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold
    # Compute PCA+LDA prediction error rate
    nDisagreePCA_LDA = (LVAL != PVAL2).sum()
    error_rate = nDisagreePCA_LDA/len(PVAL2)
    print(f"PCA-LDA error rate: {error_rate:.5f}")

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    D, L = loadData("data/trainData.txt")
    # Plot histograms for the features of the initial dataset
    #histsPlot(D, L, "", 1)
    
    # For each class, for each feature, compute ML estimate and plot the distibution density on top of the normalized histogram
    XPlot = np.linspace(-4, 4, 1000)
    for c in range(2):
        D_c = D[:, L==c]
        mu_class_ML = D_c.mean(1).reshape((D_c.shape[0], 1))
        for i in range(6):
            mu_class_fea_ML = mu_class_ML[i]
            D_class_fea = D_c[i, :].reshape(1, -1)
            C_class_fea_ML = computeCovariance(D_class_fea)
            
            plt.figure()
            plt.hist(D_c[i], bins=50, density=True)
            plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu_class_fea_ML, C_class_fea_ML)))
            plt.title(f"Gaussian Distribution of Feature {i+1} - Class {c}")
            plt.show()
    