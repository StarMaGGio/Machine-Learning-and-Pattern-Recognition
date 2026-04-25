# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from src.utils import loadData, split_db_2to1, computeCovariance, vrow, vcol, computeCorrelationMatrix
from src.visualization import histsPlot, plot_distribution_density
from src.dimensionality_reduction import trainPCAmodel, trainLDAmodel
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND
from src.gaussian_models import compute_llr_for_classification, compute_predictions_with_llr, compute_error_rate

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
    
# ---------------------------------------------
#  Compare MVG vs Tied Gaussian vs Naive Bayes
# ---------------------------------------------
def compare_gaussian_models(DTR, LTR, DVAL, LVAL):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    
    # --- MVG ---
    # Compute ML estimates for model parameters (mu, C)
    C0, mu0 = computeCovariance(DTR0)
    C1, mu1 = computeCovariance(DTR1)
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, C0, C1)
    
    # Compute predictions
    PVAL = compute_predictions_with_llr(LLRs, DVAL.shape[1], 0)
    
    # Compute error rate
    err = compute_error_rate(PVAL, LVAL)
    print("MVG error rate: ", err)
    
    # --- Naive Bayes Gaussian ---
    # Compute ML estimates for model parameters (mu, Ct)
    Ct0 = C0 * np.identity(C0.shape[1])
    Ct1 = C1 * np.identity(C1.shape[1])
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, Ct0, Ct1)
    
    # Compute predictions
    PVAL = compute_predictions_with_llr(LLRs, DVAL.shape[1], 0)
    
    # Compute error rate
    err = compute_error_rate(PVAL, LVAL)
    print("Naive Bayes Gaussian error rate: ", err)
    
    # --- Tied Gaussian ---
    # Compute ML estimates for model parameters (mu, Sw)
    Sw = ((C0*DTR0.shape[1])+(C1*DTR1.shape[1]))/float(DTR.shape[1])
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, Sw, Sw)
    
    # Compute predictions
    PVAL = compute_predictions_with_llr(LLRs, DVAL.shape[1], 0)
    
    # Compute error rate
    err = compute_error_rate(PVAL, LVAL)
    print("Tied Gaussian error rate: ", err)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    D, L = loadData("data/trainData.txt")
    # Plot histograms for the features of the initial dataset
    #histsPlot(D, L, "", 1)
    
    # Split dataset in train and eval
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # ----- LAB 5 -----
    # Try gaussian models with all the features
    compare_gaussian_models(DTR, LTR, DVAL, LVAL)
    
    #  Analize results in light of features characteristics
    # Print Covariance Matrices
    print()
    #print("Covariance Matrix (Class 0): ", computeCovariance(D[:, L==0])[0])
    #print("Covariance Matrix (Class 1): ", computeCovariance(D[:, L==1])[0])
    
    # Compute Correlation Matrices
    Corr0 = computeCorrelationMatrix(computeCovariance(D[:, L==0])[0])
    Corr1 = computeCorrelationMatrix(computeCovariance(D[:, L==1])[0])
    
    # Plot distribution densities of all the features
    #plot_distribution_density(D, L)
    
    # Try gaussian models with only features 1 to 4
    D_f1_4 = D[:4, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_f1_4, L)
    #compare_gaussian_models(DTR, LTR, DVAL, LVAL)
    print()
    
    # Try again with only features 1-2 (similar mean, different variance)
    D_f1_2 = D[:2, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_f1_2, L)
    #compare_gaussian_models(DTR, LTR, DVAL, LVAL)
    print()
    
    # Try again with only features 3-4 (different mean, similar variance)
    D_f3_4 = D[2:4, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_f3_4, L)
    #compare_gaussian_models(DTR, LTR, DVAL, LVAL)
    print()
    
    
    # Try again by reprocessing with PCA
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    m = 4
    # Estimate PCA on initial DTR
    P = trainPCAmodel(DTR, m)
    # Apply PCA on DTR and DVAL
    DTR_pca = np.dot(P.T, DTR)
    DVAL_pca = np.dot(P.T, DVAL)
    histsPlot(DTR_pca, LTR, "", 4)
    compare_gaussian_models(DTR_pca, LTR, DVAL_pca, LVAL)