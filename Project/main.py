# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from src.utils import loadData, split_db_2to1, computeCovariance, vrow, vcol, computeCorrelationMatrix, compute_confusion_matrix, quadratic_expansion, polyKernel, rbfKernel
from src.visualization import histsPlot, plot_distribution_density, plot_Bayes_error
from src.dimensionality_reduction import trainPCAmodel, trainLDAmodel
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND
from src.gaussian_models import compute_llr_for_classification, compute_predictions_with_llr, compute_error_rate, compute_llr_MVG, compute_llr_Tied_Gaussian, compute_llr_Naive_Bayes_Gaussian
from src.bayes_decisions_model import compute_optimal_bayes_decisions, compute_normalized_DCF, compute_normalized_minDCF
from src.logistic_regression import trainLogReg, trainLogRegWeighted
from src.support_vector_machines import train_dual_SVM_linear, train_dual_SVM_kernel

# --------------------------------------------
#  Analyze effects of PCA, LDA on the dataset
# --------------------------------------------
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
    
def compare_gaussian_models_with_different_features(DTR, LTR, DVAL, LVAL):
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
    compare_gaussian_models(DTR_pca, LTR, DVAL_pca, LVAL)# ----- LAB 5 -----
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

# -----------------------
#  Evaluation/Bayes Risk
# -----------------------
def compare_effPriors_and_DCFs_for_different_applications(DTR, LTR, DVAL, LVAL):
    # Define 5 different applications
    applications = [(0.5, 1.0, 1.0), # uniform prior and costs
                    (0.9, 1.0, 1.0), # prior probability of Genuine sample is higher
                    (0.1, 1.0, 1.0), # prior probability of Fake sample is higher
                    (0.5, 1.0, 9.0), # prior is uniform and cost of accepting fake image is larger
                    (0.5, 9.0, 1.0)] # prior is uniform and cost of rejecting legit image is larger
    
    # Compute effective priors for each application
    effective_priors = []
    for pi, Cfn, Cfp in applications:
        effPrior = (pi*Cfn)/((pi*Cfn)+(1-pi)*Cfp)
        effective_priors.append(effPrior)
        print(f"Application (pi={pi}, Cfn={Cfn}, Cfp={Cfp}) -> Effective Prior: {effPrior:.2f}")
        print()
    
    # Compute optimal Bayes decisions for the validation set for MVG models and its variants
    gaussian_models = ["MVG", "Tied Gaussian", "Naive Bayes Gaussian"]
    effective_priors = np.unique(effective_priors)
    for model in gaussian_models:
        print()
        print("Model: ", model)
        
        evalset_llr_binary = []
        if model == "MVG": evalset_llr_binary = compute_llr_MVG(DTR, LTR, DVAL)
        elif model == "Tied Gaussian": evalset_llr_binary = compute_llr_Tied_Gaussian(DTR, LTR, DVAL)
        elif model == "Naive Bayes Gaussian": evalset_llr_binary = compute_llr_Naive_Bayes_Gaussian(DTR, LTR, DVAL)
        
        for effPrior in effective_priors:
            PVAL = compute_optimal_bayes_decisions(effPrior, evalset_llr_binary, LVAL)
            DCF = compute_normalized_DCF(effPrior, 1, 1, compute_confusion_matrix(PVAL, LVAL))
            min_DCF = compute_normalized_minDCF(evalset_llr_binary, LVAL, effPrior, 1, 1)
            loss = DCF - min_DCF
            percent_loss = loss / min_DCF * 100
            print(f"effPrior={effPrior}: norm_DCF={DCF:.3f}, min_DCF={min_DCF:.3f}, percent_loss={percent_loss:.3f}")
            
        plot_Bayes_error(evalset_llr_binary, LVAL, model)

# ---------------------
#  Logistic Regression
# ---------------------
def analyze_logistic_regression_with_different_lambdas(DTR, LTR, DVAL, LVAL, title):
    actDCFs = []
    minDCFs = []
    lambs = np.logspace(-4, 2, 13)
    pi = 0.1
    for lamb in lambs:
        w, b = trainLogReg(DTR, LTR, lamb) # Train model -> obtain model parameters w and b
        sVal = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = np.zeros(LVAL.shape[0], dtype=np.int32) # Predict validation labels
        PVAL[sVal > 0] = 1
        PVAL[sVal < 0] = 0
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.2f' % (err*100))
        
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size # Fraction of class 1 samples
        # Compute LLR-like scores
        sValLLr = sVal - np.log(pEmp / (1 - pEmp))
        # Compute optimal decisions
        PVALllr = np.zeros(LVAL.shape[0], dtype=np.int32) # Predict validation labels
        PVALllr[sValLLr > 0] = 1
        PVALllr[sValLLr < 0] = 0
        conf_matr = compute_confusion_matrix(PVALllr, LVAL)
        minDCF = compute_normalized_minDCF(sValLLr, LVAL, pi, 1.0, 1.0)
        minDCFs.append(minDCF)
        print('minDCF: %.4f' % minDCF)
        actDCF = compute_normalized_DCF(pi, 1.0, 1.0, conf_matr)
        actDCFs.append(actDCF)
        print('actDCF: %.4f' % actDCF)
    
        print()
        
    plt.figure()
    plt.plot(lambs, minDCFs, label="minDCF", color='r')
    plt.plot(lambs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('lambda value')
    plt.title(title)
    plt.show()

def analyze_weighted_logistic_regression_with_different_lambdas(DTR, LTR, DVAL, LVAL, title):
    actDCFs = []
    minDCFs = []
    lambs = np.logspace(-4, 2, 13)
    # Compute empirical prior
    pEmp = (LTR == 1).sum() / LTR.size # Fraction of class 1 samples
    
    for lamb in lambs:
        w, b = trainLogRegWeighted(DTR, LTR, lamb, pEmp) # Train model -> obtain model parameters w and b
        sVal = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = np.zeros(LVAL.shape[0], dtype=np.int32) # Predict validation labels
        PVAL[sVal > 0] = 1
        PVAL[sVal < 0] = 0
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.2f' % (err*100))
    
        # Compute LLR-like scores
        sValLLr = sVal - np.log(pEmp / (1 - pEmp))
        # Compute optimal decisions
        PVALllr = np.zeros(LVAL.shape[0], dtype=np.int32) # Predict validation labels
        PVALllr[sValLLr > 0] = 1
        PVALllr[sValLLr < 0] = 0
        conf_matr = compute_confusion_matrix(PVALllr, LVAL)
        pi = 0.1
        minDCF = compute_normalized_minDCF(sValLLr, LVAL, pi, 1.0, 1.0)
        minDCFs.append(minDCF)
        print('minDCF: %.4f' % minDCF)
        actDCF = compute_normalized_DCF(pi, 1.0, 1.0, conf_matr)
        actDCFs.append(actDCF)
        print('actDCF: %.4f' % actDCF)
    
        print()
        
    plt.figure()
    plt.plot(lambs, minDCFs, label="minDCF", color='r')
    plt.plot(lambs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('lambda value')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    D, L = loadData("data/trainData.txt")
    # Plot histograms for the features of the initial dataset
    #histsPlot(D, L, "", 1)
    
    # Split dataset in train and eval
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    #compare_effPriors_and_DCFs_for_different_applications(DTR, LTR, DVAL, LVAL)
    
    # --- LAB 7 ---
    #analyze_logistic_regression_with_different_lambdas(DTR, LTR, DVAL, LVAL, "Full-Dataset - Non-Weighted")
    
    # Analyze Logistic Regression results with reduced dataset
    # DTR_reduced = DTR[:, ::50]
    # LTR_reduced = LTR[::50]
    #analyze_logistic_regression_with_different_lambdas(DTR_reduced, LTR_reduced, DVAL, LVAL, "1/50 Dataset - Non-Weighted")
    
    # DTR_expanded = quadratic_expansion(DTR)
    # DVAL_expanded = quadratic_expansion(DVAL)
    #analyze_logistic_regression_with_different_lambdas(DTR_expanded, LTR, DVAL_expanded, LVAL, "Expanded Dataset - Non-Weighted")
            
    # --- LAB 8 ---
    DTR_reduced = DTR#[:, ::50]
    LTR_reduced = LTR#[::50]
    DVAL_reduced = DVAL#[:, ::50]
    LVAL_reduced = LVAL#[::50]
    
    # SVM linear
    K = 1.0
    Cs = np.logspace(-5, 0, 11)   
    minDCFs = []
    actDCFs = []
    for C in Cs:
        w, b = train_dual_SVM_linear(DTR_reduced, LTR_reduced, C, K)    # Train SVM model -> Return model parameters
        SVAL = (vrow(w) @ DVAL_reduced + b).ravel()             # Compute scores
        PVAL = (SVAL > 0) * 1                           # Compute predictions
        err = (PVAL != LVAL_reduced).sum() / float(LVAL_reduced.size)   # Copute predictions error
        print('Error rate: %.1f' % (err*100))
        minDCFs.append(compute_normalized_minDCF(SVAL, LVAL_reduced, 0.1, 1.0, 1.0))
        actDCFs.append(compute_normalized_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL_reduced)))
    plt.figure()
    plt.plot(Cs, minDCFs, label="minDCF", color='r')
    plt.plot(Cs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title("SVM Linear")
    plt.legend()
    plt.show()
    print()
    
    # SVM linear centered data
    minDCFs.clear()
    actDCFs.clear()
    mu = DTR_reduced.mean(1).reshape((DTR_reduced.shape[0], 1))
    DTR_reduced_centered = DTR_reduced - mu
    DVAL_reduced_centered = DVAL_reduced - mu
    for C in Cs:
        w, b = train_dual_SVM_linear(DTR_reduced_centered, LTR_reduced, C, K)    # Train SVM model -> Return model parameters
        SVAL = (vrow(w) @ DVAL_reduced_centered + b).ravel()             # Compute scores
        PVAL = (SVAL > 0) * 1                           # Compute predictions
        err = (PVAL != LVAL_reduced).sum() / float(LVAL_reduced.size)   # Copute predictions error
        print('Error rate: %.1f' % (err*100))
        minDCFs.append(compute_normalized_minDCF(SVAL, LVAL_reduced, 0.1, 1.0, 1.0))
        actDCFs.append(compute_normalized_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL_reduced)))
    plt.figure()
    plt.plot(Cs, minDCFs, label="minDCF", color='r')
    plt.plot(Cs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title("SVM Linear Centered Data")
    plt.legend()
    plt.show()
    print()
    
    # SVM Polynomial Kernel
    minDCFs.clear()
    actDCFs.clear()
    kernelFunc = polyKernel(2, 1)
    eps = 0.0
    for C in Cs:
        fScore = train_dual_SVM_kernel(DTR_reduced, LTR_reduced, C, kernelFunc, eps)
        SVAL = fScore(DVAL_reduced)
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL_reduced).sum() / float(LVAL_reduced.size)
        print('Error rate: %.1f' % (err*100))
        minDCFs.append(compute_normalized_minDCF(SVAL, LVAL_reduced, 0.1, 1.0, 1.0))
        actDCFs.append(compute_normalized_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL_reduced)))
    plt.figure()
    plt.plot(Cs, minDCFs, label="minDCF", color='r')
    plt.plot(Cs, actDCFs, label="actDCF", color='b')
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title("SVM Polynomial Kernel")
    plt.legend()
    plt.show()
    print()
    
    # SVM RBF Kernel
    eps = 1.0
    Cs = np.logspace(-3, 2, 11)
    plt.figure()
    hLinestyles = {
        0: '-',
        1: '--',
        2: '-.',
        3: ':'
    }
    i = 0
    for kernelFunc in [rbfKernel(math.exp(-4)), rbfKernel(math.exp(-3)), rbfKernel(math.exp(-2)), rbfKernel(math.exp(-1))]:
        minDCFs.clear()
        actDCFs.clear()
        for C in Cs:
            fScore = train_dual_SVM_kernel(DTR_reduced, LTR_reduced, C, kernelFunc, eps)
            SVAL = fScore(DVAL_reduced)
            PVAL = (SVAL > 0) * 1
            err = (PVAL != LVAL_reduced).sum() / float(LVAL_reduced.size)
            print('Error rate: %.1f' % (err*100))
            minDCFs.append(compute_normalized_minDCF(SVAL, LVAL_reduced, 0.1, 1.0, 1.0))
            actDCFs.append(compute_normalized_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(PVAL, LVAL_reduced)))
        plt.plot(Cs, minDCFs, label=f"minDCF - gamma: e^{i-4}", color='r', linestyle=hLinestyles[i])
        plt.plot(Cs, actDCFs, label=f"actDCF - gamma: e^{i-4}", color='b', linestyle=hLinestyles[i])
        i = i + 1
    plt.xscale('log', base=10)
    plt.ylabel('DCF value')
    plt.xlabel('C value')
    plt.title("SVM RBF Kernel")
    plt.legend()
    plt.show()
    