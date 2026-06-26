# pyrefly: ignore [missing-import]
import numpy as np
import math
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from src.utils import loadData, split_db_2to1
from src.evaluation import compute_acc_err
from src.visualization import histsPlot, plot_distribution_density, plot_Bayes_error
from src.dimensionality_reduction import PrincipalComponentAnalysis, LinearDiscriminantAnalysis
from src.gaussian_models import MultivariateGaussianClassifier, NaiveBayesGaussianClassifier, TiedGaussianClassifier

from src.multivariate_gaussian_log_pdf import logpdf_GAU_ND
from src.bayes_decisions_model import compute_optimal_bayes_decisions, compute_normalized_DCF, compute_normalized_minDCF
from src.logistic_regression import trainLogReg, trainLogRegWeighted
from src.support_vector_machines import train_dual_SVM_linear, train_dual_SVM_kernel
from src.gaussian_mixture_models import logpdf_GMM, train_GMM_LBG_EM

# --------------------------
#  Dimensionality Reduction
# --------------------------
def PCA_LDA_effects_and_classification_analysis(D, L):

    inner_menu_option = int(input('\n Dimensionality Reduction Menu:\n\
                                    1. Analyze effects of PCA on features\n\
                                    2. Analyze effects of LDA on features\n\
                                    3. Apply LDA for classification\n\
                                    4. Apply PCA + LDA for classification\n\
                                    0. Back\n'))

    match inner_menu_option:
        case 1:
            # ANALYZE EFFECTS OF PCA ON THE FEATURES
            m = int(input('Number of PCA directions: '))
            PCA = PrincipalComponentAnalysis()
            PCA.train(D, m)
            DP = PCA.apply(D)
            # Plot histograms for the m PCA directions
            histsPlot(DP, L, 'PCA effect on features', m)
        case 2:
            # ANALYZE EFFECTS OF LDA
            m = int(input('Number of LDA directions: '))
            LDA = LinearDiscriminantAnalysis()
            LDA.train(D, L, m)
            DW = LDA.apply(D)
            # Plot histogram
            histsPlot(DW, L, "LDA effect on features", m)
        case 3:
            # APPLY LDA FOR CLASSIFICATION
            # Divide the dataset in training and validation sets
            (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

            # Compute and apply LDA matrix to training and validation sets
            m = int(input('Number of LDA directions: '))
            LDA = LinearDiscriminantAnalysis()
            LDA.train(DTR, LTR, m)
            DTRW = LDA.apply(DTR)
            DVALW = LDA.apply(DVAL)
            
            # Compute threshold (in this case the mean of the means of the two classes) for the classification
            threshold = (DTRW[0, LTR==0].mean() + DTRW[0, LTR==1].mean()) / 2.0
            print(f"Threshold: {threshold:.5f}")
            
            # Classify projected DVAL with the threshold computed from projected DTR
            PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
            PVAL[DVALW[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
            PVAL[DVALW[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold
            
            # Compute LDA prediction error rate
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"LDA-only error rate: {err:.5f}")
        case 4:
            # ------ PCA + LDA ------
            m = int(input('Number of PCA directions: '))
            # Estimate PCA on initial DTR
            PCA = PrincipalComponentAnalysis()
            PCA.train(DTR, m)
            # Apply PCA on DTR and DVAL
            DTR_pca = PCA.apply(DTR)
            DVAL_pca = PCA.apply(DVAL)
            histsPlot(DTR_pca, LTR, "DTR_pca", m)

            n = int(input("Number of LDA directions: "))
            # Estimate LDA on DTR_pca
            LDA = LinearDiscriminantAnalysis()
            LDA.train(DTR_pca, LTR, n)
            # Apply LDA on DTR_pca and DVAL_pca
            DTR_lda = LDA.apply(DTR_pca)
            histsPlot(DTR_lda, LTR, "PCA + LDA effect on training features", n)
            DVAL_lda = LDA.apply(DVAL_pca)

            
            # Estimate threshold from DTR preprocessed with PCA + LDA
            threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
            print(f"Threshold: {threshold:.5f}")

            # Classify preprocessed DVAL with estimated threshold
            PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
            PVAL[DVAL_lda[0] >= threshold] = 1 # Predict class 1 for elements greater than the threshold
            PVAL[DVAL_lda[0] < threshold] = 0 # Predict class 0 for elements lower than the threshold

            # Compute PCA + LDA prediction error rate
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"PCA + LDA error rate: {err:.5f}")
        case 0:
            return
    
# ----------------------------
#  Generative Gaussian Models
# ----------------------------
def compare_gaussian_models(D, L):

    inner_menu_option = int(input('\n Generative Gaussian Models Menu:\n\
                                    1. Multivariate Gaussian Classifier\n\
                                    2. Naive Bayes Gaussian Classifier\n\
                                    3. Tied Gaussian Classifier\n\
                                    0. Back\n'))

    first_feature, last_feature = int(input("\nFeatures range to consider (from 1 to 6): "))-1, int(input("to "))
    D_sel = D[first_feature:last_feature, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_sel, L)

    pca_selection = int(input("\nPreprocessing with PCA? (1 for yes, 0 for no): "))
    match pca_selection:
        case 1:
            m = int(input("\nNumber of PCA directions: "))
            PCA = PrincipalComponentAnalysis()
            PCA.train(DTR, m)
            DTR = PCA.apply(DTR)
            DVAL = PCA.apply(DVAL)
        case 0:
            pass

    match inner_menu_option:
        case 1:
            # --- MVG ---
            MVG = MultivariateGaussianClassifier()
            MVG.train(DTR, LTR)
            PVAL = MVG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"MVG error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
        case 2:
            # --- Naive Bayes Gaussian ---
            NBG = NaiveBayesGaussianClassifier()
            NBG.train(DTR, LTR)
            PVAL = NBG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"Naive Bayes Gaussian error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
        case 3:
            # --- Tied Gaussian ---
            TG = TiedGaussianClassifier()
            TG.train(DTR, LTR)
            PVAL = TG.predict_binary(DVAL)
            acc, err = compute_acc_err(PVAL, LVAL)
            print(f"Tied Gaussian error rate - features {first_feature+1} to {last_feature}: {err:.5f}")
        case 0:
            return

# -----------------------
#  TODO: Evaluation/Bayes Risk
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
#  TODO: Logistic Regression
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

# ------------------------
# TODO: Support Vector Machines
# ------------------------
def analyze_SVM_with_different_kernels(DTR, LTR, DVAL, LVAL):
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
    
# ------------------------
# TODO: Gaussian Mixture Models
# ------------------------
def analyze_GMM_with_different_components(DTR, LTR, DVAL, LVAL):
    n_classes_binary = len(np.unique(L))
    components_to_test = [1, 2, 4, 8, 16]

    print("\n--- GMM for binary classification ---")
    for n_components in components_to_test:
        gmm_per_class_binary = {}
        for c in range(n_classes_binary):
            DTR_c = DTR[:, LTR == c]
            gmm_per_class_binary[c] = train_GMM_LBG_EM(DTR_c, n_components)

        logSPost_binary = np.zeros((n_classes_binary, DVAL.shape[1]))
        for c in range(n_classes_binary):
            logSPost_binary[c, :] = logpdf_GMM(DVAL, gmm_per_class_binary[c]) + np.log(1/n_classes_binary)

        llr_binary = logSPost_binary[1, :] - logSPost_binary[0, :]

        PVAL_binary = np.argmax(logSPost_binary, axis=0)

        print(f"Components: {n_components}: actual DCF: {compute_normalized_DCF(0.1, 1.0, 1.0, compute_confusion_matrix(PVAL_binary, LVAL)):.4f}")

# TODO: Move these functions to a separate files
def plot_min_act_actcal_DCF_for_n_systems(raw_scores_list, calibrated_scores_list, LVAL, pi, system_names):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Print the name of all the systems
    print(f"Computing Bayes Errors on raw scores of {len(raw_scores_list)} systems: {', '.join(system_names)}...")

    rawActDCFs_list = []
    calActDCFs_list = []
    minDCFs_list = []
    
    total_iters = len(effPriors)
    for i, effPrior in enumerate(effPriors):
        print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')

        rawActDCFs = []
        calActDCFs = []
        minDCFs = []
        for raw_scores, calibrated_scores in zip(raw_scores_list, calibrated_scores_list):
            # Compute optimal decisions for raw scores
            PVAL_raw = compute_optimal_bayes_decisions(effPrior, raw_scores, LVAL)
            conf_matr_raw = compute_confusion_matrix(PVAL_raw, LVAL)
            rawActDCFs.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_raw))
            minDCFs.append(compute_normalized_minDCF(raw_scores, LVAL, effPrior, 1.0, 1.0))
            # Compute optimal decisions for calibrated scores
            PVAL_calibrated = compute_optimal_bayes_decisions(effPrior, calibrated_scores, LVAL)
            conf_matr_calibrated = compute_confusion_matrix(PVAL_calibrated, LVAL)
            calActDCFs.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_calibrated))
        rawActDCFs_list.append(rawActDCFs)
        calActDCFs_list.append(calActDCFs)
        minDCFs_list.append(minDCFs)
    print("Progress: 100.0%")

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    plt.figure()
    for i in range(len(system_names)):
        c = colors[i % len(colors)]
        plt.plot(effPriorLogOdds, [rawActDCFs[i] for rawActDCFs in rawActDCFs_list], label=f"{system_names[i]} - actDCF (raw)", color=c, linestyle=':')
        plt.plot(effPriorLogOdds, [calActDCFs[i] for calActDCFs in calActDCFs_list], label=f"{system_names[i]} - actDCF (calibrated)", color=c, linestyle='--')
        plt.plot(effPriorLogOdds, [minDCFs[i] for minDCFs in minDCFs_list], label=f"{system_names[i]} - minDCF", color=c, linestyle='-')
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Effective Prior Log Odds for Multiple Systems')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

def plot_min_act_DCF_for_n_systems(scores_list, LVAL, pi, system_names):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    # Print the name of all the systems
    print(f"Computing Bayes Errors on scores of {len(scores_list)} systems: {', '.join(system_names)}...")

    actDCFs_list = []
    minDCFs_list = []
    
    total_iters = len(effPriors)
    for i, effPrior in enumerate(effPriors):
        print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')

        actDCFs = []
        minDCFs = []
        for scores in scores_list:
            # Compute optimal decisions for raw scores
            PVAL_raw = compute_optimal_bayes_decisions(effPrior, scores, LVAL)
            conf_matr_raw = compute_confusion_matrix(PVAL_raw, LVAL)
            actDCFs.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_raw))
            minDCFs.append(compute_normalized_minDCF(scores, LVAL, effPrior, 1.0, 1.0))
        actDCFs_list.append(actDCFs)
        minDCFs_list.append(minDCFs)
    print("Progress: 100.0%")

    
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    plt.figure()
    for i in range(len(system_names)):
        c = colors[i % len(colors)]
        plt.plot(effPriorLogOdds, [actDCFs[i] for actDCFs in actDCFs_list], label=f"{system_names[i]} - actDCF", color=c, linestyle='--')
        plt.plot(effPriorLogOdds, [minDCFs[i] for minDCFs in minDCFs_list], label=f"{system_names[i]} - minDCF", color=c, linestyle='-')
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Effective Prior Log Odds for Multiple Systems')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

# ------------------------------
# TODO: Scores Calibration and Fusion
# ------------------------------
def analyze_k_fold_calibration_impact():
    # --- LAB 9 ---
    # Qualitative analysis of Logistic Regression vs SVM vs GMM models for different applications
        
    # Train & Score Weighted Logistic Regression
    print("Training Weighted Logistic Regression...")
    pEmp = (LTR == 1).sum() / LTR.size
    lamb = 10 ** -1.5
    w, b = trainLogRegWeighted(DTR, LTR, lamb, pEmp)
    sVal_lr_bias = np.dot(w.T, DVAL) + b
    sVal_lr = (sVal_lr_bias - np.log(pEmp / (1 - pEmp))).ravel() # Validation scores for Logistic Regression -> to be calibrated

    # Train & Score SVM with RBF Kernel
    print("Training SVM with RBF Kernel...")
    kernelFunc = rbfKernel(math.exp(-2))
    C = 10 ** 1.5
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
    sVal_svm = fScore(DVAL) # Validation scores for SVM with RBF Kernel -> to be calibrated

    # Train & Score GMM with 8 components
    print("Training GMM with 8 components...")
    n_components = 8
    n_classes_binary = len(np.unique(L))
    gmm_per_class_binary = {}
    for c in range(n_classes_binary):
        print(f"Progress: {c / n_classes_binary * 100:.1f}%", end='\r')
        DTR_c = DTR[:, LTR == c]
        gmm_per_class_binary[c] = train_GMM_LBG_EM(DTR_c, n_components)
    print("Progress: 100.0%")
    logSPost_binary = np.zeros((n_classes_binary, DVAL.shape[1]))
    for c in range(n_classes_binary):
        logSPost_binary[c, :] = logpdf_GMM(DVAL, gmm_per_class_binary[c]) + np.log(1/n_classes_binary)
    sVal_gmm = logSPost_binary[1, :] - logSPost_binary[0, :] # Validation scores for GMM with 8 components -> to be calibrated

    # Analyze results for different applications (effective priors)
    # effPriorLogOdds = np.linspace(-4, 4, 21)
    # effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds)) # Array of effective priors from 0.018 to 0.982 (different applications)
    
    # print("Computing Bayes Errors on raw scores of the three models...")
    # actDCFs_lr, minDCFs_lr = [], []
    # actDCFs_svm, minDCFs_svm = [], []
    # actDCFs_gmm, minDCFs_gmm = [], []

    # total_iters = len(effPriors)
    # for i, effPrior in enumerate(effPriors):
    #     print(f"Progress: {i / total_iters * 100:.1f}%", end='\r')

    #     # Logistic Regression
    #     PVAL_lr = compute_optimal_bayes_decisions(effPrior, sVal_lr, LVAL)
    #     conf_matr_lr = compute_confusion_matrix(PVAL_lr, LVAL)
    #     # minDCFs_lr.append(compute_normalized_minDCF(llr_lr, LVAL, effPrior, 1.0, 1.0))
    #     actDCFs_lr.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_lr))

    #     # SVM
    #     PVAL_svm = compute_optimal_bayes_decisions(effPrior, sVal_svm, LVAL)
    #     conf_matr_svm = compute_confusion_matrix(PVAL_svm, LVAL)
    #     # minDCFs_svm.append(compute_normalized_minDCF(llr_svm, LVAL, effPrior, 1.0, 1.0))
    #     actDCFs_svm.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_svm))

    #     # GMM
    #     PVAL_gmm = compute_optimal_bayes_decisions(effPrior, sVal_gmm, LVAL)
    #     conf_matr_gmm = compute_confusion_matrix(PVAL_gmm, LVAL)
    #     # minDCFs_gmm.append(compute_normalized_minDCF(llr_gmm, LVAL, effPrior, 1.0, 1.0))
    #     actDCFs_gmm.append(compute_normalized_DCF(effPrior, 1.0, 1.0, conf_matr_gmm))
    # print("Progress: 100.0%")

    # Plot DCFs for the three models
    # plt.figure()
    # plt.plot(effPriorLogOdds, minDCFs_lr, label="minDCF - Logistic Regression", color='r', linestyle='-')
    # plt.plot(effPriorLogOdds, actDCFs_lr, label="actDCF - Logistic Regression", color='r', linestyle='--')
    
    # plt.plot(effPriorLogOdds, minDCFs_svm, label="minDCF - SVM", color='b', linestyle='-')
    # plt.plot(effPriorLogOdds, actDCFs_svm, label="actDCF - SVM", color='b', linestyle='--')
    
    # plt.plot(effPriorLogOdds, minDCFs_gmm, label="minDCF - GMM", color='g', linestyle='-')
    # plt.plot(effPriorLogOdds, actDCFs_gmm, label="actDCF - GMM", color='g', linestyle='--')

    # plt.ylim([0, 1.1])
    # plt.xlim([-4, 4])
    # plt.xlabel("Prior Log Odds")
    # plt.ylabel("DCF")
    # plt.title("Bayes Error Plot Comparison")
    # plt.legend()
    # plt.show()

    # --- Lab 10 ---
    # Compute calibration transformations for the three models on the validation set
    # Split scores and labels into K folds

    if (False):
        # CHECK_POINT
        # Load spydata from previous steps to avoid retraining models
        data, error_msg = load_dictionary("raw_scores.spydata")
        globals().update(data)
        
        D, L = loadData("data/trainData.txt")
        # Plot histograms for the features of the initial dataset
        #histsPlot(D, L, "", 6)
        
        # Split dataset in train and eval
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    K = 5
    sVal_lr_folds = [sVal_lr[i::K] for i in range(K)]
    sVal_svm_folds = [sVal_svm[i::K] for i in range(K)]
    sVal_gmm_folds = [sVal_gmm[i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]

    # Apply a K-fold cross-validation procedure to compute the optimal logistic regression parameters for calibration (C and K) for each model 
    calibrated_sVal_lr = np.zeros_like(sVal_lr)
    calibrated_sVal_svm = np.zeros_like(sVal_svm)
    calibrated_sVal_gmm = np.zeros_like(sVal_gmm)

    for k in range(K):
        # Train the model on K-1 folds and validate on the remaining fold
        SCAL_lr, SVAL_lr = np.hstack([sVal_lr_folds[i] for i in range(K) if i != k]), sVal_lr_folds[k]
        SCAL_svm, SVAL_svm = np.hstack([sVal_svm_folds[i] for i in range(K) if i != k]), sVal_svm_folds[k]
        SCAL_gmm, SVAL_gmm = np.hstack([sVal_gmm_folds[i] for i in range(K) if i != k]), sVal_gmm_folds[k]
        LCAL, LVAL_k = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]

        # Train calibration model (logistic regression) on the calibration training set with the application prior (pEmp)
        l = 1e-3
        w_lr, b_lr = trainLogRegWeighted(vrow(SCAL_lr), LCAL, l, pEmp)    # Calibration model for Logistic Regression
        w_svm, b_svm = trainLogRegWeighted(vrow(SCAL_svm), LCAL, l, pEmp) # Calibration model for SVM with RBF kernel
        w_gmm, b_gmm = trainLogRegWeighted(vrow(SCAL_gmm), LCAL, l, pEmp) # Calibration model for GMM with 8 components

        # Compute calibrated scores on the validation fold
        calibrated_sVal_lr[k::K] = (np.dot(w_lr.T, vrow(SVAL_lr)) + b_lr - np.log(pEmp / (1 - pEmp))).ravel()
        calibrated_sVal_svm[k::K] = (np.dot(w_svm.T, vrow(SVAL_svm)) + b_svm - np.log(pEmp / (1 - pEmp))).ravel()
        calibrated_sVal_gmm[k::K] = (np.dot(w_gmm.T, vrow(SVAL_gmm)) + b_gmm - np.log(pEmp / (1 - pEmp))).ravel()

    plot_min_act_actcal_DCF_for_n_systems(raw_scores_list=[sVal_lr, sVal_svm, sVal_gmm], calibrated_scores_list=[calibrated_sVal_lr, calibrated_sVal_svm, calibrated_sVal_gmm], LVAL=LVAL, pi=pEmp, system_names=["Logistic Regression", "SVM RBF Kernel", "GMM 8 Components"])

def analyze_score_level_fusion_impact():
    # Compute score-level fusion of the three models (weighted logistic regression, SVM with RBF kernel, GMM with 8 components)
    raw_scores_fusion = np.vstack([sVal_lr, sVal_svm, sVal_gmm])
    # Apply k fold cross-validation to train the fusion model (logistic regression) on the validation set with the application prior (pEmp)
    K = 5
    sVal_fusion_folds = [raw_scores_fusion[:, i::K] for i in range(K)]
    LVAL_folds = [LVAL[i::K] for i in range(K)]

    calibrated_sVal_fusion = np.zeros_like(raw_scores_fusion[0])
    for k in range(K):
        # Train the model on K-1 folds and validate on the remaining fold
        SCAL_fusion, SVAL_fusion = np.hstack([sVal_fusion_folds[i] for i in range(K) if i != k]), sVal_fusion_folds[k]
        LCAL, LVAL_k = np.hstack([LVAL_folds[i] for i in range(K) if i != k]), LVAL_folds[k]

        l = 1e-3
        pEmp = (LTR == 1).sum() / LTR.size
        w_fusion, b_fusion = trainLogRegWeighted(SCAL_fusion, LCAL, l, pEmp)

        calibrated_sVal_fusion[k::K] = (np.dot(w_fusion.T, SVAL_fusion) + b_fusion - np.log(pEmp / (1 - pEmp))).ravel()
    
    # Compute and print DCF for the fused system
    plot_min_act_DCF_for_n_systems(scores_list=[calibrated_sVal_lr, calibrated_sVal_svm, calibrated_sVal_gmm, calibrated_sVal_fusion], LVAL=LVAL, pi=pEmp, system_names=["Logistic Regression", "SVM RBF Kernel", "GMM 8 Components", "Fused System"])
    
def final_evaluation():
    # Split dataset in train and eval
    #(DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
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
            
    # CHECK_POINT
    # Load calibrated_scores
    data, error_msg = load_dictionary("calibrated_scores.spydata")
    globals().update(data)
    
    D, L = loadData("data/trainData.txt")
    # Plot histograms for the features of the initial dataset
    #histsPlot(D, L, "", 6)
    
    # Split dataset in train and eval
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    # --- Final Evaluation ---
    # Load Evaluation data and compute scores for the three models and the fused system
    DEVAL, LEVAL = loadData("data/evalData.txt")
    pi = pEmp

    print("\n--- Evaluating Models on Evaluation Set ---")

    # Compute evaluation scores for Logistic Regression
    lamb = 10 ** -1.5
    w, b = trainLogRegWeighted(DTR, LTR, lamb, pi)
    raw_sEval_lr = np.dot(w.T, DEVAL) + b - np.log(pi / (1 - pi))     # raw eval scores

    # Compute evaluation scores for SVM with RBF Kernel
    kernelFunc = rbfKernel(math.exp(-2))
    C = 10 ** 1.5
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
    raw_sEval_svm = fScore(DEVAL)                                           # raw eval scores

    # Compute evaluation scores for GMM with 8 components
    n_components = 8
    n_classes_binary = len(np.unique(L))
    gmm_per_class_binary = {}
    for c in range(n_classes_binary):
        DTR_c = DTR[:, LTR == c]
        gmm_per_class_binary[c] = train_GMM_LBG_EM(DTR_c, n_components)

    logSPost_binary_eval = np.zeros((n_classes_binary, DEVAL.shape[1]))
    for c in range(n_classes_binary):
        logSPost_binary_eval[c, :] = logpdf_GMM(DEVAL, gmm_per_class_binary[c]) + np.log(1/n_classes_binary)
    raw_sEval_gmm = logSPost_binary_eval[1, :] - logSPost_binary_eval[0, :] # raw eval scores

    # Train calibration models on whole DVAL raw scores
    l = 1e-3
    pEmp = (LTR == 1).sum() / LTR.size
    w_cal_lr, b_cal_lr = trainLogRegWeighted(vrow(sVal_lr), LVAL, l, pEmp)
    w_cal_svm, b_cal_svm = trainLogRegWeighted(vrow(sVal_svm), LVAL, l, pEmp)
    w_cal_gmm, b_cal_gmm = trainLogRegWeighted(vrow(sVal_gmm), LVAL, l, pEmp)

    raw_scores_fusion = np.vstack([sVal_lr, sVal_svm, sVal_gmm])
    w_cal_fusion, b_cal_fusion = trainLogRegWeighted(raw_scores_fusion, LVAL, l, pEmp)

    # Compute calibrated evaluation scores
    cal_sEval_lr = (np.dot(w_cal_lr.T, vrow(raw_sEval_lr)) + b_cal_lr - np.log(pi / (1 - pi))).ravel()
    cal_sEval_svm = (np.dot(w_cal_svm.T, vrow(raw_sEval_svm)) + b_cal_svm - np.log(pi / (1 - pi))).ravel()
    cal_sEval_gmm = (np.dot(w_cal_gmm.T, vrow(raw_sEval_gmm)) + b_cal_gmm - np.log(pi / (1 - pi))).ravel()
    raw_sEval_fusion = np.vstack([raw_sEval_lr, raw_sEval_svm, raw_sEval_gmm])
    cal_sEval_fusion = (np.dot(w_cal_fusion.T, raw_sEval_fusion) + b_cal_fusion - np.log(pi / (1 - pi))).ravel()

    # Plot DCFs for the three models and the fused system on the evaluation set
    plot_min_act_DCF_for_n_systems(scores_list=[cal_sEval_lr, cal_sEval_svm, cal_sEval_gmm, cal_sEval_fusion], LVAL=LEVAL, pi=pi, system_names=["Logistic Regression", "SVM RBF Kernel", "GMM 8 Components", "Fused System"])

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)
    D, L = loadData("data/trainData.txt")
    
    while True:
        menu_option = int(input("\nMenu\n\
                                    1. Dimensionality Reduction\n\
                                    2. Generative Gaussian Models\n\
                                    0. Exit\n"))

        match menu_option:
            case 1:
                PCA_LDA_effects_and_classification_analysis(D, L)
            case 2:
                compare_gaussian_models(D, L)
            case 0:
                break