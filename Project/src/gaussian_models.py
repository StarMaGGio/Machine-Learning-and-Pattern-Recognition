import numpy as np
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND
from src.utils import computeCovariance

def compute_llr_for_classification(X, mu0, mu1, C0, C1):
    return logpdf_GAU_ND(X, mu1, C1) - logpdf_GAU_ND(X, mu0, C0)

def compute_predictions_with_llr(llr, size, t):
    PVAL = np.zeros(size, np.int32)
    PVAL[llr < t] = 0
    PVAL[llr >= t] = 1
    
    return PVAL

def compute_error_rate(PVAL, LVAL):
    n_correct_predictions = np.array([PVAL == LVAL]).sum()
    acc = n_correct_predictions / PVAL.shape[0]
    return 1 - acc

def compute_llr_MVG(DTR, LTR, DVAL):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    
    # --- MVG ---
    # Compute ML estimates for model parameters (mu, C)
    C0, mu0 = computeCovariance(DTR0)
    C1, mu1 = computeCovariance(DTR1)
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, C0, C1)
    
    return LLRs
    
def compute_llr_Tied_Gaussian(DTR, LTR, DVAL):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    
    C0, mu0 = computeCovariance(DTR0)
    C1, mu1 = computeCovariance(DTR1)
    
    # --- Tied Gaussian ---
    # Compute ML estimates for model parameters (mu, Sw)
    Sw = ((C0*DTR0.shape[1])+(C1*DTR1.shape[1]))/float(DTR.shape[1])
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, Sw, Sw)
    
    return LLRs
    
def compute_llr_Naive_Bayes_Gaussian(DTR, LTR, DVAL):
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    
    C0, mu0 = computeCovariance(DTR0)
    C1, mu1 = computeCovariance(DTR1)
    
    # --- Naive Bayes Gaussian ---
    # Compute ML estimates for model parameters (mu, Ct)
    Ct0 = C0 * np.identity(C0.shape[1])
    Ct1 = C1 * np.identity(C1.shape[1])
    
    # Compute LLRs
    LLRs = compute_llr_for_classification(DVAL, mu0, mu1, Ct0, Ct1)
    
    return LLRs