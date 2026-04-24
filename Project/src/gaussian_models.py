import numpy as np
from src.ML_estimate_for_Gaussian import logpdf_GAU_ND

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