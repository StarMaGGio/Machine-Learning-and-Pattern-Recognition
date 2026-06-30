# pyrefly: ignore [missing-import]
import numpy as np
from src.utils import compute_confusion_matrix

def compute_bayes_risk(pi, Cfn, Cfp, cm):
    """
    Compute Bayes risk (un-normalized actual DCF) for a given confusion matrix.
    
    Args:
        pi (float): Prior probability of genuine sample.
        Cfn (float): Cost of false negative.
        Cfp (float): Cost of false positive.
        cm (np.ndarray): Confusion matrix.
    
    Returns:
        float: Bayes risk.
    """
    # Compute false negative and false positive probabilities
    Pfn = cm[0][1]/(cm[0][1]+cm[1][1])
    Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
    
    return pi*Cfn*Pfn + (1-pi)*Cfp*Pfp

def compute_actual_DCF(pi, cm, Cfn=1.0, Cfp=1.0):
    """
    Compute (normalized) actual DCF for a given confusion matrix.
    
    Args:
        pi (float): Prior probability of genuine sample.
        cm (np.ndarray): Confusion matrix.
        Cfn (float): Cost of false negative.
        Cfp (float): Cost of false positive.
    
    Returns:
        float: Actual DCF.
    """
    # Compute un-normalized actual DCF
    DCFu = compute_bayes_risk(pi, Cfn, Cfp, cm)
    
    # Compute normalized actual DCF
    return DCFu/min(pi*Cfn, (1-pi)*Cfp)

def compute_minimum_DCF(LLRs, LVAL, pi, Cfn=1.0, Cfp=1.0):
    """
    Compute normalized minDCF for a given set of log-likelihood ratios.
    
    Args:
        LLRs (np.ndarray): Log-likelihood ratios.
        LVAL (np.ndarray): True labels.
        pi (float): Prior probability of genuine sample.
        Cfn (float): Cost of false negative.
        Cfp (float): Cost of false positive.
    
    Returns:
        float: Normalized minDCF.
    """
    minDCF = 1000
    
    for score in np.unique(LLRs):
        # Compute predicted labels using the score of the current sample as a threshold
        PVAL = np.zeros(LVAL.shape[0], dtype=np.int32)
        PVAL[LLRs > score] = 1
        PVAL[LLRs <= score] = 0
        
        # Compute normalized DCF
        currentDCF = compute_actual_DCF(pi, compute_confusion_matrix(PVAL, LVAL), Cfn, Cfp)
    
        # Update minimum DCF
        minDCF = min(minDCF, currentDCF)
    
    # Return the minimum DCF possible on the evaluation set
    return minDCF

def compute_predictions_with_llr(llr, t):
    """
    Function to compute predictions from log-likelihood ratios.
    Compare LLRs with a threshold t and return predictions.

    Parameters
    ----------
    llr : (numpy.ndarray)
        Log-likelihood ratios of shape (n_samples,).
    t : (float)
        Decision threshold.

    Returns
    -------
    PVAL : (numpy.ndarray)
        Predictions of shape (n_samples,).

    """
    PVAL = np.zeros(llr.shape[1], dtype=np.int32)
    PVAL[llr >= t] = 1
    PVAL[llr < t] = 0
    
    return PVAL
