# pyrefly: ignore [missing-import]
import numpy as np

def compute_acc_err(PVAL, LVAL):
    """
    Compute accuracy and error rate from predictions and labels

    Args:
        PVAL: predictions (array)
        LVAL: labels (array)
    
    Returns:
        acc: accuracy (float)
        err: error rate (float)
    """
    n_correct_predictions = np.array([PVAL == LVAL]).sum()
    acc = n_correct_predictions / PVAL.shape[0]
    return acc, 1 - acc