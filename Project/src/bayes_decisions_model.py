import numpy as np
from src.utils import compute_confusion_matrix

def compute_optimal_bayes_decisions(effPrior, evalset_llr_binary, evalset_labels_binary):
    t = np.log((1-effPrior)/effPrior)
    predicted_labels_binary = np.zeros(evalset_labels_binary.shape[0], dtype=np.int32)
    predicted_labels_binary[evalset_llr_binary > t] = 1
    predicted_labels_binary[evalset_llr_binary <= t] = 0
    
    return predicted_labels_binary

def compute_bayes_risk(pi, Cfn, Cfp, cm):
    Pfn = cm[0][1]/(cm[0][1]+cm[1][1])
    Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
    
    return pi*Cfn*Pfn + (1-pi)*Cfp*Pfp

def compute_normalized_DCF(pi, Cfn, Cfp, cm):
    DCFu = compute_bayes_risk(pi, Cfn, Cfp, cm)
    
    return DCFu/min(pi*Cfn, (1-pi)*Cfp)

def compute_normalized_minDCF(evalset_llr_binary, evalset_labels_binary, pi, Cfn, Cfp):
    minDCF = 1000
    predicted_labels_binary = np.zeros(evalset_labels_binary.shape[0], dtype=np.int32)
    # for each "threshold" in the llr list
    for t in np.unique(evalset_llr_binary):
        # Compute predicted labels using the score as a threshold
        predicted_labels_binary[evalset_llr_binary > t] = 1
        predicted_labels_binary[evalset_llr_binary <= t] = 0
        
        # Compute confusion matrix
        conf = compute_confusion_matrix(predicted_labels_binary, evalset_labels_binary)
        
        # Compute normalized DCF
        currentDCF = compute_normalized_DCF(pi, Cfn, Cfp, conf)
    
        minDCF = min(minDCF, currentDCF)
    
    return minDCF