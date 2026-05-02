import numpy as np
import matplotlib.pyplot as plt

def vrow(x):
    return x.reshape((1, x.size))

def vcol(x):
    return x.reshape((x.size, 1))

def compute_confusion_matrix(predictions, true_labels):
    n = len(np.unique(true_labels))
    conf_matrix = np.zeros((n, n), np.int32)
    
    for p,t in zip(predictions, true_labels):
        conf_matrix[p][t] += 1
        
    return conf_matrix

def compute_optimal_bayes_decision_threshold(pi, Cfn, Cfp):
    return np.log(((1-pi)*Cfp)/(pi*Cfn))

def compute_bayes_risk(pi, Cfn, Cfp, cm):
    Pfn = cm[0][1]/(cm[0][1]+cm[1][1])
    Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
    
    return pi*Cfn*Pfn + (1-pi)*Cfp*Pfp
    
def compute_normalized_DCF(pi, Cfn, Cfp, DCFu):
    return DCFu/min(pi*Cfn, (1-pi)*Cfp)

def plot_ROC_curve(evalset_llr_binary, evalset_labels_binary, pi, Cfn, Cfp):
    x = []
    y = []
    plt.figure()
    for t in evalset_llr_binary:
        predicted_labels_binary[evalset_llr_binary > t] = 1
        predicted_labels_binary[evalset_llr_binary <= t] = 0
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(predicted_labels_binary, evalset_labels_binary)
        
        Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
        Ptp = 1 - cm[0][1]/(cm[0][1]+cm[1][1])
        
        x.append(Pfp)
        y.append(Ptp)
        
    x = np.array(x)
    y = np.array(y)
    ordered_idx = np.argsort(x)
    x = x[ordered_idx]
    y = y[ordered_idx]
    
    plt.plot(x, y)
    plt.title("ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

if __name__ == "__main__":
    
    # --- Confusion matrices ---
    # Load data
    evalset_ll = np.load("Data/evalset_ll.npy") # class conditional log-likelihoods
    evalset_labels = np.load("Data/evalset_labels.npy") # ground truth labels
    
    # Predicted classes
    predicted_labels = evalset_ll.argmax(axis=0)
    
    # Confusion matrix
    conf_matrix = compute_confusion_matrix(predicted_labels, evalset_labels)
    print("Confusion Matrix with Uniform priors and costs")
    print(conf_matrix)
    print()
    
    # --- Binary task: optimal Bayes decision (threshold) ---
    print("Binary Task")
    # Load data
    evalset_llr_binary = np.load("Data/evalset_llr_binary.npy")
    evalset_labels_binary = np.load("Data/evalset_labels_binary.npy")
    pi = 0.5
    Cfn = 10
    Cfp = 1
    print(f"pi: {pi} - Cfn: {Cfn} - Cfp: {Cfp}")
    
    # Compute optimal bayes decision (threshold)
    r = compute_optimal_bayes_decision_threshold(pi, Cfn, Cfp)
    
    # Predict classes using optimal bayes decision
    predicted_labels_binary = np.zeros(evalset_labels_binary.shape[0], dtype=np.int32)
    predicted_labels_binary[evalset_llr_binary > r] = 1
    predicted_labels_binary[evalset_llr_binary <= r] = 0
    
    # Compute confusion matrix
    conf = compute_confusion_matrix(predicted_labels_binary, evalset_labels_binary)
    print("Confusion Matrix")
    print(conf)
    print()
    
    # --- Binary task: evaluation ---
    # empirical Bayes Risk / Detection Cost Function
    DCFu = compute_bayes_risk(pi, Cfn, Cfp, conf)
    print("Empirical Bayes risk")
    print(DCFu)
    print()
    
    # Normalized DCF
    DCF = compute_normalized_DCF(pi, Cfn, Cfp, DCFu)
    print("Normalized DCF")
    print(DCF)
    print()
    
    # Minimum Detection costs
    minDCF = 1000
    # for each "threshold" in the llr list
    for t in np.unique(evalset_llr_binary):
        # Compute predicted labels using the score as a threshold
        predicted_labels_binary[evalset_llr_binary > t] = 1
        predicted_labels_binary[evalset_llr_binary <= t] = 0
        
        # Compute confusion matrix
        conf = compute_confusion_matrix(predicted_labels_binary, evalset_labels_binary)
        
        # Compute normalized DCF
        currentDCF = compute_normalized_DCF(pi, Cfn, Cfp, compute_bayes_risk(pi, Cfn, Cfp, conf))
    
        minDCF = min(minDCF, currentDCF)
        
    print("Min DCF")
    print(minDCF)
    
    # ROC curves
    plot_ROC_curve(evalset_llr_binary, evalset_labels_binary, pi, Cfn, Cfp)
        
    
    
    
    