import numpy as np
import scipy
import scipy.optimize as op
import sklearn.datasets
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load_iris_dataset():
    iris = sklearn.datasets.load_iris()
    D = iris.data.T
    L = iris.target
    return D, L

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C, mu

def compute_confusion_matrix(predictions, true_labels):
    n = len(np.unique(true_labels))
    conf_matrix = np.zeros((n, n), np.int32)
    
    for p,t in zip(predictions, true_labels):
        conf_matrix[p][t] += 1
        
    return conf_matrix

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

def logpdf_GAU_ND_fast(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((X-mu) * (P @ (X-mu))).sum(0)

def compute_optimal_bayes_decisions(effPrior, evalset_llr_binary, evalset_labels_binary):
    t = np.log((1-effPrior)/effPrior)
    predicted_labels_binary = np.zeros(evalset_labels_binary.shape[0], dtype=np.int32)
    predicted_labels_binary[evalset_llr_binary > t] = 1
    predicted_labels_binary[evalset_llr_binary <= t] = 0
    
    return predicted_labels_binary




def trainLogRegWeighted(DTR, LTR, l, pi):
    
    ZTR = LTR * 2.0 - 1.0
    
    # Compute weights for the two classes
    wTrue = pi / (ZTR > 0).sum()
    wFalse = (1 - pi) / (ZTR < 0).sum()
    
    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, DTR).ravel() + b
        
        # Compute loss
        loss = np.logaddexp(0, -ZTR * S)
        # Apply the weights (Epsilon) to the loss computation
        loss[ZTR>0] *= wTrue
        loss[ZTR<0] *= wFalse
        
        # Compute Gradient
        G = -ZTR / (1.0 + np.exp(ZTR * S))
        # Apply the weights to the gradient computation
        G[ZTR>0] *= wTrue
        G[ZTR<0] *= wFalse
        
        Gw = l*w.ravel() + (vrow(G)*DTR).sum(1)
        Gb = G.sum()
        
        return loss.sum() + l / 2 * np.linalg.norm(w)**2, np.hstack([Gw, np.array(Gb)])
    
    vf = op.fmin_l_bfgs_b(func=logreg_obj, x0=np.zeros(DTR.shape[0]+1))[0]
    print("Weighted Log-reg - Lambda = %.4f - J*(w, b) = %.4f" % (l, logreg_obj(vf)[0]))
    return vf[: -1], vf[-1]


def plot_min_act_DCF_for_two_systems(scores_1, scores_2, labels, pi, title):
    # Compute and plot min and act DCF for the two systems
    minDCFs_1, min_DCFs_2, act_DCFs_1, act_DCFs_2 = [], [], [], []
    prior_log_odds = np.linspace(-3, 3, 21)
    eff_priors = 1.0/(1.0+np.exp(-prior_log_odds))

    for eff_prior in eff_priors:
        PVAL_1 = compute_optimal_bayes_decisions(eff_prior, scores_1, labels)
        PVAL_2 = compute_optimal_bayes_decisions(eff_prior, scores_2, labels)
        minDCFs_1.append(compute_normalized_minDCF(scores_1, labels, eff_prior, 1.0, 1.0))
        min_DCFs_2.append(compute_normalized_minDCF(scores_2, labels, eff_prior, 1.0, 1.0))
        act_DCFs_1.append(compute_normalized_DCF(eff_prior, 1.0, 1.0, compute_confusion_matrix(PVAL_1, labels)))
        act_DCFs_2.append(compute_normalized_DCF(eff_prior, 1.0, 1.0, compute_confusion_matrix(PVAL_2, labels)))

    plt.figure()
    plt.plot(prior_log_odds, minDCFs_1, label="minDCF System 1", linestyle = "--", color = "lightblue")
    plt.plot(prior_log_odds, min_DCFs_2, label="minDCF System 2", linestyle = "--", color = "orange")
    plt.plot(prior_log_odds, act_DCFs_1, label="actDCF System 1", linestyle = "-", color = "lightblue")
    plt.plot(prior_log_odds, act_DCFs_2, label="actDCF System 2", linestyle = "-", color = "orange")
    plt.xlabel("Prior Log Odds")
    plt.ylabel("Normalized DCF")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title(title)
    plt.legend()
    plt.show()

def scores_calibration_single_fold(scores_1, scores_2, labels, pi):
        # Split the scores and labels into 3 folds
    SCAL1, SVAL1 = scores_1[::3], np.hstack([scores_1[1::3], scores_1[2::3]])
    SCAL2, SVAL2 = scores_2[::3], np.hstack([scores_2[1::3], scores_2[2::3]])
    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    # Plot the min and act DCF for the two systems on the calibration validation set
    plot_min_act_DCF_for_two_systems(SVAL1, SVAL2, LVAL, pi, "Single fold - sctual and minimum DCF of original raw scores - calibration validation set")

    # Train calibration model (logistic regression) on the calibration training set
    l = 1e-3
    w1, b1 = trainLogRegWeighted(vrow(SCAL1), LCAL, l, pi)
    w2, b2 = trainLogRegWeighted(vrow(SCAL2), LCAL, l, pi)
    
    # Compute calibrated scores on the validation set
    # Reshape validation scores to 1xN before applying weights and bias, and subtract the prior log-odds shift
    calibrated_SVAL1 = (np.dot(w1.T, vrow(SVAL1)) + b1 - np.log(pi / (1 - pi))).ravel()
    calibrated_SVAL2 = (np.dot(w2.T, vrow(SVAL2)) + b2 - np.log(pi / (1 - pi))).ravel()
    
    # Plot the min and act DCF for the two systems with calibrated scores
    plot_min_act_DCF_for_two_systems(calibrated_SVAL1, calibrated_SVAL2, LVAL, pi, "Single fold - actual and minimum DCF of calibrated scores - calibration validation set")


if __name__ == '__main__':
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")
    pi = 0.2

    #plot_min_act_DCF_for_two_systems(scores_1, scores_2, labels, pi, "single fold - actual and minimum DCF of original raw scores - full dataset")

    scores_calibration_single_fold(scores_1, scores_2, labels, pi)