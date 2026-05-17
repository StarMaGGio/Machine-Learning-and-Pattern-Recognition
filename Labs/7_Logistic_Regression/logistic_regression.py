import numpy as np
import scipy.optimize as op
import sklearn

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

def f(yz):
    y = yz[0]
    z = yz[1]
    
    fx = (y + 3)**2 + np.sin(y) + (z + 1)**2
    
    return fx

def f_grad(yz):
    y = yz[0]
    z = yz[1]
    
    fx = (y + 3)**2 + np.sin(y) + (z + 1)**2
    grady = 2*(y + 3) + np.cos(y)
    gradz = 2*(z + 1)
    
    return fx, np.array([grady, gradz])

# return filtered dataset (versicolor = 1, virginica = 0)
def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]
    L = L[L != 0]
    L[L==2] = 0
    return D, L

def split_db_2to1(D, L, seed = 0):
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

def compute_bayes_risk(pi, Cfn, Cfp, cm):
    Pfn = cm[0][1]/(cm[0][1]+cm[1][1])
    Pfp = cm[1][0]/(cm[1][0]+cm[0][0])
    
    return pi*Cfn*Pfn + (1-pi)*Cfp*Pfp

def compute_normalized_DCF(pi, Cfn, Cfp, DCFu):
    return DCFu/min(pi*Cfn, (1-pi)*Cfp)

def compute_normalized_minDCF(evalset_llr_binary, LVAL, pi, Cfn, Cfp):
    minDCF = 1000
    predicted_labels_binary = np.zeros(LVAL.size, dtype=np.int32)
    # for each "threshold" in the llr listS
    for t in np.unique(evalset_llr_binary):
        # Compute predicted labels using the score as a threshold
        predicted_labels_binary[evalset_llr_binary > t] = 1
        predicted_labels_binary[evalset_llr_binary <= t] = 0
        
        # Compute confusion matrix
        conf = compute_confusion_matrix(predicted_labels_binary, LVAL)
        
        # Compute normalized DCF
        currentDCF = compute_normalized_DCF(pi, Cfn, Cfp, compute_bayes_risk(pi, Cfn, Cfp, conf))
    
        minDCF = min(minDCF, currentDCF)
    
    return minDCF

def trainLogReg(DTR, LTR, l):
    
    ZTR = LTR * 2.0 - 1.0
    
    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, DTR).ravel() + b
        
        loss = np.logaddexp(0, -ZTR * S)
        
        G = -ZTR / (1.0 + np.exp(ZTR * S))
        Gw = l*w.ravel() + (vrow(G)*DTR).mean(1)
        Gb = G.mean()
        
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([Gw, np.array(Gb)])
    
    vf = op.fmin_l_bfgs_b(func=logreg_obj, x0=np.zeros(DTR.shape[0]+1))[0]
    print("Log-reg - Lambda = %e - J*(w, b) = %e" % (l, logreg_obj(vf)[0]))
    return vf[: -1], vf[-1]

if __name__ == "__main__":
    # --- Numerical optimization ---
    # Approximated Gradient
    #x1, f1, d1 = op.fmin_l_bfgs_b(f, np.array([0,0]), approx_grad=True)
    # Explicit Gradient
    #x2, f2, d2 = op.fmin_l_bfgs_b(f_grad, np.array([0,0]))
    
    # --- Binary logistic regression
    # Load and Split dataset
    D, L = load_iris_binary()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Train LogReg model
    for lamb in [1e-3, 1e-1, 1.0]:
        w, b = trainLogReg(DTR, LTR, lamb) # Train model -> obtain model parameters w and b
        S = np.dot(w.T, DVAL) + b # Compute validation scores
        PVAL = np.zeros(LVAL.shape[0]) # Predict validation labels
        PVAL[S > 0] = 1
        PVAL[S < 0] = 0
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err*100))
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size # Fraction of class 1 samples
        # Compute LLR-like scores
        Sllr = S - np.log(pEmp / (1 - pEmp))
        # Compute optimal decisions for the three priors
        print('minDCF - pT = 0.5: %.4f' % compute_normalized_minDCF(Sllr, LVAL, 0.5, 1.0, 1.0))
    
        print()
