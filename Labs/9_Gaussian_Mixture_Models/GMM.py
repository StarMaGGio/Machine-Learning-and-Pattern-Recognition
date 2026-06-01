import numpy as np
import scipy
import sklearn.datasets
import matplotlib.pyplot as plt

from Data.GMM_load import load_gmm

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





# Computes the log-density of a GMM for a set of samples contained in matrix X
# gmm = [(w1, mu1, C1), (x2, mu2, C2), ...]
def logpdf_GMM(X, gmm):
    M = len(gmm)
    S = np.zeros((M, X.shape[1])) # S shape (M, N). S[g, ;] is the log-density of the g-th Gaussian component for all samples in X
    for g in range(M):
        w, mu, C = gmm[g]
        S[g, :] = logpdf_GAU_ND_fast(X, mu, C) + np.log(w)
    # We use the log-sum-exp trick to compute the log-density of the GMM
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens # Output is a vector (N,) where each element is the log-density of the GMM for the corresponding sample in X
    
# Algorithm to estimate the parameters of a GMM that maximize the likelihood for a training set X
def GMM_EM_estimation(X, gmm_init, psi=None, eps_ll=1e-6):
    M = len(gmm_init)
    gmm = gmm_init.copy()
    S = np.zeros((M, X.shape[1]))

    currentLoglikelihood = None

    while True:
        # Step 1 (E-Step): compute posterior prob. for each component of GMM for each sample
        for g in range(M):
            w, mu, C = gmm[g]
            S[g, :] = logpdf_GAU_ND_fast(X, mu, C) + np.log(w) # Joint density of g-th component and X
            
        logSMarginal = scipy.special.logsumexp(S, axis=0) # Marginal densities
        newLoglikelihood = logSMarginal.mean()
        
        # Stopping criterion: we can check the log-likelihood of the data given the model parameters and stop when it does not increase significantly anymore
        if currentLoglikelihood is not None and np.abs(newLoglikelihood - currentLoglikelihood) < eps_ll:
            break
            
        currentLoglikelihood = newLoglikelihood
        
        logSPost = S - logSMarginal
        SPost = np.exp(logSPost)

        # Step 2 (M-Step): Update model parameters
        for g in range(M):
            Z_g = SPost[g, :].sum() # Normalization factor for g-th component
            F_g = vcol((X * SPost[g, :]).sum(axis=1)) # Weighted sum of samples for g-th component
            S_g = (X * SPost[g, :]) @ X.T # Weighted sum of squared samples for g-th component

            new_mu_g = F_g / Z_g
            new_C_g = S_g / Z_g - new_mu_g @ new_mu_g.T
            new_w_g = Z_g / X.shape[1]
            
            # Regularize covariance matrix if psi is not None
            if psi is not None:
                U, s, Vh = np.linalg.svd(new_C_g)
                s[s < psi] = psi
                new_C_g = U @ (vcol(s) * U.T)

            gmm[g] = (new_w_g, new_mu_g, new_C_g)

    return gmm

# Function to split a Gaussian component into two components with mean shift alpha along the direction of maximum variance (used in LBG algorithm)
def LBG_split(gmm_in, alpha):
    gmm_out = []
    for w, mu, C in gmm_in:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmm_out.append((w/2, mu - d, C))
        gmm_out.append((w/2, mu + d, C))
    return gmm_out

def train_GMM_LBG_EM(X, numComponents, alpha=0.1, psi=0.01):
    C, mu = computeCovariance(X)
    
    # Regularize covariance matrix if psi is not None
    if psi is not None:
        U, s, Vh = np.linalg.svd(C)
        s[s < psi] = psi
        C = U @ (vcol(s) * U.T)
        
    gmm = [(1.0, mu, C)]
    while len(gmm) < numComponents:
        gmm = LBG_split(gmm, alpha) # Split each component into two components with mean shift alpha along the direction of maximum variance
        gmm = GMM_EM_estimation(X, gmm, psi=psi) # Estimate the parameters of the GMM with EM algorithm after splitting the components
    return gmm

if __name__ == '__main__':
    # --- Check logpdf_GMM with reference dataset ---
    GMM_data_4D = np.load("Data/GMM_data_4D.npy")
    GMM_4D_3G_init = load_gmm("Data/GMM_4D_3G_init.json")

    logdens = logpdf_GMM(GMM_data_4D, GMM_4D_3G_init)

    logdens_ref = np.load("Data/GMM_4D_3G_init_ll.npy")

    # --- GMM EM estimation ---
    # GMM EM estimation for 4D dataset with 3 components
    gmm_estimated = GMM_EM_estimation(GMM_data_4D, GMM_4D_3G_init)
    gmm_estimated_ref = load_gmm("Data/GMM_4D_3G_EM.json")

    # GMM EM estimation for 1D dataset with 3 components
    GMM_data_1D = np.load("Data/GMM_data_1D.npy")
    GMM_1D_3G_init = load_gmm("Data/GMM_1D_3G_init.json")
    
    gmm_estimated_1D = GMM_EM_estimation(GMM_data_1D, GMM_1D_3G_init)

    # Plot estimated GMM density for 1D dataset
    XPlot = np.linspace(-10, 5, 1000)
    logdens_estimated_1D = logpdf_GMM(vrow(XPlot), gmm_estimated_1D)
    plt.figure()
    plt.hist(GMM_data_1D.ravel(), bins=30, density=True)
    plt.plot(XPlot.ravel(), np.exp(logdens_estimated_1D))
    plt.show()

    # --- LBG algorithm ---
    gmm_estimated_4G = train_GMM_LBG_EM(GMM_data_4D, 4)
    gmm_estimated_4G_ref = load_gmm("Data/GMM_4D_4G_EM_LBG.json")

    # --- GMM for classification ---
    # Classification of full Iris dataset with GMMs trained with LBG algorithm for different number of components per class
    D, L = load_iris_dataset()
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    n_classes = len(np.unique(L))
    components_to_test = [1, 2, 4, 8, 16]

    print("\n--- GMM for classification ---")
    for n_components in components_to_test:
        # Train a GMM for each class
        gmm_per_class = {}
        for c in range(n_classes):
            DTR_c = DTR[:, LTR == c]
            gmm_per_class[c] = train_GMM_LBG_EM(DTR_c, n_components)

        # Compute log-posterior probabilities for the validation set
        logSPost = np.zeros((n_classes, DVAL.shape[1]))
        for c in range(n_classes):
            logSPost[c, :] = logpdf_GMM(DVAL, gmm_per_class[c]) + np.log(1/n_classes)

        # Get predictions by taking the class with the highest posterior probability
        PVAL = np.argmax(logSPost, axis=0)

        # Compute and print the error rate
        err = (PVAL != LVAL).sum() / LVAL.shape[0]
        print(f"Error rate for {n_components}-component GMMs per class: {err*100:.2f}%")

    # Binary classification task
    D_binary = np.load("Data/ext_data_binary.npy")
    L_binary = np.load("Data/ext_data_binary_labels.npy")

    (DTR_binary, LTR_binary), (DVAL_binary, LVAL_binary) = split_db_2to1(D_binary, L_binary)

    n_classes_binary = len(np.unique(L_binary))
    
    print("\n--- GMM for binary classification ---")
    for n_components in components_to_test:
        gmm_per_class_binary = {}
        for c in range(n_classes_binary):
            DTR_c = DTR_binary[:, LTR_binary == c]
            gmm_per_class_binary[c] = train_GMM_LBG_EM(DTR_c, n_components)

        logSPost_binary = np.zeros((n_classes_binary, DVAL_binary.shape[1]))
        for c in range(n_classes_binary):
            logSPost_binary[c, :] = logpdf_GMM(DVAL_binary, gmm_per_class_binary[c]) + np.log(1/n_classes_binary)

        llr_binary = logSPost_binary[1, :] - logSPost_binary[0, :]

        PVAL_binary = np.argmax(logSPost_binary, axis=0)

        print(f"Components: {n_components}: minDCF: {compute_normalized_minDCF(llr_binary, LVAL_binary, pi=0.5, Cfn=1, Cfp=1):.4f} / actDCF: {compute_normalized_DCF(pi=0.5, Cfn=1, Cfp=1, cm=compute_confusion_matrix(PVAL_binary, LVAL_binary)):.4f}")

