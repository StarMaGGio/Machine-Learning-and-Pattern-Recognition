import numpy as np
import scipy.special

def vrow(x):
    return x.reshape((1, x.size))

def vcol(x):
    return x.reshape((x.size, 1))

def load(fileName):
    dataMatrix = []
    labels = []
    hLabels = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
        }
    with open(fileName) as f:
        for line in f:
            x_j = np.array([float(i) for i in line.split(",")[0:-1]])
            x_j = x_j.reshape((x_j.size, 1))
            dataMatrix.append(x_j)
            
            l_j = line.split(",")[-1].strip()
            labels.append(hLabels[l_j])
    return (np.hstack(dataMatrix), np.array(labels))

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

def computeCovariance(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    DC = D - mu
    C = DC @ DC.T / float(D.shape[1])
    return C

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5*X.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((X-mu) * (P @ (X-mu))).sum(0)

# ----------------------------------
#  Multivariate Gaussian Classifier
# ----------------------------------
def multivariate_gaussian_classifier(DTR, LTR, DVAL, LVAL):
    # --- Compute ML estimates for classifier parameters (mu, C) ---
    DTR0 = DTR[:, LTR == 0] # Columns of the training dataset corresponding to class 0
    DTR1 = DTR[:, LTR == 1] # Columns of the training dataset corresponding to class 1
    DTR2 = DTR[:, LTR == 2] # Columns of the training dataset corresponding to class 2
    
    mu0 = DTR0.mean(1).reshape((DTR0.shape[0], 1))
    mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
    mu2 = DTR2.mean(1).reshape((DTR2.shape[0], 1))
    
    C0 = computeCovariance(DTR0)
    C1 = computeCovariance(DTR1)
    C2 = computeCovariance(DTR2)
    
    # --- Compute class posterior probabilities P(c|x) ---
    # Step 1: compute likelihoods for each test sample
    log_density0 = logpdf_GAU_ND(DVAL, mu0, C0) # log(f_{X|C}(x_t|0))
    log_density1 = logpdf_GAU_ND(DVAL, mu1, C1) # log(f_{X|C}(x_t|1))
    log_density2 = logpdf_GAU_ND(DVAL, mu2, C2) # log(f_{X|C}(x_t|2))
    
    likelihoods0 = np.exp(log_density0) # f_{X|C}(x_t|0)
    likelihoods1 = np.exp(log_density1) # f_{X|C}(x_t|1)
    likelihoods2 = np.exp(log_density2) # f_{X|C}(x_t|2)
    
    S = np.array([likelihoods0, likelihoods1, likelihoods2]) # S[i, j] = class-conditional probability for sample j given class i
    
    # Step 2: compute joint distribution for test samples and classes
    SJoint = S * (1/3) # f_{X,C}(x_t,c) = f_{X|C}(x_t|c)*P_C(c)
    #SJoint_Sol = np.load("Solution/SJoint_MVG.npy")
    
    # Step 3: compute class posterior probabilities
    SMarginal = vrow(SJoint.sum(0)) # f_X(x_t) = sum_c(f_{X,C}(x_t,c))
    
    SPost = SJoint / SMarginal # P(C=c|X=x_t)
    #SPost_Sol = np.load("Solution/Posterior_MVG.npy")
    
    # Predicted classes
    Predicted_Labels = SPost.argmax(axis=0)
    
    # --- Compute model accuracy and error rate ---
    n_correct_predictions = np.array([Predicted_Labels == LVAL]).sum()
    
    accuracy = n_correct_predictions / DVAL.shape[1]
    error_rate = 1 - accuracy
    
    # --- Do it again with log-densities to address numerical issues ---
    logS = np.array([log_density0, log_density1, log_density2]) # logS[i, j] = class-conditional log-likelihood for sample j given class i
    
    # Compute log-densities for test samples and classes
    logSJoint = logS + vcol(np.log(1/3))
    #logSJoint_Sol = np.load("Solution/logSJoint_MVG.npy")
    
    # Compute log-posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    
    logSPost = logSJoint - logSMarginal
    SPost2 = np.exp(logSPost)
    
    return SPost, SPost2, error_rate

# ----------------------------------
#  Naive Bayes Gaussian Classifier
# ----------------------------------
def naive_bayes_gaussian_classifier(DTR, LTR, DVAL, LVAL):
    # --- Compute ML estimates for classifier parameters (mu, C) ---
    DTR0 = DTR[:, LTR == 0] # Columns of the training dataset corresponding to class 0
    DTR1 = DTR[:, LTR == 1] # Columns of the training dataset corresponding to class 1
    DTR2 = DTR[:, LTR == 2] # Columns of the training dataset corresponding to class 2
    
    mu0 = DTR0.mean(1).reshape((DTR0.shape[0], 1))
    mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
    mu2 = DTR2.mean(1).reshape((DTR2.shape[0], 1))
    
    MVGC0 = computeCovariance(DTR0)
    C0 = MVGC0 * np.identity(MVGC0.shape[0])
    MVGC1 = computeCovariance(DTR1)
    C1 = MVGC1 * np.identity(MVGC1.shape[0])
    MVGC2 = computeCovariance(DTR2)
    C2 = MVGC2 * np.identity(MVGC2.shape[0])
    
    # --- Compute class posterior probabilities P(c|x) ---
    
    # --- LIKELIHOOD DOMAIN ---
    # Step 1: compute likelihoods for each test sample
    log_density0 = logpdf_GAU_ND(DVAL, mu0, C0) # log(f_{X|C}(x_t|0))
    log_density1 = logpdf_GAU_ND(DVAL, mu1, C1) # log(f_{X|C}(x_t|1))
    log_density2 = logpdf_GAU_ND(DVAL, mu2, C2) # log(f_{X|C}(x_t|2))
    
    likelihoods0 = np.exp(log_density0) # f_{X|C}(x_t|0)
    likelihoods1 = np.exp(log_density1) # f_{X|C}(x_t|1)
    likelihoods2 = np.exp(log_density2) # f_{X|C}(x_t|2)
    
    S = np.array([likelihoods0, likelihoods1, likelihoods2]) # S[i, j] = class-conditional probability for sample j given class i
    
    # Step 2: compute joint distribution for test samples and classes
    SJoint = S * (1/3) # f_{X,C}(x_t,c) = f_{X|C}(x_t|c)*P_C(c)
    #SJoint_Sol = np.load("Solution/SJoint_MVG.npy")
    
    # Step 3: compute class posterior probabilities
    SMarginal = vrow(SJoint.sum(0)) # f_X(x_t) = sum_c(f_{X,C}(x_t,c))
    
    SPost = SJoint / SMarginal # P(C=c|X=x_t)
    #SPost_Sol = np.load("Solution/Posterior_MVG.npy")
    
    # Predicted classes
    Predicted_Labels = SPost.argmax(axis=0)
    
    # --- Compute model accuracy and error rate ---
    n_correct_predictions = np.array([Predicted_Labels == LVAL]).sum()
    
    accuracy = n_correct_predictions / DVAL.shape[1]
    error_rate = 1 - accuracy
    
    # --- LOG-LIKELIHOOD DOMAIN ---
    logS = np.array([log_density0, log_density1, log_density2]) # logS[i, j] = class-conditional log-likelihood for sample j given class i
    
    # Compute log-densities for test samples and classes
    logSJoint = logS + vcol(np.log(1/3))
    #logSJoint_Sol = np.load("Solution/logSJoint_MVG.npy")
    
    # Compute log-posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    
    logSPost = logSJoint - logSMarginal
    SPost2 = np.exp(logSPost)
    
    return SPost, SPost2, error_rate

# -------------------------------------
#  Tied Covariance Gaussian Classifier
# -------------------------------------
def tied_covariance_gaussian_classifier(DTR, LTR, DVAL, LVAL):
    # --- Compute ML estimates for classifier parameters (mu, C) ---
    DTR0 = DTR[:, LTR == 0] # Columns of the training dataset corresponding to class 0
    DTR1 = DTR[:, LTR == 1] # Columns of the training dataset corresponding to class 1
    DTR2 = DTR[:, LTR == 2] # Columns of the training dataset corresponding to class 2
    
    mu0 = DTR0.mean(1).reshape((DTR0.shape[0], 1))
    mu1 = DTR1.mean(1).reshape((DTR1.shape[0], 1))
    mu2 = DTR2.mean(1).reshape((DTR2.shape[0], 1))
    
    C0 = computeCovariance(DTR0)
    C1 = computeCovariance(DTR1)
    C2 = computeCovariance(DTR2)
    
    S_W = ((C0*DTR0.shape[1])+(C1*DTR1.shape[1])+(C2*DTR2.shape[1]))/float(DTR.shape[1])
    
    # --- Compute class posterior probabilities P(c|x) ---
    
    # --- LIKELIHOOD DOMAIN ---
    # Step 1: compute likelihoods for each test sample
    log_density0 = logpdf_GAU_ND(DVAL, mu0, S_W) # log(f_{X|C}(x_t|0))
    log_density1 = logpdf_GAU_ND(DVAL, mu1, S_W) # log(f_{X|C}(x_t|1))
    log_density2 = logpdf_GAU_ND(DVAL, mu2, S_W) # log(f_{X|C}(x_t|2))
    
    likelihoods0 = np.exp(log_density0) # f_{X|C}(x_t|0)
    likelihoods1 = np.exp(log_density1) # f_{X|C}(x_t|1)
    likelihoods2 = np.exp(log_density2) # f_{X|C}(x_t|2)
    
    S = np.array([likelihoods0, likelihoods1, likelihoods2]) # S[i, j] = class-conditional probability for sample j given class i
    
    # Step 2: compute joint distribution for test samples and classes
    SJoint = S * (1/3) # f_{X,C}(x_t,c) = f_{X|C}(x_t|c)*P_C(c)
    #SJoint_Sol = np.load("Solution/SJoint_MVG.npy")
    
    # Step 3: compute class posterior probabilities
    SMarginal = vrow(SJoint.sum(0)) # f_X(x_t) = sum_c(f_{X,C}(x_t,c))
    
    SPost = SJoint / SMarginal # P(C=c|X=x_t)
    #SPost_Sol = np.load("Solution/Posterior_MVG.npy")
    
    # Predicted classes
    Predicted_Labels = SPost.argmax(axis=0)
    
    # --- Compute model accuracy and error rate ---
    n_correct_predictions = np.array([Predicted_Labels == LVAL]).sum()
    
    accuracy = n_correct_predictions / DVAL.shape[1]
    error_rate = 1 - accuracy
    
    # --- LOG-LIKELIHOOD DOMAIN ---
    logS = np.array([log_density0, log_density1, log_density2]) # logS[i, j] = class-conditional log-likelihood for sample j given class i
    
    # Compute log-densities for test samples and classes
    logSJoint = logS + vcol(np.log(1/3))
    #logSJoint_Sol = np.load("Solution/logSJoint_MVG.npy")
    
    # Compute log-posterior probabilities
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    
    logSPost = logSJoint - logSMarginal
    SPost2 = np.exp(logSPost)
    
    return SPost, SPost2, error_rate
    
if __name__ == "__main__":
    # --- Load and split data ---
    D, L = load("iris.csv")
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    SPost, SPost2, error_rate = tied_covariance_gaussian_classifier(DTR, LTR, DVAL, LVAL)
    