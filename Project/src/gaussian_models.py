# pyrefly: ignore [missing-import]
import scipy
# pyrefly: ignore [missing-import]
import numpy as np
from src.utils import vcol, vrow
from src.multivariate_gaussian_log_pdf import logpdf_GAU_ND
from src.utils import computeCovariance, computeMean

def loglikelihoods(X, mu_MLs, C_MLs):
    """
    Function to compute the value of per-class log-densities for a set of samples X.

    Parameters
    ----------
    X : (numpy.ndarray)
        Sample feature matrix of shape (n_features, n_samples).
    mu_MLs : (numpy.ndarray)
        Distribution of means vectors of shape (n_features, n_classes).
    C_MLs : (numpy.ndarray)
        Distribution of covariance matrices of shape (n_features, n_features, n_classes).

    Returns
    -------
    S : (numpy.ndarray)
        Class-conditional log-densities matrix of shape (n_classes, n_samples).

    """
    S = np.zeros((mu_MLs.shape[1], X.shape[1])) # S[i, j] = class-conditional probability for sample j given class i
    for c in range(S.shape[0]):
        S[c, :] = logpdf_GAU_ND(X, mu_MLs[:, c:c+1], C_MLs[:, :, c:c+1])
    return S

def logPosteriors(S_logLikelihoods, priors):
    """
    Function to compute the value of per-class log-posteriors for a set of samples X.

    Parameters
    ----------
    S_logLikelihoods : (numpy.ndarray)
        Class-conditional log-densities matrix of shape (n_classes, n_samples).
    priors : (numpy.ndarray)
        Prior probabilities of shape (n_classes,).

    Returns
    -------
    S_logPost : (numpy.ndarray)
        Class-conditional log-posterior matrix of shape (n_classes, n_samples).

    """
    S_logJoint = S_logLikelihoods + vcol(np.log(priors))
    S_logMarginal = vrow(scipy.special.logsumexp(S_logJoint, axis=0))
    S_logPost = S_logJoint - S_logMarginal
    return S_logPost

class MultivariateGaussianClassifier:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.covariances = {}
        
    def train(self, X, L):
        """
        Train the model by computing empirical mean and covariance for each class.
        (Maximum Likelihood approach)

        Parameters
        ----------
        X : (numpy.ndarray)
            Training Features matrix of shape (n_samples, n_features).
        L : (numpy.ndarray)
            Labels vector of shape (n_samples,).

        Returns
        -------
        None.

        """
        self.classes = np.unique(L)
        
        for c in self.classes:
            XTR_c = X[:, L == c] # Takes only samples from class c
            
            # Evaluate class means vector
            mu_c = computeMean(XTR_c)
            self.means[c] = mu_c
            
            # Evaluate class covariance matrix
            C_c = computeCovariance(XTR_c)
            self.covariances[c] = C_c
            
    def predict_multiclass(self, X):
        """
        Function to compute the optimal Bayes decision rule for a multiclass Gaussian model.

        Parameters
        ----------
        X : (numpy.ndarray)
            Validation Features matrix of shape (n_samples, n_features).

        Returns
        -------
        PVAL : (numpy.ndarray)
            Predictions of shape (n_samples,).

        """
        # Compute class-conditional log-likelihoods for validation samples
        S_logLikelihoods = loglikelihoods(X, self.means, self.covariances)

        # Compute class-conditional log-posteriors
        S_logPost = logPosteriors(S_logLikelihoods, priors=np.ones(len(self.classes))/len(self.classes))

        # Compute predictions
        PVAL = S_logPost.argmax(0)

        return PVAL

    def predict_binary(self, X, t=0):
        """
        Function to compute the optimal Bayes decision rule for a binary Gaussian model.

        Parameters
        ----------
        X : (numpy.ndarray)
            Validation Features matrix of shape (n_samples, n_features).
        t : (float)
            Decision threshold = log(P(w1)/(1-P(w1))). For the optimal Bayes classifier t = 0.

        Returns
        -------
        PVAL : (numpy.ndarray)
            Predictions of shape (n_samples,).

        """
        # Compute log-likelihood ratios
        LLRs = compute_llr_for_classification(X, self.means[0], self.means[1], self.covariances[0], self.covariances[1])
        
        # Compute predictions
        PVAL = compute_predictions_with_llr(LLRs, X.shape[1], t)
        
class TiedGaussianClassifier(MultivariateGaussianClassifier):
    def __init__(self):
        super().__init__()
        self.Sw = None
    
    def train(self, X, L):
        """
        Train the model by computing empirical mean and tied covariance matrix.
        (Maximum Likelihood approach)

        Parameters
        ----------
        X : (numpy.ndarray)
            Training Features matrix of shape (n_samples, n_features).
        L : (numpy.ndarray)
            Labels vector of shape (n_samples,).

        Returns
        -------
        None.

        """
        self.classes = np.unique(L)
        
        for c in self.classes:
            XTR_c = X[:, L == c] # Takes only samples from class c
            
            # Evaluate class means vector
            mu_c = computeMean(XTR_c)
            self.means[c] = mu_c

            # Evaluate class covariance matrix
            C_c = computeCovariance(XTR_c)
            self.covariances[c] = C_c

        # Compute Sw
        self.Sw = ((self.covariances[0]*X[:, L==0].shape[1])+(self.covariances[1]*X[:, L==1].shape[1]))/float(X.shape[1])
    
    def predict_binary(self, X, t=0):
        """
        Function to compute the optimal Bayes decision rule for a binary Gaussian model.

        Parameters
        ----------
        X : (numpy.ndarray)
            Validation Features matrix of shape (n_samples, n_features).
        t : (float)
            Decision threshold = log(P(w1)/(1-P(w1))). For the optimal Bayes classifier t = 0.

        Returns
        -------
        PVAL : (numpy.ndarray)
            Predictions of shape (n_samples,).

        """
        # Compute log-likelihood ratios
        LLRs = compute_llr_for_classification(X, self.means[0], self.means[1], self.Sw, self.Sw)
        
        # Compute predictions
        PVAL = compute_predictions_with_llr(LLRs, X.shape[1], t)

        return PVAL

class NaiveBayesGaussianClassifier(MultivariateGaussianClassifier):
    def __init__(self):
        super().__init__()
    
    def train(self, X, L):
        """
        Train the model by computing empirical mean and diagonal covariance for each class.
        (Maximum Likelihood approach)

        Parameters
        ----------
        X : (numpy.ndarray)
            Training Features matrix of shape (n_samples, n_features).
        L : (numpy.ndarray)
            Labels vector of shape (n_samples,).

        Returns
        -------
        None.

        """
        self.classes = np.unique(L)
        
        for c in self.classes:
            XTR_c = X[:, L == c] # Takes only samples from class c
            
            # Evaluate class means vector
            mu_c = computeMean(XTR_c)
            self.means[c] = mu_c
            
            # Evaluate class covariance matrix (diagonal)
            C_c = computeCovariance(XTR_c)
            C_diag = C_c * np.identity(C_c.shape[1])
            self.covariances[c] = C_diag

    def predict_binary(self, X, t=0):
        return super().predict_binary(X, t)

def compute_llr_for_classification(X, mu0, mu1, C0, C1):
    """
    Function to compute the log-likelihood ratios (LLR) for a set of samples X.

    Parameters
    ----------
    X : (numpy.ndarray)
        Sample feature matrix of shape (n_features, n_samples).
    mu0 : (numpy.ndarray)
        Mean vector of class 0 of shape (n_features, 1).
    mu1 : (numpy.ndarray)
        Mean vector of class 1 of shape (n_features, 1).
    C0 : (numpy.ndarray)
        Covariance matrix of class 0 of shape (n_features, n_features).
    C1 : (numpy.ndarray)
        Covariance matrix of class 1 of shape (n_features, n_features).

    Returns
    -------
    LLRs : (numpy.ndarray)
        Log-likelihood ratios of the samples in X.

    """
    return logpdf_GAU_ND(X, mu1, C1) - logpdf_GAU_ND(X, mu0, C0)

def compute_predictions_with_llr(llr, size, t):
    """
    Function to compute predictions from log-likelihood ratios.
    Compa

    Parameters
    ----------
    llr : (numpy.ndarray)
        Log-likelihood ratios of shape (n_samples,).
    size : (int)
        Number of samples.
    t : (float)
        Decision threshold.

    Returns
    -------
    PVAL : (numpy.ndarray)
        Predictions of shape (n_samples,).

    """
    PVAL = np.zeros(size, np.int32)
    PVAL[llr < t] = 0
    PVAL[llr >= t] = 1
    
    return PVAL

# TODO: Move in evaluation_metrics file
def compute_error_rate(PVAL, LVAL):
    n_correct_predictions = np.array([PVAL == LVAL]).sum()
    acc = n_correct_predictions / PVAL.shape[0]
    return 1 - acc