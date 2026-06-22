# pyrefly: ignore [missing-import]
import numpy as np

def logpdf_GAU_ND_singleSample(x, mu, C):
    """
    Function to compute the logarithm of a Gaussian distribution density for a sample x.

    Parameters
    ----------
    x : (numpy.ndarray)
        Sample feature vector of shape (n_features, 1).
    mu : (numpy.ndarray)
        Distribution mean vector of shape (n_features, 1).
    C : (numpy.ndarray)
        Distribution covariance matrix of shape (n_features, n_features).

    Returns
    -------
    log_density : (float)
        Log-density of the sample x.

    """

    C_inv = np.linalg.inv(C) # Precision matrix
    M = x.shape[0] # Dimension of the feature vector (num of features)
    
    # Compute: log N(x|mu,C) = -(M/2)*log(2pi) - (1/2)*log|C| - (1/2)*C^-1(x-mu)^2
    return -0.5*M*np.log(2*np.pi) - 0.5*np.linalg.slogdet(C)[1] - 0.5*((x-mu).T @ C_inv @ (x-mu)).ravel()

def logpdf_GAU_ND(X, mu, C):
    """
    Function to compute the logarithm of a Gaussian distribution density for a set of samples X.

    Parameters
    ----------
    X : (numpy.ndarray)
        Sample feature matrix of shape (n_features, n_samples).
    mu : (numpy.ndarray)
        Distribution mean vector of shape (n_features, 1).
    C : (numpy.ndarray)
        Distribution covariance matrix of shape (n_features, n_features).

    Returns
    -------
    log_densities : (numpy.ndarray)
        Log-densities of the samples in X.

    """
    num_samples = X.shape[1]
    log_densities = [logpdf_GAU_ND_singleSample(X[:, i:i+1], mu, C) for i in range(num_samples)]
    return np.array(log_densities).ravel()