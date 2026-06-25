# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy
from src.utils import computeCovariance

class PrincipalComponentAnalysis:
    """
    Class for Principal Component Analysis.

    Attributes
    ----------
    P : (numpy.ndarray)
        Matrix of the m leading eigenvectors.

    Methods
    -------
    train(self, D, m)
        Train PCA model by computing the matrix P of the m leading eigenvectors.
    apply(self, D)
        Apply PCA to the data by projecting it onto the subspace spanned by the leading eigenvectors.

    """
    def __init__(self):
        self.P = None # Matrix of the m leading eigenvectors.

    def train(self, D, m):
        """
        Train PCA model by computing the matrix P of the m leading eigenvectors.

        Parameters
        ----------
        D : (numpy.ndarray)
            Training Features matrix of shape (n_samples, n_features).
        m : (int)
            Number of dimensions to reduce to.

        Returns
        -------
        None.

        """
        # Compute the covariance matrix
        C = computeCovariance(D)[0]
        # Extract eigenvalues and eigenvectors by diagonalizing C
        s, U = np.linalg.eigh(C)
        # Take the m leading eigenvectors
        self.P = U[:, ::-1][:, 0:m]

    def apply(self, D):
        """
        Apply PCA to the data by projecting it onto the subspace spanned by the leading eigenvectors.

        Parameters
        ----------
        D : (numpy.ndarray)
            Features matrix of shape (n_samples, n_features).

        Returns
        -------
        (numpy.ndarray)
            Projected features matrix of shape (n_samples, m).

        """
        return D @ self.P

class LinearDiscriminantAnalysis:
    """
    Class for Linear Discriminant Analysis.

    Attributes
    ----------
    W : (numpy.ndarray)
        Matrix of the m leading discriminant eigenvectors.

    Methods
    -------
    train(self, D, L, m)
        Train LDA model by computing the matrix W of the m leading discriminant eigenvectors.
    apply(self, D)
        Apply LDA to the data by projecting it onto the subspace spanned by the leading discriminant eigenvectors.

    """
    def __init__(self):
        self.W = None # Matrix of the m leading discriminant eigenvectors.

    def train(self, D, L, m):
        """
        Train LDA model by computing the matrix W of the m leading discriminant eigenvectors.

        Parameters
        ----------
        D : (numpy.ndarray)
            Training Features matrix of shape (n_samples, n_features).
        L : (numpy.ndarray)
            Labels of the training features.
        m : (int)
            Number of dimensions to reduce to.

        Returns
        -------
        None.

        """
        # Separate data for each class
        D1 = D[:, L == 0] # Columns corresponding to the class 0
        D2 = D[:, L == 1] # Columns corresponding to the class 1

        # Means
        mu = D.mean(1).reshape((D.shape[0], 1))
        mu1 = D1.mean(1).reshape((D1.shape[0], 1))
        mu2 = D2.mean(1).reshape((D2.shape[0], 1))

        # Between class covariance matrix
        S_B = ((mu1-mu)@(mu1-mu).T*D1.shape[1]+\
               (mu2-mu)@(mu2-mu).T*D2.shape[1])/float(D.shape[1])

        # Within class covariance matrix
        C1 = computeCovariance(D1)
        C2 = computeCovariance(D2)
        S_W = ((C1*D1.shape[1])+(C2*D2.shape[1]))/float(D.shape[1])
        
        # Solve generalized eigenvalue problem to extract m leading discriminant eigenvectors
        s, U = scipy.linalg.eigh(S_B, S_W)
        # Take the m leading eigenvectors
        self.W = U[:, ::-1][:, 0:m]

    def apply(self, D):
        """
        Apply LDA to the data by projecting it onto the subspace spanned by the leading discriminant eigenvectors.

        Parameters
        ----------
        D : (numpy.ndarray)
            Features matrix of shape (n_samples, n_features).

        Returns
        -------
        (numpy.ndarray)
            Projected features matrix of shape (n_samples, m).

        """
        return D @ self.W
