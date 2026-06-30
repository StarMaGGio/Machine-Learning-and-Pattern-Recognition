# pyrefly: ignore [missing-import]
import numpy as np
from src.utils import vcol, vrow
# pyrefly: ignore [missing-import]
import scipy.optimize as op
from src.bayes_decisions_model import compute_predictions_with_llr

class LogisticRegression():
    def __init__(self):
        self.w = None # Weights vector
        self.b = None # Bias
        self.pEmp = None # Empirical Prior of the training set

    def train(self, DTR, LTR, lamb):
        """
        Function to train the model using logistic regression
        Uses the fmin_l_bfgs_b function to find the optimal weights vector and bias
        that minimize the regularized logistic loss function
        The loss function is defined as:
        J(w, b) = (1/m) * sum(log(1 + exp(-ZTR * (w.T * DTR + b)))) + (lambda/2) * ||w||^2

        Parameters
        ----------
        DTR : (numpy.ndarray)
            Training Features matrix of shape (n_features, n_samples).
        LTR : (numpy.ndarray)
            Training Labels vector of shape (n_samples,).
        lamb : (float)
            Regularization parameter.

        Returns
        -------
        self.w : (numpy.ndarray)
            Weights vector of shape (n_features,).
        self.b : (float)
            Bias.
        """
        self.pEmp = (LTR == 1).sum() / LTR.size # Empirical Prior of the training set -> for Posterior Compensation
        ZTR = LTR * 2.0 - 1.0
    
        # Objective function
        def logreg_obj(v):
            w, b = v[:-1], v[-1] # Split the vector into weight vector and bias
            S = np.dot(vcol(w).T, DTR).ravel() + b # Compute the decision score
            
            loss = np.logaddexp(0, -ZTR * S) # Logistic Loss Function (minimize errors)
            
            # Compute Gradient
            G = -ZTR / (1.0 + np.exp(ZTR * S)) # partial derivative of the logistic loss with respect to S
            Gw = lamb*w.ravel() + (vrow(G)*DTR).mean(1) # Regularized Gradient w.r.t w
            Gb = G.mean() # Regularized Gradient w.r.t b
            
            # Return Regularized Loss Function and Gradient
            return loss.mean() + lamb / 2 * np.linalg.norm(w)**2, np.hstack([Gw, np.array(Gb)])
        
        # Compiute optimal weights vector and bias
        vf = op.fmin_l_bfgs_b(func=logreg_obj, x0=np.zeros(DTR.shape[0]+1))[0]
        print("Regularized Logistic Regression - Lambda = %.4f - Final objective value of J*(w, b) = %.4f" % (lamb, logreg_obj(vf)[0]))

        # Set optimal weights vector and bias
        self.w = vf[: -1]
        self.b = vf[-1]

    def get_log_likelihood_ratios(self, X):
        """
        Function to compute the value of per-class log-densities for a set of samples X.

        Parameters
        ----------
        X : (numpy.ndarray)
            Sample feature matrix of shape (n_features, n_samples).

        Returns
        -------
        LLRs : (numpy.ndarray)
            Class-conditional log-densities matrix of shape (1, n_samples).

        """
        # Compute validation scores
        sVal = np.dot(self.w.T, X) + self.b

        # Posterior Compensation
        LLRs = sVal - np.log(self.pEmp / (1 - self.pEmp)) # Compute LLR-like scores

        return LLRs

    def predict_binary(self, X):
        """
        Predict the binary class labels for a given feature matrix X

        Parameters
        ----------
        X : (numpy.ndarray)
            Feature matrix of shape (n_features, n_samples).

        Returns
        -------
        PVAL : (numpy.ndarray)
            Predicted labels vector of shape (n_samples,).
        """

        # Compute validation scores
        sVal = np.dot(self.w.T, X) + self.b

        # Posterior Compensation
        LLRs = sVal - np.log(self.pEmp / (1 - self.pEmp)) # Compute LLR-like scores

        # Compute optimal decisions
        PVAL = compute_predictions_with_llr(LLRs, t=0)
        
class WeightedLogisticRegression(LogisticRegression):

    def __init__(self):
        super().__init__(self)
        self.wTrue = None
        self.wFalse = None

    def train(self, DTR, LTR, lamb):
        """
        Train the model using weighted logistic regression

        Parameters
        ----------
        DTR : (numpy.ndarray)
            Training Features matrix of shape (n_features, n_samples).
        LTR : (numpy.ndarray)
            Training Labels vector of shape (n_samples,).
        lamb : (float)
            Regularization parameter.

        Returns
        -------
        None.
        """
        self.pEmp = (LTR == 1).sum() / LTR.size # Empirical Prior of the training set -> for Posterior Compensation
        ZTR = LTR * 2.0 - 1.0
    
        # Compute class weights for prior compensation
        self.wTrue = self.pEmp / (ZTR > 0).sum()
        self.wFalse = (1 - self.pEmp) / (ZTR < 0).sum()
        
        # Define objective function
        def logreg_obj(v):
            w, b = v[:-1], v[-1]
            S = np.dot(vcol(w).T, DTR).ravel() + b
            
            # Compute loss function
            loss = np.logaddexp(0, -ZTR * S)
            # Apply the class weights to the loss function (Prior compensation)
            loss[ZTR>0] *= self.wTrue
            loss[ZTR<0] *= self.wFalse
            
            # Compute Gradient
            G = -ZTR / (1.0 + np.exp(ZTR * S))
            # Apply the class weights to the gradient computation
            G[ZTR>0] *= self.wTrue
            G[ZTR<0] *= self.wFalse
            
            Gw = lamb*w.ravel() + (vrow(G)*DTR).sum(1)
            Gb = G.sum()
            
            # Return Regularized Weighted Loss Function and Gradient
            return loss.sum() + lamb / 2 * np.linalg.norm(w)**2, np.hstack([Gw, np.array(Gb)])
        
        # Compute optimal weights vector and bias
        vf = op.fmin_l_bfgs_b(func=logreg_obj, x0=np.zeros(DTR.shape[0]+1))[0]
        print("Weighted Logistic Regression - Lambda = %.4f - Final objective value of J*(w, b) = %.4f" % (lamb, logreg_obj(vf)[0]))

        # Set optimal weights vector and bias
        self.w = vf[: -1]
        self.b = vf[-1]

    def get_log_likelihood_ratios(self, X):
        return super().get_log_likelihood_ratios(X)

    def predict_binary(self, X):
        return super().predict_binary(X)