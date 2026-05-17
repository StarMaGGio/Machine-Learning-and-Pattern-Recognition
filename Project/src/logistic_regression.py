import numpy as np
from src.utils import vcol, vrow
import scipy.optimize as op

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
    print("Log-reg - Lambda = %.4f - J*(w, b) = %.4f" % (l, logreg_obj(vf)[0]))
    return vf[: -1], vf[-1]

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
        

