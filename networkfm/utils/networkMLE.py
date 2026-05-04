"""
Created on Thu Sep 25 2025

Authors: Zizhong Yan
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import sys
import time
import numpy as np
import scipy as sp
import logging
try: 
    import torch
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
except Exception as e:
    logging.warning(e)
    logging.warning("To use networkfm command, please install PyTorch. (See pytorch.org)")

from scipy import optimize
#----------------------------------------------------------
# Fit the model and run the regression
#----------------------------------------------------------
def networkFits(G, X=None, Z=None, directed=True, mutual=True,
                  bc_method="likelihood", drop_separation=False, algorithm="JML", 
                  sv=None, silent=False,ape_compute=False,seps=0,
                  dummyIndicatorX=None,dummyIndicatorZ=None):
    #----------------------------------------------------------
    # Initializations
    #----------------------------------------------------------
    # [> Global variables in this function <]
    global FE_theta 
    global parameterUpdate
    # [> Epsilon of numerical Python floating number <]
    FLOAT_EPS   = np.finfo(float).eps
    FLOAT_EPStc = torch.finfo(float).eps
    # [> In/out-degree sequence array (excl. the node N) <]
    ds_out_d   = np.sum(G,axis=1)[:-1]
    ds_in_b    = np.sum(G,axis=0)[:-1]
    # [> Dimensions of data & variables <]
    N = np.shape(G)[0]
    if X is not None: kx = np.shape(X)[2]
    if Z is not None: kz = np.shape(Z)[2]
    # [> Starting values of parameters in the optimization <]
    # If not provided, starting values are all zeroes by default
    if directed==True and mutual==True:
        numpara = kx+kz
        initials    = np.zeros((numpara+N+N-2))-0.0; initials[numpara:]=0
    if directed==True and mutual==False:
        numpara = kx
        initials    = np.zeros((numpara+N+N-2))-0.0; initials[numpara:]=0
    if directed==False:
        numpara = kz
        initials    = np.zeros((numpara+N-1))-0.0; initials[numpara:]=0
        X=Z
    # With provided sv (starting values)
    if sv is not None: 
        if np.size(sv)!=np.size(initials):
            sys.exit("Error: the number of starting values does not match the number of parameters.")
        else:
            initials=sv.reshape(-1,1)
    #----------------------------------------------------------
    # Objective function
    #----------------------------------------------------------
    def negLogLike(parameter):
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute log-likelihood
            logll=np.triu(Xb*G, 1).sum()-np.triu(np.log(1+np.exp(Xb)),1).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L  = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll + np.log(np.sum(L,axis=1)[:-1]).sum()/2
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            expXb_mat = (1+np.exp(Xb_u)+np.exp(Xb_u.T)+ np.exp(Xb_u+Xb_u.T+Zb_m))
            np.fill_diagonal(expXb_mat, 1)
            logfenmu=np.log(np.sqrt(expXb_mat)).sum()
            logll=np.sum((Xb_u + Zb_m*G.T/2)*G) -logfenmu
            #compute penalty if specified
            if bc_method=="likelihood":
                P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
                np.fill_diagonal(P, 0)
                Lp   = P*(1-P)
                logll = logll + (np.log(np.sum(Lp,axis=1)[:-1]).sum()+np.log(np.sum(Lp,axis=0)[:-1]).sum())/2
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            q=2*G-1
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            qL = sp.stats.logistic._cdf(q*Xb)
            logll=np.log(np.clip(qL,FLOAT_EPS,1)).sum()- np.log(np.clip(np.diag(qL),FLOAT_EPS,1)).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll + (np.log(np.sum(L,axis=1)[:-1]).sum()+np.log(np.sum(L,axis=0)[:-1]).sum())/2
        return -(logll)
    def negLogLike_tc(parameter_tc):
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=torch.tile((torch.vstack((parameter_tc[kz:],torch.tensor([0])))).reshape(1,N),(N,1)).reshape(N,N)
            Xb = (X_tc @ parameter_tc[:kz])[:,:,0] + alphamat + alphamat.T
            #compute log-likelihood
            logll=torch.triu(Xb*G_tc, 1).sum()-torch.triu(torch.log(1+torch.exp(Xb)),1).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                Pij = torch.sigmoid(Xb)
                PijPij = Pij*(1-Pij)
                PijPij.fill_diagonal_(0)
                penalty_p1 = torch.log(torch.sum(PijPij,axis=0))
                penalty_p1[-1] = 0
                penalty    = 0.5*torch.sum(penalty_p1)
                logll = logll + penalty
        if directed==True and mutual==True:  \
            #organize parameters and compute linear pred
            alphamat=torch.tile((torch.vstack((parameter_tc[kx+kz:kx+kz+N-1],torch.tensor([0])))).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=torch.tile((torch.vstack((parameter_tc[kx+kz+N-1:],     torch.tensor([0])))).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m  = (Z_tc @ parameter_tc[kx:kx+kz])[:,:,0]
            Xb_u = (X_tc @ parameter_tc[:kx])[:,:,0] + alphamat + gammamat.T
            Xb   = Xb_u + Zb_m*G_tc.T/2 
            #compute log-likelihood
            logfenzi=torch.sum(Xb*G_tc) 
            expXb_mat = (1+torch.exp(Xb_u)+torch.exp(Xb_u.T)+ torch.exp(Xb_u+Xb_u.T+ Zb_m))
            expXb_mat.fill_diagonal_(1)
            logfenmu=torch.log(torch.sqrt(expXb_mat)).sum()
            logll=logfenzi-logfenmu
            #compute penalty if specified
            if bc_method=="likelihood":
                Pij_p1 = torch.exp(Xb_u)
                Pij_p2 = torch.exp(Xb_u.T)
                Pij_p3 = torch.exp(Xb_u+Xb_u.T+ Zb_m)
                Pij_fenzi = Pij_p1 + Pij_p3
                Pij_fenmu = 1+ Pij_p1 + Pij_p2 + Pij_p3
                Pij = Pij_fenzi / Pij_fenmu
                PijPij = Pij*(1-Pij)
                PijPij.fill_diagonal_(0)
                penalty_p1 = torch.log(torch.sum(PijPij,axis=1))
                penalty_p2 = torch.log(torch.sum(PijPij,axis=0))
                penalty_p1[-1] = 0
                penalty_p2[-1] = 0
                penalty    = 0.5*( torch.sum(penalty_p1) + torch.sum(penalty_p2) )
                logll = logll + penalty
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            q=2*G_tc-1
            alphamat=torch.tile((torch.vstack((parameter_tc[kx:kx+N-1],torch.tensor([0])))).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=torch.tile((torch.vstack((parameter_tc[kx+N-1:]  ,torch.tensor([0])))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X_tc @ parameter_tc[:kx])[:,:,0]+ alphamat + gammamat.T
            #compute log-likelihood
            qL = torch.sigmoid(q*Xb)
            logll=torch.log(torch.clip(qL,FLOAT_EPStc,1)).sum()- torch.log(torch.clip(torch.diag(qL),FLOAT_EPStc,1)).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                PijPij = torch.exp(Xb) / ((1+torch.exp(Xb))**2)
                PijPij.fill_diagonal_(0)
                penalty_p1 = torch.log(torch.sum(PijPij,axis=1))
                penalty_p2 = torch.log(torch.sum(PijPij,axis=0))
                penalty_p1[-1] = 0
                penalty_p2[-1] = 0
                penalty    = 0.5*( torch.sum(penalty_p1) + torch.sum(penalty_p2) )
                logll = logll + penalty
        return -(logll)
    # Objective functions - optimization with concentration scheme 
    def negLogLikeConcen(parameterSlice):
        global parameterUpdate
        parameterUpdate[newindices[start_i:end_i]] = parameterSlice
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameterUpdate[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameterUpdate[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute log-likelihood
            logll=np.triu(Xb*G, 1).sum()-np.triu(np.log(1+np.exp(Xb)),1).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L  = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll+np.log(np.sum(L,axis=1)[:-1]).sum()/2
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameterUpdate[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameterUpdate[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            expXb_mat = (1+np.exp(Xb_u)+np.exp(Xb_u.T)+ np.exp(Xb_u+Xb_u.T+Zb_m))
            np.fill_diagonal(expXb_mat, 1)
            logfenmu=np.log(np.sqrt(expXb_mat)).sum()
            logll=np.sum((Xb_u + Zb_m*G.T/2)*G) -logfenmu
            #compute penalty if specified
            if bc_method=="likelihood":
                P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
                np.fill_diagonal(P, 0)
                Lp   = P*(1-P)
                logll = logll+(np.log(np.sum(Lp,axis=1)[:-1]).sum()+np.log(np.sum(Lp,axis=0)[:-1]).sum())/2
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            q=2*G-1
            alphamat=np.tile(np.append(parameterUpdate[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            qL = sp.stats.logistic._cdf(q*Xb)
            logll=np.log(np.clip(qL,FLOAT_EPS,1)).sum()- np.log(np.clip(np.diag(qL),FLOAT_EPS,1)).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll+(np.log(np.sum(L,axis=1)[:-1]).sum()+np.log(np.sum(L,axis=0)[:-1]).sum())/2
        return -(logll)
    # Objective functions - fixed point iterations
    def negLogLike_thetaFP(parameter):
        # get FE parameters updated globally during fixed point calculations below. 
        global FE_theta 
        # fixed point calculations of FE parameters
        FE_theta  = sp.optimize.fixed_point(FixedPointFE, FE_theta,
                                            args=(parameter,),
                                            xtol=1e-6, maxiter=6000, method='iteration')
        # objective function for thetas
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(FE_theta,np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter.reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute log-likelihood
            logll=np.triu(Xb*G, 1).sum()-np.triu(np.log(1+np.exp(Xb)),1).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L  = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll + np.log(np.sum(L,axis=1)[:-1]).sum()/2
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(FE_theta[:N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(FE_theta[N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            expXb_mat = (1+np.exp(Xb_u)+np.exp(Xb_u.T)+ np.exp(Xb_u+Xb_u.T+Zb_m))
            np.fill_diagonal(expXb_mat, 1)
            logfenmu=np.log(np.sqrt(expXb_mat)).sum()
            logll=np.sum((Xb_u + Zb_m*G.T/2)*G) -logfenmu
            #compute penalty if specified
            if bc_method=="likelihood":
                P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
                np.fill_diagonal(P, 0)
                Lp   = P*(1-P)
                logll = logll + (np.log(np.sum(Lp,axis=1)[:-1]).sum()+np.log(np.sum(Lp,axis=0)[:-1]).sum())/2
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            q=2*G-1
            alphamat=np.tile(np.append(FE_theta[:N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(FE_theta[N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter.reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute log-likelihood
            qL = sp.stats.logistic._cdf(q*Xb)
            logll=np.log(np.clip(qL,FLOAT_EPS,1)).sum()- np.log(np.clip(np.diag(qL),FLOAT_EPS,1)).sum()
            #compute penalty if specified
            if bc_method=="likelihood":
                L = sp.stats.logistic._pdf(Xb)
                np.fill_diagonal(L, 0)
                logll = logll + (np.log(np.sum(L,axis=1)[:-1]).sum()+np.log(np.sum(L,axis=0)[:-1]).sum())/2
        return -(logll)
    #----------------------------------------------------------
    # Fixed point iterator for estimating FE parameters
    #----------------------------------------------------------
    def FixedPointFE(fe_para, theta):
        '''
        This function returns one iterate of the fixed point iteration procedure
        as in Chatterjee, Diaconis and Sly (2011, Annals of Applied Probability)
        and Graham (2017, Econometrica), and further adapted here.

        INPUT:
            fe_para: column vector of fixed effects parameters
            theta:   column vector, theta enters as an argument and is fixed
        OUTPUT:
            1d array of fixed point iterate update
        '''
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(fe_para,np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ theta.reshape(-1,1)).reshape(N,N) # + alphamat + alphamat.T
            #computations
            firstTerm  = ds_out_d
            secondTermij = (np.exp(Xb+alphamat.T)/( 1+np.exp(Xb+ alphamat+alphamat.T)  ))
            np.fill_diagonal(secondTermij, 0)
            secondTerm = (secondTermij.sum(1))[:-1]
            #compute derivative of penalty if specified
            if bc_method=="likelihood":
                firstTerm = firstTerm+scoreBiasHelper(np.append(theta,fe_para))[kz:]
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(fe_para[:N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(fe_para[N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ theta[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ theta[:kx].reshape(-1,1)).reshape(N,N) #+ alphamat + gammamat.T
            #computations
            firstTerm        = np.append(ds_out_d,ds_in_b)
            fenmu = (1+ np.exp(Xb_u+ alphamat + gammamat.T) + np.exp(Xb_u+ alphamat + gammamat.T).T + np.exp(Xb_u+ alphamat + gammamat.T+Xb_u.T+alphamat.T + gammamat+Zb_m))
            secondTerm_ij1 = (np.exp(Xb_u+gammamat.T) + np.exp(Xb_u+gammamat.T+Xb_u.T+alphamat.T+gammamat+Zb_m))/fenmu
            secondTerm_ij2 = (np.exp(Xb_u+alphamat  ) + np.exp(Xb_u+alphamat  +Xb_u.T+alphamat.T+gammamat+Zb_m))/fenmu
            np.fill_diagonal(secondTerm_ij1, 0)
            np.fill_diagonal(secondTerm_ij2, 0)
            secondTerm = np.append(((secondTerm_ij1).sum(1))[:-1],((secondTerm_ij2).sum(0))[:-1])
            #compute derivative of penalty if specified
            if bc_method=="likelihood":
                firstTerm = firstTerm+scoreBiasHelper(np.append(theta,fe_para))[kx+kz:]
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(fe_para[:N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(fe_para[N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ theta.reshape(-1,1)).reshape(N,N) #+ alphamat + gammamat.T
            #computations
            firstTerm        = np.append(ds_out_d,ds_in_b)
            fenmu = 1+np.exp(Xb+ alphamat + gammamat.T)
            secondTerm_ij1 = np.exp(Xb + gammamat.T)/fenmu
            secondTerm_ij2 = np.exp(Xb + alphamat  )/fenmu
            np.fill_diagonal(secondTerm_ij1, 0)
            np.fill_diagonal(secondTerm_ij2, 0)
            secondTerm = np.append(((secondTerm_ij1).sum(1))[:-1],((secondTerm_ij2).sum(0))[:-1])
            #compute derivative of penalty if specified
            if bc_method=="likelihood":
                firstTerm = firstTerm+scoreBiasHelper(np.append(theta,fe_para))[kx:]
        return np.log(firstTerm) - np.log(secondTerm) 
    #----------------------------------------------------------
    # Negative Hessian matrix and negative Jacobian/score
    #----------------------------------------------------------
    def negHessAll(parameter):
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute hessian
            L  = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            negHess = np.zeros((kz+N-1,kz+N-1))
            negHess[kz:,kz:] = L[:-1,:-1] + np.diag(np.sum(L,axis=1)[:-1]) #lambda part
            negHess[:kz,:kz] = ((L.reshape(-1,1)*X.reshape(-1,kz)).T@X.reshape(-1,kz))/2 #theta part
            negHess[kz:,:kz] = (L.reshape(-1,1)*X.reshape(-1,kz)).reshape(N,N,kz).sum(axis=0)[:-1,:]
            negHess[:kz,kz:] = negHess[kz:,:kz].T
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            np.fill_diagonal(P, 0)
            Lp   = P*(1-P)
            #lambda part: diagonals
            negHess_lambda = np.zeros((N+N-2,N+N-2))
            negHess_lambda[:N-1,:N-1] = np.diag(np.sum(Lp,axis=1)[:-1])
            negHess_lambda[N-1:,N-1:] = np.diag(np.sum(Lp,axis=0)[:-1])
            #lambda part: off-diagonals
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            P01 = np.exp(Xb_u.T) / fenmu
            P00 = 1 / fenmu
            np.fill_diagonal(P11, 0)
            np.fill_diagonal(P01, 0)
            np.fill_diagonal(P00, 0)
            Lp11 = P11 - P*P.T
            negHess_lambda[:N-1,N-1:] = Lp[:-1,:-1] + np.diag(np.sum(Lp11,axis=1)[:-1])
            negHess_lambda[N-1:,:N-1] = negHess_lambda[:N-1,N-1:].T
            negHess_lambda[:N-1,:N-1] = negHess_lambda[:N-1,:N-1] + Lp11[:-1,:-1]
            negHess_lambda[N-1:,N-1:] = negHess_lambda[N-1:,N-1:] + Lp11[:-1,:-1]
            #beta and delta parts
            transX = X.transpose(1,0,2)
            negHess_beta  = ((Lp.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx))/2 + (((Lp.T).reshape(-1,1)*transX.reshape(-1,kx)).T@transX.reshape(-1,kx))/2 + (Lp11.reshape(-1,1)*X.reshape(-1,kx)).T@transX.reshape(-1,kx)
            negHess_delta = (((P11*(1-P11)).reshape(-1,1)*Z.reshape(-1,kz)).T@Z.reshape(-1,kz))/2
            negHess_betadelta = (((P11*(1-P)).reshape(-1,1)*X.reshape(-1,kx)).T+((P11*(1-P.T)).reshape(-1,1)*transX.reshape(-1,kx)).T)@Z.reshape(-1,kz)/2
            #cross parts
            D1 = (Lp.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx) +(Lp11.reshape(-1,1)*transX.reshape(-1,kx)).reshape(N,N,kx)
            negHess_betalambda  = np.vstack(((D1.sum(axis=1))[:-1,:], (D1.sum(axis=0))[:-1,:]))
            D2 = ((P11*(P00+P01)).reshape(-1,1)*Z.reshape(-1,kz)).reshape(N,N,kz)
            negHess_deltalambda = np.vstack(((D2.sum(axis=1))[:-1,:], (D2.sum(axis=0))[:-1,:]))
            #put into the final hessian matrix
            negHess = np.zeros((kx+kz+N+N-2,kx+kz+N+N-2))
            negHess[kx+kz:,kx+kz:]     = negHess_lambda
            negHess[:kx,:kx]           = negHess_beta
            negHess[kx:kx+kz,kx:kx+kz] = negHess_delta
            negHess[:kx,kx:kx+kz]      = negHess_betadelta
            negHess[kx:kx+kz,:kx]      = negHess[:kx,kx:kx+kz].T
            negHess[kx+kz:,:kx]        = negHess_betalambda
            negHess[:kx,kx+kz:]        = negHess[kx+kz:,:kx].T
            negHess[kx+kz:,kx:kx+kz]   = negHess_deltalambda
            negHess[kx:kx+kz,kx+kz:]   = negHess[kx+kz:,kx:kx+kz].T
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            L = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            #lambda part: diagonals
            negHess_lambda = np.zeros((N+N-2,N+N-2))
            negHess_lambda[:N-1,:N-1] = np.diag(np.sum(L,axis=1)[:-1])
            negHess_lambda[N-1:,N-1:] = np.diag(np.sum(L,axis=0)[:-1])
            #lambda part: off-diagonals
            negHess_lambda[:N-1,N-1:] = L[:-1,:-1]
            negHess_lambda[N-1:,:N-1] = L[:-1,:-1].T
            #put into the final hessian matrix
            negHess = np.zeros((kx+N+N-2,kx+N+N-2))
            negHess[kx:,kx:] = negHess_lambda
            negHess[:kx,:kx] = ((L.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx)) #theta part
            negHess[kx:,:kx] = np.vstack(((L.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx).sum(axis=1)[:-1,:],(L.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx).sum(axis=0)[:-1,:]))
            negHess[:kx,kx:] = negHess[kx:,:kx].T
        return negHess
    def negJacobianAll(parameter):
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute jacobian
            P     = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = np.triu(((G-P)*X.T),1).sum(1).sum(1)
            jacFE = ds_out_d - P.sum(1)[:-1]
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            np.fill_diagonal(P, 0)
            np.fill_diagonal(P11, 0)
            jacB  = np.append(((G-P)*X.transpose(2,0,1)).sum(1).sum(1), ((G*G.T-P11)*Z.T).sum(1).sum(1)/2)
            firstTerm = np.append(ds_out_d,ds_in_b)
            secondTerm = np.append((P).sum(1)[:-1],   (P.T).sum(1)[:-1])
            jacFE = firstTerm - secondTerm
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:].reshape(-1,1),np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute jacobian
            P = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = ((G-P)*X.T).sum(1).sum(1)
            firstTerm = np.append(ds_out_d,ds_in_b)
            secondTerm = np.append((P).sum(1)[:-1],   (P.T).sum(1)[:-1])
            jacFE = firstTerm - secondTerm
        jac = np.append(jacB, jacFE)
        if bc_method=="likelihood":
            jac = jac + scoreBiasHelper(parameter)
        return -jac
    def negHessConcen(parameterSlice):
        global parameterUpdate
        parameterUpdate[newindices[start_i:end_i]] = parameterSlice
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameterUpdate[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameterUpdate[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute hessian
            L  = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            negHess = np.zeros((kz+N-1,kz+N-1))
            negHess[kz:,kz:] = L[:-1,:-1] + np.diag(np.sum(L,axis=1)[:-1]) #lambda part
            negHess[:kz,:kz] = ((L.reshape(-1,1)*X.reshape(-1,kz)).T@X.reshape(-1,kz))/2 #theta part
            negHess[kz:,:kz] = (L.reshape(-1,1)*X.reshape(-1,kz)).reshape(N,N,kz).sum(axis=0)[:-1,:]
            negHess[:kz,kz:] = negHess[kz:,:kz].T
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameterUpdate[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameterUpdate[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            np.fill_diagonal(P, 0)
            Lp   = P*(1-P)
            #lambda part: diagonals
            negHess_lambda = np.zeros((N+N-2,N+N-2))
            negHess_lambda[:N-1,:N-1] = np.diag(np.sum(Lp,axis=1)[:-1])
            negHess_lambda[N-1:,N-1:] = np.diag(np.sum(Lp,axis=0)[:-1])
            #lambda part: off-diagonals
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            P01 = np.exp(Xb_u.T) / fenmu
            P00 = 1 / fenmu
            np.fill_diagonal(P11, 0)
            np.fill_diagonal(P01, 0)
            np.fill_diagonal(P00, 0)
            Lp11 = P11 - P*P.T
            negHess_lambda[:N-1,N-1:] = Lp[:-1,:-1] + np.diag(np.sum(Lp11,axis=1)[:-1])
            negHess_lambda[N-1:,:N-1] = negHess_lambda[:N-1,N-1:].T
            negHess_lambda[:N-1,:N-1] = negHess_lambda[:N-1,:N-1] + Lp11[:-1,:-1]
            negHess_lambda[N-1:,N-1:] = negHess_lambda[N-1:,N-1:] + Lp11[:-1,:-1]
            #beta and delta parts
            transX = X.transpose(1,0,2)
            negHess_beta  = ((Lp.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx))/2 + (((Lp.T).reshape(-1,1)*transX.reshape(-1,kx)).T@transX.reshape(-1,kx))/2 + (Lp11.reshape(-1,1)*X.reshape(-1,kx)).T@transX.reshape(-1,kx)
            negHess_delta = (((P11*(1-P11)).reshape(-1,1)*Z.reshape(-1,kz)).T@Z.reshape(-1,kz))/2
            negHess_betadelta = (((P11*(1-P)).reshape(-1,1)*X.reshape(-1,kx)).T+((P11*(1-P.T)).reshape(-1,1)*transX.reshape(-1,kx)).T)@Z.reshape(-1,kz)/2
            #cross parts
            D1 = (Lp.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx) +(Lp11.reshape(-1,1)*transX.reshape(-1,kx)).reshape(N,N,kx)
            negHess_betalambda  = np.vstack(((D1.sum(axis=1))[:-1,:], (D1.sum(axis=0))[:-1,:]))
            D2 = ((P11*(P00+P01)).reshape(-1,1)*Z.reshape(-1,kz)).reshape(N,N,kz)
            negHess_deltalambda = np.vstack(((D2.sum(axis=1))[:-1,:], (D2.sum(axis=0))[:-1,:]))
            #put into the final hessian matrix
            negHess = np.zeros((kx+kz+N+N-2,kx+kz+N+N-2))
            negHess[kx+kz:,kx+kz:]     = negHess_lambda
            negHess[:kx,:kx]           = negHess_beta
            negHess[kx:kx+kz,kx:kx+kz] = negHess_delta
            negHess[:kx,kx:kx+kz]      = negHess_betadelta
            negHess[kx:kx+kz,:kx]      = negHess[:kx,kx:kx+kz].T
            negHess[kx+kz:,:kx]        = negHess_betalambda
            negHess[:kx,kx+kz:]        = negHess[kx+kz:,:kx].T
            negHess[kx+kz:,kx:kx+kz]   = negHess_deltalambda
            negHess[kx:kx+kz,kx+kz:]   = negHess[kx+kz:,kx:kx+kz].T
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameterUpdate[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            L = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            #lambda part: diagonals
            negHess_lambda = np.zeros((N+N-2,N+N-2))
            negHess_lambda[:N-1,:N-1] = np.diag(np.sum(L,axis=1)[:-1])
            negHess_lambda[N-1:,N-1:] = np.diag(np.sum(L,axis=0)[:-1])
            #lambda part: off-diagonals
            negHess_lambda[:N-1,N-1:] = L[:-1,:-1]
            negHess_lambda[N-1:,:N-1] = L[:-1,:-1].T
            #put into the final hessian matrix
            negHess = np.zeros((kx+N+N-2,kx+N+N-2))
            negHess[kx:,kx:] = negHess_lambda
            negHess[:kx,:kx] = ((L.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx)) #theta part
            negHess[kx:,:kx] = np.vstack(((L.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx).sum(axis=1)[:-1,:],(L.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx).sum(axis=0)[:-1,:]))
            negHess[:kx,kx:] = negHess[kx:,:kx].T
        return negHess[newindices[start_i:end_i],:][:,newindices[start_i:end_i]]
    def negJacobianConcen(parameterSlice):
        global parameterUpdate
        parameterUpdate[newindices[start_i:end_i]] = parameterSlice
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameterUpdate[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameterUpdate[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute jacobian
            P     = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = np.triu(((G-P)*X.T),1).sum(1).sum(1)
            jacFE = ds_out_d - P.sum(1)[:-1]
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameterUpdate[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameterUpdate[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            np.fill_diagonal(P, 0)
            np.fill_diagonal(P11, 0)
            jacB  = np.append(((G-P)*X.transpose(2,0,1)).sum(1).sum(1), ((G*G.T-P11)*Z.T).sum(1).sum(1)/2)
            firstTerm = np.append(ds_out_d,ds_in_b)
            secondTerm = np.append((P).sum(1)[:-1],   (P.T).sum(1)[:-1])
            jacFE = firstTerm - secondTerm
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameterUpdate[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameterUpdate[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameterUpdate[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute jacobian
            P = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = ((G-P)*X.T).sum(1).sum(1)
            firstTerm = np.append(ds_out_d,ds_in_b)
            secondTerm = np.append((P).sum(1)[:-1],   (P.T).sum(1)[:-1])
            jacFE = firstTerm - secondTerm
        jac = np.append(jacB, jacFE)
        if bc_method=="likelihood":
            jac = jac + scoreBiasHelper(parameterUpdate)
        return -jac[newindices[start_i:end_i]]
    #----------------------------------------------------------
    # Hessian and Jacobian for theta given FE parameters fixed
    #----------------------------------------------------------
    def negHessTheta(parameter):
        parameter = np.append(parameter,FE_theta) #FE_theta is a global parameter here
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute hessian
            L  = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            negHB = ((L.reshape(-1,1)*X.reshape(-1,kz)).T@X.reshape(-1,kz))/2 #theta part
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            np.fill_diagonal(P, 0)
            Lp   = P*(1-P)
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            np.fill_diagonal(P11, 0)
            Lp11 = P11 - P*P.T
            transX = X.transpose(1,0,2)
            negHB = np.zeros((kx+kz,kx+kz))
            negHB[:kx,:kx]           = ((Lp.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx))/2 + (((Lp.T).reshape(-1,1)*transX.reshape(-1,kx)).T@transX.reshape(-1,kx))/2 + (Lp11.reshape(-1,1)*X.reshape(-1,kx)).T@transX.reshape(-1,kx)
            negHB[kx:kx+kz,kx:kx+kz] = (((P11*(1-P11)).reshape(-1,1)*Z.reshape(-1,kz)).T@Z.reshape(-1,kz))/2
            negHB[:kx,kx:kx+kz]      = (((P11*(1-P)).reshape(-1,1)*X.reshape(-1,kx)).T+((P11*(1-P.T)).reshape(-1,1)*transX.reshape(-1,kx)).T)@Z.reshape(-1,kz)/2
            negHB[kx:kx+kz,:kx]      = negHB[:kx,kx:kx+kz].T
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute hessian
            L = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(L, 0)
            negHB = ((L.reshape(-1,1)*X.reshape(-1,kx)).T@X.reshape(-1,kx)) #theta part
        return negHB
    def negJacobianTheta(parameter):
        parameter = np.append(parameter,FE_theta) #FE_theta is a global parameter here
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute jacobian
            P     = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = np.triu(((G-P)*X.T),1).sum(1).sum(1)
            if bc_method=="likelihood":
                jacB = jacB+scoreBiasHelper(parameter)[:kz]
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            np.fill_diagonal(P, 0)
            np.fill_diagonal(P11, 0)
            jacB  = np.append(((G-P)*X.transpose(2,0,1)).sum(1).sum(1), ((G*G.T-P11)*Z.T).sum(1).sum(1)/2)
            if bc_method=="likelihood":
                jacB = jacB+scoreBiasHelper(parameter)[:kx+kz]
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute jacobian
            P = sp.stats.logistic._cdf(Xb)  
            np.fill_diagonal(P, 0)
            jacB  = ((G-P)*X.T).sum(1).sum(1)
            if bc_method=="likelihood":
                jacB = jacB+scoreBiasHelper(parameter)[:kx]
        return -jacB
    #----------------------------------------------------------
    # Functions for APE
    #----------------------------------------------------------
    # Calculate the APE
    def apeCalc(parameter):
        # Compute APE
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #compute ape
            pdf = sp.stats.logistic._pdf(Xb) 
            cdf = sp.stats.logistic._cdf(Xb)
            matone = np.ones((N,N))
            np.fill_diagonal(pdf, 0)
            np.fill_diagonal(cdf, 0)
            np.fill_diagonal(matone, 0)
            ape_mat =(parameter[:kz].reshape(-1)*pdf.reshape(-1,1)).reshape(N,N,kz)
            for kkk in range(kz):
                if dummyIndicatorZ[kkk]==True: 
                    X0B = Xb - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    ape_mat[:,:,kkk] = (sp.stats.logistic._cdf(X1B)-sp.stats.logistic._cdf(X0B)).reshape(N,N)
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute ape
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            pdf_direct =P*(1-P)
            pdf_mutual =P11*(1-P11)
            matone = np.ones((N,N))
            np.fill_diagonal(pdf_direct, 0)
            np.fill_diagonal(pdf_mutual, 0)
            np.fill_diagonal(matone, 0)
            ape_directmat =(parameter[:kx].reshape(-1)*pdf_direct.reshape(-1,1)).reshape(N,N,kx)
            ape_mutualmat =(parameter[kx:kx+kz].reshape(-1)*pdf_mutual.reshape(-1,1)).reshape(N,N,kz)
            for kkk in range(kx):
                if dummyIndicatorX[kkk]==True: 
                    X0B = Xb_u - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    cdfX1B = (np.exp(X1B) + np.exp(X1B+X1B.T+Zb_m)) / (1+ np.exp(X1B) + np.exp(X1B).T + np.exp(X1B+X1B.T+Zb_m))
                    cdfX0B = (np.exp(X0B) + np.exp(X0B+X0B.T+Zb_m)) / (1+ np.exp(X0B) + np.exp(X0B).T + np.exp(X0B+X0B.T+Zb_m))
                    ape_directmat[:,:,kkk] = (cdfX1B-cdfX0B).reshape(N,N)
            for kkk in range(kz):
                if dummyIndicatorZ[kkk]==True: 
                    Z0B = Zb_m - Z[:,:,kkk]*parameter[kx+kkk]
                    Z1B = Z0B + matone*parameter[kx+kkk]
                    cdfZ1B = np.exp(Xb_u+Xb_u.T+Z1B) / (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Z1B))
                    cdfZ0B = np.exp(Xb_u+Xb_u.T+Z0B) / (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Z0B))
                    ape_mutualmat[:,:,kkk] = (cdfZ1B-cdfZ0B).reshape(N,N)
            ape_mat = np.concatenate((ape_directmat,ape_mutualmat),axis=2)
        if directed==True and mutual==False:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #compute ape
            pdf = sp.stats.logistic._pdf(Xb) 
            cdf = sp.stats.logistic._cdf(Xb)
            matone = np.ones((N,N))
            np.fill_diagonal(pdf, 0)
            np.fill_diagonal(cdf, 0)
            np.fill_diagonal(matone, 0)
            ape_mat =(parameter[:kx].reshape(-1)*pdf.reshape(-1,1)).reshape(N,N,kx)
            for kkk in range(kx):
                if dummyIndicatorX[kkk]==True: 
                    X0B = Xb - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    ape_mat[:,:,kkk] = (sp.stats.logistic._cdf(X1B)-sp.stats.logistic._cdf(X0B)).reshape(N,N)
        ape = ape_mat.sum(0).sum(0)/(N*(N-1))
        return ape, ape_mat
    # APE's asymptotic standard errors
    def apeSE(parameter):
        ape,ape_mat = apeCalc(parameter)
        d_theta, d_lambda, dd_lambda = ape_helper(parameter)
        negHess = negHessAll(parameter)
        invH = np.linalg.inv(negHess[numpara:,numpara:])
        invW = np.linalg.inv(negHess)[:numpara,:numpara]
        crossL = -negHess[:numpara,numpara:]
        term1 = d_theta + d_lambda@invH@crossL.T
        var2 = term1@invW@term1.T + d_lambda@invH@d_lambda.T #- 2*term1@invW@crossL@invH@d_lambda.T
        var1=np.zeros((numpara,numpara))
        ape_mat_tilde = ape_mat - ape
        term2 = np.matmul(ape_mat_tilde.transpose(2,0,1), ape_mat_tilde.transpose(2,1,0))
        term2=term2.sum(1).sum(1)-np.diagonal(term2,axis1=1,axis2=2).sum(1)
        term1 = np.matmul(ape_mat_tilde.transpose(2,1,0), ape_mat_tilde.transpose(2,0,1))
        term1=term1.sum(1).sum(1)-np.diagonal(term1,axis1=1,axis2=2).sum(1)
        var1=np.diag(term1+term2)
        var1 = var1/((N*(N-1))**2)
        return np.sqrt(np.diag(var1+var2))
    #----------------------------------------------------------
    # Helper functions for instantly computing various terms
    #----------------------------------------------------------
    # Helper function for computing various partial derivatives of log-likelihood for each obs.
    def scoreBiasHelper(parameter):
        # Undirected model
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #computations
            P     = sp.stats.logistic._cdf(Xb)        
            derP  = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(P, 0)
            np.fill_diagonal(derP, 0)
            derderP   = derP*(1-2*P)
            fenmu = derP.sum(1)
            sB1 = (((((derderP.reshape(-1,1)*X.reshape(-1,kz)).reshape(N,N,kz)).sum(1))/(fenmu.reshape(-1,1)))[:-1,:].sum(0))/2
            derderPmat = derderP + np.diag(derderP.sum(1))
            sB2 = (((derderPmat/fenmu)[:,:-1].sum(1))/2 )[:-1]       
        # Directed model with mutual
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            np.fill_diagonal(P, 0)
            np.fill_diagonal(P11, 0)
            derP   = P*(1-P)
            derP11 = P11 - P*P.T
            derderP = derP*(1-2*P)
            derderP11 = derP11*(1-2*P)
            transX = X.transpose(1,0,2)
            np.fill_diagonal(derP, 0)
            np.fill_diagonal(derP11, 0)
            derFactor = 1-2*P
            np.fill_diagonal(derFactor, 0)
            fenmu1 = derP.sum(1)
            fenmu0 = derP.sum(0)
            sB1b = (((((derderP.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx)).sum(1)+((derderP11.reshape(-1,1)*transX.reshape(-1,kx)).reshape(N,N,kx)).sum(1))/(fenmu1.reshape(-1,1)))[:-1,:].sum(0)
                   +((((derderP.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx)).sum(0)+((derderP11.reshape(-1,1)*transX.reshape(-1,kx)).reshape(N,N,kx)).sum(0))/(fenmu0.reshape(-1,1)))[:-1,:].sum(0))/2
            sB1d = ((((((((1-P)*P11).reshape(-1,1)*Z.reshape(-1,kz)))*derFactor.reshape(-1,1)).reshape(N,N,kz).sum(1))/(fenmu1.reshape(-1,1)))[:-1,:].sum(0)
                   +(((((((1-P)*P11).reshape(-1,1)*Z.reshape(-1,kz)))*derFactor.reshape(-1,1)).reshape(N,N,kz).sum(0))/(fenmu0.reshape(-1,1)))[:-1,:].sum(0))/2
            sB1 = np.hstack((sB1b, sB1d))
            sB2 = np.hstack(((((((derderP).sum(1)/fenmu1) + ((derderP11).sum(0)/fenmu0))+(((derderP/fenmu0.reshape(1,-1))[:,:-1].sum(1)) + ((derderP11.T/fenmu1.reshape(1,-1))[:,:-1].sum(1))))/2)[:-1]  ,(((((derderP.T/fenmu1.reshape(1,-1))[:,:-1].sum(1)) + ((derderP11/fenmu0.reshape(1,-1))[:,:-1].sum(1)))+((derderP.sum(0)/fenmu0) + ((derderP11).sum(1)/fenmu1)))/2)[:-1]))
        # Directed model without mutual
        if directed==True and mutual==False: 
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations
            P     = sp.stats.logistic._cdf(Xb)        
            derP  = sp.stats.logistic._pdf(Xb)
            np.fill_diagonal(P, 0)
            np.fill_diagonal(derP, 0)
            derderP   = derP*(1-2*P)
            derderP_b1 = ((derderP.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx)).sum(1)
            derderP_b0 = ((derderP.reshape(-1,1)*X.reshape(-1,kx)).reshape(N,N,kx)).sum(0)
            fenmu1 = derP.sum(1)
            fenmu0 = derP.sum(0)
            sB1 = ((derderP_b1/(fenmu1.reshape(-1,1)))[:-1,:].sum(0)+(derderP_b0/(fenmu0.reshape(-1,1)))[:-1,:].sum(0))/2
            sB2 = np.hstack(((((derderP.sum(1)/fenmu1)+((derderP/fenmu0.reshape(1,-1))[:,:-1].sum(1)))/2)[:-1]  ,((((derderP.T/fenmu1.reshape(1,-1))[:,:-1].sum(1))+(derderP.sum(0)/fenmu0))/2)[:-1]))
        return np.hstack((sB1,sB2))
    # Helper function for computing the first and second order derivatives of APE wrt FE parass
    def ape_helper(parameter):
        if directed==False:
            #organize parameters and compute linear pred
            alphamat=np.tile((np.append(parameter[kz:],np.array([0]))).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kz)@ parameter[:kz].reshape(-1,1)).reshape(N,N) + alphamat + alphamat.T
            #computations - first derivative wrt beta and FE parameters
            pdf = sp.stats.logistic._pdf(Xb) 
            cdf = sp.stats.logistic._cdf(Xb) 
            np.fill_diagonal(pdf, 0)
            np.fill_diagonal(cdf, 0)
            der_lambda_it =  cdf-3*cdf**2+2*cdf**3
            der_theta_it =   der_lambda_it.reshape(-1,1)*X.reshape(-1,kz)
            der_theta = (1/(N*(N-1)))*(parameter[:kz].reshape(-1,1)@der_theta_it.sum(axis=0).reshape(1,-1)
                                              +np.diag(np.repeat(pdf.reshape(-1).sum(),kz)))
            der_lambda_mat = (2/(N*(N-1)))*(parameter[:kz].reshape(-1)*der_lambda_it.reshape(-1,1)).reshape(N,N,kz)
            der_lambda = (der_lambda_mat.sum(1).T)[:,:-1]
            #computations - second derivative wrt FE parameters
            derder_lambda = (cdf-7*cdf**2+12*cdf**3-6*cdf**4).reshape(-1,1)
            derder_lambdamat = ((2/(N*(N-1)))*(parameter[:kz].reshape(-1)*(derder_lambda)).reshape(N,N,kz))
            derder_lambdamat_diag = derder_lambdamat.sum(1)
            for kkk in range(kz):
                derder_lambdamat[:,:,kkk] = derder_lambdamat[:,:,kkk]+ np.diag(derder_lambdamat_diag[:,kkk])
            derder_lambda_final = derder_lambdamat[:-1,:-1,:]
            #computations - discrete terms
            matone = np.ones((N,N))
            np.fill_diagonal(matone, 0)
            for kkk in range(kz):
                if dummyIndicatorZ[kkk]==True: 
                    X0B = Xb - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    PDF1 = sp.stats.logistic._pdf(X1B); PDF0 = sp.stats.logistic._pdf(X0B)
                    CDF1 = sp.stats.logistic._cdf(X1B); CDF0 = sp.stats.logistic._cdf(X0B)
                    X1=np.copy(X); X0=np.copy(X)
                    X1[:,:,kkk] = 1* (np.ones(N)-np.eye(int(N))) 
                    X0[:,:,kkk] = 0
                    der_theta[kkk,:] = (1/(N*(N-1)))*(PDF1.reshape(N,N,1)*X1 - PDF0.reshape(N,N,1)*X0).sum(0).sum(0)
                    der_lambda[kkk,:] = (2/(N*(N-1)))*((PDF1 - PDF0).sum(1))[:-1]
                    derder_lambdamat_kx = (2/(N*(N-1)))*((CDF1-3*CDF1**2+2*CDF1**3)-(CDF0-3*CDF0**2+2*CDF0**3))#* (np.ones(N)-np.eye(int(N)))
                    derder_lambdamat_diag_kx = derder_lambdamat_kx.sum(1)
                    derder_lambda_final[:,:,kkk] = (derder_lambdamat_kx+np.diag(derder_lambdamat_diag_kx))[:-1,:-1]
        if directed==True and mutual==False:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx:kx+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Xb = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations - first derivative wrt beta and FE parameters
            pdf = sp.stats.logistic._pdf(Xb) 
            cdf = sp.stats.logistic._cdf(Xb) 
            np.fill_diagonal(pdf, 0)
            np.fill_diagonal(cdf, 0)
            der_lambda_it =  cdf-3*cdf**2+2*cdf**3
            der_theta_it =   der_lambda_it.reshape(-1,1)*X.reshape(-1,kx)
            der_theta = (1/(N*(N-1)))*(parameter[:kx].reshape(-1,1)@der_theta_it.sum(axis=0).reshape(1,-1)
                                              +np.diag(np.repeat(pdf.reshape(-1).sum(),kx)))
            der_lambda_mat = (1/(N*(N-1)))*(parameter[:kx].reshape(-1)*der_lambda_it.reshape(-1,1)).reshape(N,N,kx)
            der_lambda = np.vstack((der_lambda_mat.sum(axis=1)[:-1,:],der_lambda_mat.sum(axis=0)[:-1,:])).T
            #computations - second derivative wrt FE parameters
            derder_lambda = (cdf-7*cdf**2+12*cdf**3-6*cdf**4).reshape(-1,1)
            derder_lambdamat = ((1/(N*(N-1)))*(parameter[:kx].reshape(-1)*(derder_lambda)).reshape(N,N,kx))
            derder_lambdamat_diag = np.vstack((derder_lambdamat.sum(1)[:-1,:],derder_lambdamat.sum(0)[:-1,:]))
            derder_lambda_final = np.zeros((2*N-2,2*N-2,kx))
            for kkk in range(kx):
                derder_lambda_final[:,:,kkk] = np.diag(derder_lambdamat_diag[:,kkk])
                derder_lambda_final[:N-1,N-1:,kkk] = derder_lambdamat[:,:,kkk][:-1,:-1]
                derder_lambda_final[N-1:,:N-1,kkk] = derder_lambda_final[:N-1,N-1:,kkk].T
            #computations - discrete terms
            matone = np.ones((N,N))
            np.fill_diagonal(matone, 0)
            for kkk in range(kx):
                if dummyIndicatorX[kkk]==True: 
                    X0B = Xb - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    PDF1 = sp.stats.logistic._pdf(X1B); PDF0 = sp.stats.logistic._pdf(X0B)
                    CDF1 = sp.stats.logistic._cdf(X1B); CDF0 = sp.stats.logistic._cdf(X0B)
                    X1=np.copy(X); X0=np.copy(X)
                    X1[:,:,kkk] = 1* (np.ones(N)-np.eye(int(N))) 
                    X0[:,:,kkk] = 0
                    der_theta[kkk,:] = (1/(N*(N-1)))*(PDF1.reshape(N,N,1)*X1 - PDF0.reshape(N,N,1)*X0).sum(0).sum(0)
                    der_lambda[kkk,:N-1] = (1/(N*(N-1)))*((PDF1 - PDF0).sum(1))[:-1]
                    der_lambda[kkk,N-1:] = (1/(N*(N-1)))*((PDF1 - PDF0).sum(0))[:-1]
                    derder_lambdamat_kx = (1/(N*(N-1)))*((CDF1-3*CDF1**2+2*CDF1**3)-(CDF0-3*CDF0**2+2*CDF0**3))#* (np.ones(N)-np.eye(int(N)))
                    derder_lambdamat_diag_kx =  np.append(derder_lambdamat_kx.sum(1)[:-1],derder_lambdamat_kx.sum(0)[:-1])
                    derder_lambda_final[:,:,kkk] = np.diag(derder_lambdamat_diag_kx)
                    derder_lambda_final[:N-1,N-1:,kkk] = (derder_lambdamat_kx)[:-1,:-1]
                    derder_lambda_final[N-1:,:N-1,kkk] = derder_lambda_final[:N-1,N-1:,kkk].T
        if directed==True and mutual==True:  
            #organize parameters and compute linear pred
            alphamat=np.tile(np.append(parameter[kx+kz:kx+kz+N-1],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            gammamat=np.tile(np.append(parameter[kx+kz+N-1:],np.array([0])).reshape(1,N),(N,1)).reshape(N,N).T
            Zb_m = (Z.reshape(-1,kz)@ parameter[kx:kx+kz].reshape(-1,1)).reshape(N,N)
            Xb_u = (X.reshape(-1,kx)@ parameter[:kx].reshape(-1,1)).reshape(N,N) + alphamat + gammamat.T
            #computations - first derivative wrt beta and FE parameters
            fenmu = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Zb_m))
            P   = (np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Zb_m)) / fenmu
            P11 = np.exp(Xb_u+Xb_u.T+Zb_m) / fenmu
            P10 = np.exp(Xb_u) / fenmu
            P01 = np.exp(Xb_u.T) / fenmu
            np.fill_diagonal(P, 0)
            np.fill_diagonal(P11, 0)
            np.fill_diagonal(P10, 0)
            np.fill_diagonal(P01, 0)
            pdf_direct =P*(1-P)
            pdf_mutual =P11*(1-P11)
            pdf_direct_cross  =P11-P*P.T
            pdf_mutual_cross1  =P11-P.T*P11
            pdf_mutual_cross2  =P11-P*P11
            transX = X.transpose(1,0,2) 
            der_bPij= (  (pdf_direct.reshape(-1,1)*X.reshape(-1,kx))
                       + (pdf_direct_cross.reshape(-1,1)*transX.reshape(-1,kx)))
            der_dPij = (pdf_mutual_cross2.reshape(-1,1)*Z.reshape(-1,kz))
            t1_beta = parameter[:kx].reshape(-1,1)@((((der_bPij - 2*P.reshape(-1,1)*der_bPij).sum(0))/(N*(N-1))).reshape(1,-1))+np.diag(np.repeat(pdf_direct.sum()/(N*(N-1)),kx))
            t1_delta = parameter[:kx].reshape(-1,1)@((((der_dPij - 2*P.reshape(-1,1)*der_dPij).sum(0))/(N*(N-1))).reshape(1,-1))
            der_bPij11= (  (P11.reshape(-1,1)*(X.reshape(-1,kx)+transX.reshape(-1,kx)))
                         - ((P11*P10).reshape(-1,1)*X.reshape(-1,kx))
                         - ((P11*P01).reshape(-1,1)*transX.reshape(-1,kx))
                         - ((P11*P11).reshape(-1,1)*(X.reshape(-1,kx)+transX.reshape(-1,kx))))
            der_dPij11 = (pdf_mutual.reshape(-1,1)*Z.reshape(-1,kz))
            t2_beta  = parameter[kx:kx+kz].reshape(-1,1)@((((der_bPij11 - 2*P11.reshape(-1,1)*der_bPij11).sum(0))/(N*(N-1))).reshape(1,-1))
            t2_delta = parameter[kx:kx+kz].reshape(-1,1)@((((der_dPij11 - 2*P11.reshape(-1,1)*der_dPij11).sum(0))/(N*(N-1))).reshape(1,-1))+np.diag(np.repeat(pdf_mutual.sum()/(N*(N-1)),kz))
            der_theta = np.vstack((np.hstack((t1_beta,t1_delta)), np.hstack((t2_beta,t2_delta))))
            t1_lambda =np.vstack(((parameter[:kx].reshape(-1)*(((pdf_direct*(1-2*P)).sum(1)+(pdf_direct_cross*(1-2*P)).sum(0))/(N*(N-1))).reshape(-1,1))[:-1,:],
                                 (parameter[:kx].reshape(-1)*(((pdf_direct*(1-2*P)).sum(0)+(pdf_direct_cross*(1-2*P)).sum(1))/(N*(N-1))).reshape(-1,1))[:-1,:],
                                 )).T
            t2_lambda=np.vstack(((parameter[kx:kx+kz].reshape(-1)*(((pdf_mutual_cross1*(1-2*P11)).sum(0)+(pdf_mutual_cross2*(1-2*P11)).sum(1))/(N*(N-1))).reshape(-1,1))[:-1,:],
                                 (parameter[kx:kx+kz].reshape(-1)*(((pdf_mutual_cross1*(1-2*P11)).sum(1)+(pdf_mutual_cross2*(1-2*P11)).sum(0))/(N*(N-1))).reshape(-1,1))[:-1,:],
                                 )).T
            der_lambda = np.vstack((t1_lambda,t2_lambda))
            #computations - second derivative wrt FE parameters
            pij = P
            pji = P.T
            pij11 = P11.T
            d_ai_pij   = (pij-pij**2)
            d_aj_pji   = (pji-pji**2)
            d_ai_pij11 = (pij11-pij11*pij)
            d_aj_pij11 = (pij11-pij11*pji)
            d_ai_pji   = (pij11-pij*pji)
            d_aj_pij   = np.copy(d_ai_pji)
            d_gi_pij   = np.copy(d_ai_pji)
            d_gj_pji   = np.copy(d_ai_pji)
            d_gi_pij11 = (pij11-pij11*pji)
            d_gj_pij11 = (pij11-pij11*pij)
            d_gi_pji   = (pji-pji**2)
            d_gj_pij   = (pij-pij**2)
            # beta part
            a_diag = (d_ai_pij - 6*pij*d_ai_pij + 6*(pij**2)*d_ai_pij
                        + d_ai_pij11 - d_ai_pij*pji - d_ai_pji*pij 
                        - 2*d_ai_pji*pij11 - 2*d_ai_pij11*pji 
                        + 2*d_ai_pij*(pji**2) + 4*d_ai_pji*pij*pji)
            a_offdiag = (d_aj_pij - 6*pij*d_aj_pij + 6*(pij**2)*d_aj_pij
                         + d_aj_pij11 - d_aj_pij*pji - d_aj_pji*pij 
                         - 2*d_aj_pji*pij11 - 2*d_aj_pij11*pji 
                         + 2*d_aj_pij*(pji**2) + 4*d_aj_pji*pij*pji)
            M11 = (a_offdiag[:-1,:-1] + np.diag(a_diag.sum(1)[:-1]))/(N*(N-1))
            a_cross_diag = (d_gi_pij - 6*pij*d_gi_pij + 6*(pij**2)*d_gi_pij
                            + d_gi_pij11 - d_gi_pij*pji - d_gi_pji*pij 
                            - 2*d_gi_pji*pij11 - 2*d_gi_pij11*pji 
                            + 2*d_gi_pij*(pji**2) + 4*d_gi_pji*pij*pji)
            a_cross_offdiag = (d_gj_pij - 6*pij*d_gj_pij + 6*(pij**2)*d_gj_pij
                               + d_gj_pij11 - d_gj_pij*pji - d_gj_pji*pij 
                               - 2*d_gj_pji*pij11 - 2*d_gj_pij11*pji 
                               + 2*d_gj_pij*(pji**2) + 4*d_gj_pji*pij*pji)
            M12 = (a_cross_offdiag[:-1,:-1] + np.diag(a_cross_diag.sum(1)[:-1]))/(N*(N-1))
            g_diag = (d_gi_pji - 6*pji*d_gi_pji + 6*(pji**2)*d_gi_pji
                        + d_gi_pij11 - d_gi_pji*pij - d_gi_pij*pji 
                        - 2*d_gi_pij*pij11 - 2*d_gi_pij11*pij 
                        + 2*d_gi_pji*(pij**2) + 4*d_gi_pij*pji*pij)
            g_offdiag = (d_gj_pji - 6*pji*d_gj_pji + 6*(pji**2)*d_gj_pji
                         + d_gj_pij11 - d_gj_pji*pij - d_gj_pij*pji 
                         - 2*d_gj_pij*pij11 - 2*d_gj_pij11*pij 
                         + 2*d_gj_pji*(pij**2) + 4*d_gj_pij*pji*pij)
            M22 = (g_offdiag[:-1,:-1] + np.diag(g_diag.sum(1)[:-1]))/(N*(N-1))
            g_cross_diag = (d_ai_pji - 6*pji*d_ai_pji + 6*(pji**2)*d_ai_pji
                            + d_ai_pij11 - d_ai_pji*pij - d_ai_pij*pji 
                            - 2*d_ai_pij*pij11 - 2*d_ai_pij11*pij 
                            + 2*d_ai_pji*(pij**2) + 4*d_ai_pij*pji*pij)
            g_cross_offdiag = (d_aj_pji - 6*pji*d_aj_pji + 6*(pji**2)*d_aj_pji
                               + d_aj_pij11 - d_aj_pji*pij - d_aj_pij*pji 
                               - 2*d_aj_pij*pij11 - 2*d_aj_pij11*pij 
                               + 2*d_aj_pji*(pij**2) + 4*d_aj_pij*pji*pij)
            M21 = (g_cross_offdiag[:-1,:-1] + np.diag(g_cross_diag.sum(1)[:-1]))/(N*(N-1))
            derder_lambda_b = (parameter[:kx].reshape(-1)*np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22)))).reshape(-1,1)).reshape(2*N-2,2*N-2,kx)
            # delta part
            a_diag =    (d_ai_pij11 - d_ai_pij11*pij - d_ai_pij*pij11 -4*pij11*d_ai_pij11
                         +4*pij11*pij*d_ai_pij11 + 2*(pij11**2)*d_ai_pij )
            a_offdiag = (d_aj_pij11 - d_aj_pij11*pij - d_aj_pij*pij11 -4*pij11*d_aj_pij11
                         +4*pij11*pij*d_aj_pij11 + 2*(pij11**2)*d_aj_pij )
            M11 = (a_offdiag[:-1,:-1] + np.diag(a_diag.sum(1)[:-1]))/(N*(N-1))
            a_cross_diag = (d_gi_pij11 - d_gi_pij11*pij - d_gi_pij*pij11 -4*pij11*d_gi_pij11
                           +4*pij11*pij*d_gi_pij11 + 2*(pij11**2)*d_gi_pij )
            a_cross_offdiag = (d_gj_pij11 - d_gj_pij11*pij - d_gj_pij*pij11 -4*pij11*d_gj_pij11
                           +4*pij11*pij*d_gj_pij11 + 2*(pij11**2)*d_gj_pij )
            M12 = (a_cross_offdiag[:-1,:-1] + np.diag(a_cross_diag.sum(1)[:-1]))/(N*(N-1))
            g_diag =    (d_gi_pij11 - d_gi_pij11*pji - d_gi_pji*pij11 -4*pij11*d_gi_pij11
                         +4*pij11*pji*d_gi_pij11 + 2*(pij11**2)*d_gi_pji )
            g_offdiag = (d_gj_pij11 - d_gj_pij11*pji - d_gj_pji*pij11 -4*pij11*d_gj_pij11
                         +4*pij11*pji*d_gj_pij11 + 2*(pij11**2)*d_gj_pji )
            M22 = (g_offdiag[:-1,:-1] + np.diag(g_diag.sum(1)[:-1]))/(N*(N-1))
            g_cross_diag =    (d_ai_pij11 - d_ai_pij11*pji - d_ai_pji*pij11 -4*pij11*d_ai_pij11
                                +4*pij11*pji*d_ai_pij11 + 2*(pij11**2)*d_ai_pji )
            g_cross_offdiag = (d_aj_pij11 - d_aj_pij11*pji - d_aj_pji*pij11 -4*pij11*d_aj_pij11
                                +4*pij11*pji*d_aj_pij11 + 2*(pij11**2)*d_aj_pji )
            M21 = (g_cross_offdiag[:-1,:-1] + np.diag(g_cross_diag.sum(1)[:-1]))/(N*(N-1))
            derder_lambda_d = 2*(parameter[kx:kx+kz].reshape(-1)*np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22)))).reshape(-1,1)).reshape(2*N-2,2*N-2,kz)
            derder_lambda_final = np.concatenate((derder_lambda_b,derder_lambda_d),axis=2)
            #computations - discrete terms
            matone = np.ones((N,N))
            np.fill_diagonal(matone, 0)
            for kkk in range(kx):
                if dummyIndicatorX[kkk]==True: 
                    X0B = Xb_u - X[:,:,kkk]*parameter[kkk]
                    X1B = X0B + matone*parameter[kkk]
                    X1=np.copy(X); X0=np.copy(X)
                    X1[:,:,kkk] = 1* (np.ones(N)-np.eye(int(N))) 
                    X0[:,:,kkk] = 0
                    transX1 = X1.transpose(1,0,2) 
                    transX0 = X0.transpose(1,0,2) 
                    fenmu_1 = (1+ np.exp(X1B) + np.exp(X1B).T + np.exp(X1B+X1B.T+Zb_m))
                    fenmu_0 = (1+ np.exp(X0B) + np.exp(X0B).T + np.exp(X0B+X0B.T+Zb_m))
                    P_1 = ((np.exp(X1B) + np.exp(X1B+X1B.T+Zb_m)) / fenmu_1).reshape(N,N,1)
                    P_0 = ((np.exp(X0B) + np.exp(X0B+X0B.T+Zb_m)) / fenmu_0).reshape(N,N,1)
                    P11_1 = ((np.exp(X1B+X1B.T+Zb_m))/ fenmu_1).reshape(N,N,1)
                    P11_0 = ((np.exp(X0B+X0B.T+Zb_m))/ fenmu_0).reshape(N,N,1)
                    P10_1 = ((np.exp(X1B))   / fenmu_1).reshape(N,N,1)
                    P10_0 = ((np.exp(X0B))   / fenmu_0).reshape(N,N,1)
                    P01_1 = ((np.exp(X1B).T) / fenmu_1).reshape(N,N,1)
                    P01_0 = ((np.exp(X0B).T) / fenmu_0).reshape(N,N,1)
                    PDF1b = P10_1*X1 + P11_1*(X1+transX1) - P_1*(P10_1*X1+P01_1*transX1+P11_1*(X1+transX1))
                    PDF0b = P10_0*X0 + P11_0*(X0+transX0) - P_0*(P10_0*X0+P01_0*transX0+P11_0*(X0+transX0))
                    der_theta[kkk,:kx]   = (1/(N*(N-1)))*(PDF1b - PDF0b).sum(0).sum(0)
                    PDF1d = (P11_1-P_1*P11_1)*Z
                    PDF0d = (P11_0-P_0*P11_0)*Z
                    der_theta[kkk,kx:]   = (1/(N*(N-1)))*(PDF1d - PDF0d).sum(0).sum(0)
                    
                    pij_1 = P_1.reshape(N,N)
                    pji_1 = P_1.reshape(N,N).T
                    pij11_1 = P11_1.reshape(N,N).T
                    d_ai_pij_1   = (pij_1-pij_1**2)
                    d_aj_pji_1   = (pji_1-pji_1**2)
                    d_ai_pij11_1 = (pij11_1-pij11_1*pij_1)
                    d_aj_pij11_1 = (pij11_1-pij11_1*pji_1)
                    d_ai_pji_1   = (pij11_1-pij_1*pji_1)
                    d_aj_pij_1   = np.copy(d_ai_pji_1)
                    d_gi_pij_1   = np.copy(d_ai_pji_1)
                    d_gj_pji_1   = np.copy(d_ai_pji_1)
                    d_gi_pij11_1 = (pij11_1-pij11_1*pji_1)
                    d_gj_pij11_1 = (pij11_1-pij11_1*pij_1)
                    d_gi_pji_1   = (pji_1-pji_1**2)
                    d_gj_pij_1   = (pij_1-pij_1**2)
                     
                    pij_0 = P_0.reshape(N,N)
                    pji_0 = P_0.reshape(N,N).T
                    pij11_0 = P11_0.reshape(N,N).T
                    d_ai_pij_0   = (pij_0-pij_0**2)
                    d_aj_pji_0   = (pji_0-pji_0**2)
                    d_ai_pij11_0 = (pij11_0-pij11_0*pij_0)
                    d_aj_pij11_0 = (pij11_0-pij11_0*pji_0)
                    d_ai_pji_0   = (pij11_0-pij_0*pji_0)
                    d_aj_pij_0   = np.copy(d_ai_pji_0)
                    d_gi_pij_0   = np.copy(d_ai_pji_0)
                    d_gj_pji_0   = np.copy(d_ai_pji_0)
                    d_gi_pij11_0 = (pij11_0-pij11_0*pji_0)
                    d_gj_pij11_0 = (pij11_0-pij11_0*pij_0)
                    d_gi_pji_0   = (pji_0-pji_0**2)
                    d_gj_pij_0   = (pij_0-pij_0**2)
                    
                    der_lambda[kkk,:N-1] = (1/(N*(N-1)))*((d_ai_pij_1+d_ai_pji_1).sum(1) - (d_ai_pij_0+d_ai_pji_0).sum(1)).reshape(-1)[:-1]
                    der_lambda[kkk,N-1:] = (1/(N*(N-1)))*((d_gi_pij_1+d_gi_pji_1).sum(1) - (d_gi_pij_0+d_gi_pji_0).sum(1)).reshape(-1)[:-1]
                    
                    a_diag = (  (d_ai_pij_1 - 2*pij_1*d_ai_pij_1 + d_ai_pij11_1 - d_ai_pji_1*pij_1 - pji_1*d_ai_pij_1) 
                              - (d_ai_pij_0 - 2*pij_0*d_ai_pij_0 + d_ai_pij11_0 - d_ai_pji_0*pij_0 - pji_0*d_ai_pij_0) )
                    a_offdiag = (  (d_aj_pij_1 - 2*pij_1*d_aj_pij_1 + d_aj_pij11_1 - d_aj_pji_1*pij_1 - pji_1*d_aj_pij_1) 
                                 - (d_aj_pij_0 - 2*pij_0*d_aj_pij_0 + d_aj_pij11_0 - d_aj_pji_0*pij_0 - pji_0*d_aj_pij_0) )
                    M11 = (a_offdiag[:-1,:-1] + np.diag(a_diag.sum(1)[:-1]))/(N*(N-1))
                    a_cross_diag = (  (d_gi_pij_1 - 2*pij_1*d_gi_pij_1 + d_gi_pij11_1 - d_gi_pji_1*pij_1 - pji_1*d_gi_pij_1) 
                                    - (d_gi_pij_0 - 2*pij_0*d_gi_pij_0 + d_gi_pij11_0 - d_gi_pji_0*pij_0 - pji_0*d_gi_pij_0) )
                    a_cross_offdiag = (  (d_gj_pij_1 - 2*pij_1*d_gj_pij_1 + d_gj_pij11_1 - d_gj_pji_1*pij_1 - pji_1*d_gj_pij_1) 
                                       - (d_gj_pij_0 - 2*pij_0*d_gj_pij_0 + d_gj_pij11_0 - d_gj_pji_0*pij_0 - pji_0*d_gj_pij_0) )
                    M12 = (a_cross_offdiag[:-1,:-1] + np.diag(a_cross_diag.sum(1)[:-1]))/(N*(N-1))
                    g_diag = (  (d_gi_pji_1 - 2*pji_1*d_gi_pji_1 + d_gi_pij11_1 - d_gi_pji_1*pij_1 - pji_1*d_gi_pij_1) 
                              - (d_gi_pji_0 - 2*pji_0*d_gi_pji_0 + d_gi_pij11_0 - d_gi_pji_0*pij_0 - pji_0*d_gi_pij_0) )
                    g_offdiag = (  (d_gj_pji_1 - 2*pji_1*d_gj_pji_1 + d_gj_pij11_1 - d_gj_pji_1*pij_1 - pji_1*d_gj_pij_1) 
                              - (   d_gj_pji_0 - 2*pji_0*d_gj_pji_0 + d_gj_pij11_0 - d_gj_pji_0*pij_0 - pji_0*d_gj_pij_0) )
                    M22 = (g_offdiag[:-1,:-1] + np.diag(g_diag.sum(1)[:-1]))/(N*(N-1))
                    g_cross_diag = (  (d_ai_pji_1 - 2*pji_1*d_ai_pji_1 + d_ai_pij11_1 - d_ai_pji_1*pij_1 - pji_1*d_ai_pij_1) 
                                    - (d_ai_pji_0 - 2*pji_0*d_ai_pji_0 + d_ai_pij11_0 - d_ai_pji_0*pij_0 - pji_0*d_ai_pij_0) )
                    g_cross_offdiag =    (  (d_aj_pji_1 - 2*pji_1*d_aj_pji_1 + d_aj_pij11_1 - d_aj_pji_1*pij_1 - pji_1*d_aj_pij_1) 
                                       - (   d_aj_pji_0 - 2*pji_0*d_aj_pji_0 + d_aj_pij11_0 - d_aj_pji_0*pij_0 - pji_0*d_aj_pij_0) )
                    M21 = (g_cross_offdiag[:-1,:-1] + np.diag(g_cross_diag.sum(1)[:-1]))/(N*(N-1))
                    derder_lambda_final[:,:,kkk] = (np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22)))).reshape(-1,1)).reshape(2*N-2,2*N-2)
            for kkk in range(kz):
                if dummyIndicatorZ[kkk]==True: 
                    Z0B = Zb_m - Z[:,:,kkk]*parameter[kx+kkk]
                    Z1B = Z0B + matone*parameter[kx+kkk]
                    Z1=np.copy(Z); Z0=np.copy(Z)
                    Z1[:,:,kkk] = 1* (np.ones(N)-np.eye(int(N))) 
                    Z0[:,:,kkk] = 0
                    fenmu_1 = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Z1B))
                    fenmu_0 = (1+ np.exp(Xb_u) + np.exp(Xb_u).T + np.exp(Xb_u+Xb_u.T+Z0B))
                    P_1 = ((np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Z1B)) / fenmu_1).reshape(N,N,1)
                    P_0 = ((np.exp(Xb_u) + np.exp(Xb_u+Xb_u.T+Z0B)) / fenmu_0).reshape(N,N,1)
                    P11_1 = (np.exp(Xb_u+Xb_u.T+Z1B) / fenmu_1).reshape(N,N,1)
                    P11_0 = (np.exp(Xb_u+Xb_u.T+Z0B) / fenmu_0).reshape(N,N,1)
                    P10_1 = ((np.exp(Xb_u))   / fenmu_1).reshape(N,N,1)
                    P10_0 = ((np.exp(Xb_u))   / fenmu_0).reshape(N,N,1)
                    P01_1 = ((np.exp(Xb_u).T) / fenmu_1).reshape(N,N,1)
                    P01_0 = ((np.exp(Xb_u).T) / fenmu_0).reshape(N,N,1)
                    PDF1b = P11_1*(X+transX) - P11_1*(P10_1*X+P01_1*transX+P11_1*(X+transX))
                    PDF0b = P11_0*(X+transX) - P11_0*(P10_0*X+P01_0*transX+P11_0*(X+transX))
                    der_theta[kx+kkk,:kx]   = (1/(N*(N-1)))*(PDF1b - PDF0b).sum(0).sum(0)
                    PDF1d = (P11_1-P11_1**2)*Z1
                    PDF0d = (P11_0-P11_0**2)*Z0
                    der_theta[kx+kkk,kx:]   = (1/(N*(N-1)))*(PDF1d - PDF0d).sum(0).sum(0)
                    
                    pij_1 = P_1.reshape(N,N)
                    pji_1 = P_1.reshape(N,N).T
                    pij11_1 = P11_1.reshape(N,N).T
                    d_ai_pij_1   = (pij_1-pij_1**2)
                    d_aj_pji_1   = (pji_1-pji_1**2)
                    d_ai_pij11_1 = (pij11_1-pij11_1*pij_1)
                    d_aj_pij11_1 = (pij11_1-pij11_1*pji_1)
                    d_ai_pji_1   = (pij11_1-pij_1*pji_1)
                    d_aj_pij_1   = np.copy(d_ai_pji_1)
                    d_gi_pij_1   = np.copy(d_ai_pji_1)
                    d_gj_pji_1   = np.copy(d_ai_pji_1)
                    d_gi_pij11_1 = (pij11_1-pij11_1*pji_1)
                    d_gj_pij11_1 = (pij11_1-pij11_1*pij_1)
                    d_gi_pji_1   = (pji_1-pji_1**2)
                    d_gj_pij_1   = (pij_1-pij_1**2)
                     
                    pij_0 = P_0.reshape(N,N)
                    pji_0 = P_0.reshape(N,N).T
                    pij11_0 = P11_0.reshape(N,N).T
                    d_ai_pij_0   = (pij_0-pij_0**2)
                    d_aj_pji_0   = (pji_0-pji_0**2)
                    d_ai_pij11_0 = (pij11_0-pij11_0*pij_0)
                    d_aj_pij11_0 = (pij11_0-pij11_0*pji_0)
                    d_ai_pji_0   = (pij11_0-pij_0*pji_0)
                    d_aj_pij_0   = np.copy(d_ai_pji_0)
                    d_gi_pij_0   = np.copy(d_ai_pji_0)
                    d_gj_pji_0   = np.copy(d_ai_pji_0)
                    d_gi_pij11_0 = (pij11_0-pij11_0*pji_0)
                    d_gj_pij11_0 = (pij11_0-pij11_0*pij_0)
                    d_gi_pji_0   = (pji_0-pji_0**2)
                    d_gj_pij_0   = (pij_0-pij_0**2)
                    
                    der_lambda[kx+kkk,:N-1] = (1/(N*(N-1)))*((2*d_ai_pij11_1).sum(1) - (2*d_ai_pij11_0).sum(1) ).reshape(-1)[:-1]
                    der_lambda[kx+kkk,N-1:] = (1/(N*(N-1)))*((2*d_gi_pij11_1).sum(1) - (2*d_gi_pij11_0).sum(1)).reshape(-1)[:-1]
                    
                    a_diag =    (  (d_ai_pij11_1*(1-pij_1) - pij11_1*d_ai_pij_1) 
                                 - (d_ai_pij11_0*(1-pij_0) - pij11_0*d_ai_pij_0) )
                    a_offdiag = (  (d_aj_pij11_1*(1-pij_1) - pij11_1*d_aj_pij_1) 
                                 - (d_aj_pij11_0*(1-pij_0) - pij11_0*d_aj_pij_0) )
                    M11 = (a_offdiag[:-1,:-1] + np.diag(a_diag.sum(1)[:-1]))/(N*(N-1))
                    a_cross_diag =    (  (d_gi_pij11_1*(1-pij_1) - pij11_1*d_gi_pij_1) 
                                       - (d_gi_pij11_0*(1-pij_0) - pij11_0*d_gi_pij_0) )
                    a_cross_offdiag = (  (d_gj_pij11_1*(1-pij_1) - pij11_1*d_gj_pij_1) 
                                       - (d_gj_pij11_0*(1-pij_0) - pij11_0*d_gj_pij_0) )
                    M12 = (a_cross_offdiag[:-1,:-1] + np.diag(a_cross_diag.sum(1)[:-1]))/(N*(N-1))
                    g_diag =    (  (d_gi_pij11_1*(1-pji_1) - pij11_1*d_gi_pji_1) 
                                 - (d_gi_pij11_0*(1-pji_0) - pij11_0*d_gi_pji_0) )
                    g_offdiag = (  (d_gj_pij11_1*(1-pji_1) - pij11_1*d_gj_pji_1) 
                                 - (d_gj_pij11_0*(1-pji_0) - pij11_0*d_gj_pji_0) )
                    M22 = (g_offdiag[:-1,:-1] + np.diag(g_diag.sum(1)[:-1]))/(N*(N-1))
                    g_cross_diag =    (  (d_ai_pij11_1*(1-pji_1) - pij11_1*d_ai_pji_1) 
                                       - (d_ai_pij11_0*(1-pji_0) - pij11_0*d_ai_pji_0) )
                    g_cross_offdiag = (  (d_aj_pij11_1*(1-pji_1) - pij11_1*d_aj_pji_1) 
                                       - (d_aj_pij11_0*(1-pji_0) - pij11_0*d_aj_pji_0) )
                    M21 = (g_cross_offdiag[:-1,:-1] + np.diag(g_cross_diag.sum(1)[:-1]))/(N*(N-1))
                    derder_lambda_final[:,:,kx+kkk] = 2*(np.vstack((np.hstack((M11,M12)),np.hstack((M21,M22)))).reshape(-1,1)).reshape(2*N-2,2*N-2)
        return der_theta, der_lambda ,derder_lambda_final
    #-----------------------------------------------------------------------
    # Functions for analytical bias corrections for common parameter and APE 
    #-----------------------------------------------------------------------
    torch.inverse(torch.tensor([[1.0,2],[3,4]]))

    # Bias corrections for common parameter -- for estimator based analytical bias correction 
    def biasTerm(parameter):
        derPenalty = scoreBiasHelper(parameter)
        derPenalty_beta   = derPenalty[:numpara].reshape(-1,1)
        derPenalty_lambda = derPenalty[numpara:].reshape(-1,1)
        negHess = negHessAll(parameter)
        H = negHess[numpara:,numpara:]
        crossL = -negHess[:numpara,numpara:]
        if bc_method == "estimator":      bias = -np.linalg.inv(negHess)[:numpara,:numpara]@(derPenalty_beta)
        if bc_method == "estimatorChain": bias = -np.linalg.inv(negHess)[:numpara,:numpara]@(derPenalty_beta + crossL@np.linalg.inv(H)@derPenalty_lambda)
        return bias.reshape(-1)
    # Further bias corrections for APE after likelihood correction
    def apeBC(parameter):
        ape,ape_mat = apeCalc(parameter)
        invH = np.linalg.inv(negHessAll(parameter)[numpara:,numpara:])
        _, _, dd_lambda = ape_helper(parameter)
        ape_bias = np.zeros(numpara)
        for kk in range(numpara):
            ape_bias[kk] = np.trace(dd_lambda[:,:,kk]@invH)/2
        return ape - ape_bias
    #----------------------------------------------------------
    # Implementations
    #----------------------------------------------------------
    start_time = time.time()  # Setup a timer
    # [> Point estimation - No bias correction or Likelihood correction <]
    # JML - No bias correction
    bc_method_keep = bc_method
    bc_method = "nocorr" # Estimate the uncorrected first if data is identified
    if drop_separation==True or seps==0:
        res = sp.optimize.minimize(negLogLike,x0=initials, method='Newton-CG', jac=negJacobianAll, hess=negHessAll)
        sucMess = res.success
        funcLogLike = res.fun
        estNocorr = res.x.reshape(-1)
        estTheta = (res.x.reshape(-1))[:numpara]
    bc_method = bc_method_keep
    # JML - Likelihood correction
    if bc_method == "likelihood":
        res = sp.optimize.minimize(negLogLike,x0=initials, method='BFGS', jac=negJacobianAll, hess=None) # Method can be BFGS or Newton-CG, for likelihood correction, hessian of prior is not provided, and hence BFGS can be more accurate
        sucMess = res.success
        funcLogLike = res.fun
        estLikeCorr = res.x.reshape(-1)
        estTheta = (res.x.reshape(-1))[:numpara]
    # Iterative MLE algorithm - No bias correction or Likelihood correction
    meth_use='Newton-CG'
    if bc_method=="likelihood": hess_use = None; meth_use='BFGS'
    if algorithm == "Iter":
        hess_use = negHessConcen
        if bc_method=="likelihood": hess_use = None
        if bc_method=="likelihood" or bc_method == "nocorr":
            # Starting values -- from JML to speed up the algorithm
                ## ( here, parameterUpdate will be a global variable in this algorithm)
            if bc_method=="likelihood": parameterUpdate = np.copy(estLikeCorr) # parameterUpdate will be a global variable in this algorithm
            if bc_method=="nocorr":     parameterUpdate = np.copy(estNocorr) # parameterUpdate will be a global variable in this algorithm
            # Initilizations
            funVal = -999999
            np.random.seed(1)
            iterlist = range(2000)
            # Iterative MLE starts here
            for sweep in iterlist:
                np.random.seed(sweep)
                stepsize = np.random.randint(50,90)
                newindices=np.arange(np.size(initials))
                np.random.shuffle(newindices)
                for start_i in np.arange(0,np.size(initials),stepsize):
                    end_i = start_i+stepsize
                    parameterSlice = parameterUpdate[newindices[start_i:end_i]]
                    res = sp.optimize.minimize(negLogLikeConcen,x0=parameterSlice, method='Newton-CG', jac=negJacobianConcen, hess=hess_use)
                    if start_i==0: 
                        funValDiff = np.abs(funVal - res.fun)
                        funVal = res.fun
                # Exit criteron 
                    ## ( this is a bit strict -- to ensure more precise estimates obtained
                    ##   by scipy optimization toolbox)
                if funValDiff<1e-15:
                    sucMess = res.success
                    funcLogLike = res.fun
                    break
                # print(sweep, stepsize, funValDiff,parameterUpdate[:4])
            if bc_method=="nocorr":     estNocorr   = np.copy(parameterUpdate)
            if bc_method=="likelihood": estLikeCorr = np.copy(parameterUpdate)
            estTheta = parameterUpdate[:numpara].reshape(-1)
    # Fixed point iteration algorithm - No bias correction or Likelihood correction
    if algorithm == "FP":
        hess_use = negHessTheta
        if bc_method=="likelihood": hess_use = None; 
        if bc_method == "likelihood" or bc_method == "nocorr":
            # Starting values -- from JML to speed up the algorithm
                ## ( here, FE_theta (FE given theta) is a global variable)
            if bc_method=="likelihood": theta_sv = (estLikeCorr[:numpara]) ;  FE_theta = (estLikeCorr[numpara:])
            if bc_method=="nocorr":     theta_sv = (estNocorr[:numpara]) ;  FE_theta = (estNocorr[numpara:])
            # FP starts here
            res = sp.optimize.minimize(negLogLike_thetaFP,x0=theta_sv.reshape(-1), method=meth_use, jac=negJacobianTheta, hess=hess_use)
            sucMess = res.success
            funcLogLike = res.fun
            parameterUpdate   = np.append(res.x, FE_theta).reshape(-1)
            if bc_method=="nocorr":     estNocorr   = np.copy(parameterUpdate)
            if bc_method=="likelihood": estLikeCorr = np.copy(parameterUpdate)
            estTheta = parameterUpdate[:numpara].reshape(-1)
    # [> Point estimation - Bias correction on beta estimate <]
    if bc_method == "estimator" or bc_method == "estimatorChain":
        # Iterative MLE algorithm - Bias correction on beta estimate
        if algorithm == "Iter":
            # Starting values -- from JML to speed up the algorithm
                ## ( here, parameterUpdate will be a global variable in this algorithm)
            parameterUpdate = np.copy(estNocorr) # parameterUpdate will be a global variable in this algorithm
            # Initilizations
            funVal = -999999
            np.random.seed(1)
            iterlist = range(2000)
            # Iterative MLE starts here
            for sweep in iterlist:
                np.random.seed(sweep)
                stepsize = np.random.randint(50,90)
                newindices=np.arange(np.size(initials))
                np.random.shuffle(newindices)
                for start_i in np.arange(0,np.size(initials),stepsize):
                    end_i = start_i+stepsize
                    parameterSlice = parameterUpdate[newindices[start_i:end_i]]
                    res = sp.optimize.minimize(negLogLikeConcen,x0=parameterSlice, method='Newton-CG', jac=negJacobianConcen, hess=negHessConcen)
                    if start_i==0: 
                        funValDiff = np.abs(funVal - res.fun)
                        funVal = res.fun
                # Exit criteron 
                    ## ( this is a bit strict -- to ensure more precise estimates obtained
                    ##   by scipy optimization toolbox)
                if funValDiff<1e-15:
                    sucMess = res.success
                    funcLogLike = res.fun
                    break
                # print(sweep, stepsize, funValDiff,parameterUpdate[:4])
            estNocorr   = np.copy(parameterUpdate)
        # Fixed point iteration algorithm - Bias correction on beta estimate
        if algorithm == "FP":
            # Starting values -- from JML to speed up the algorithm
                ## ( here, FE_theta (FE given theta) is a global variable)
            theta_sv = (estNocorr[:numpara]) 
            FE_theta = (estNocorr[numpara:])
            # FP starts here
            res = sp.optimize.minimize(negLogLike_thetaFP,x0=theta_sv, method=meth_use, jac=negJacobianTheta, hess=negHessTheta)
            sucMess = res.success
            funcLogLike = res.fun
            estNocorr   = np.append(res.x, FE_theta).reshape(-1)
        # Bias correction on beta
        estTheta   = estNocorr[:numpara]-biasTerm(estNocorr)
        estEstcorr = np.append(estTheta,estNocorr[numpara:])
    # [> Organize parameters to output <]
    if bc_method == "likelihood":
        estFE = estLikeCorr[numpara:].reshape(-1)
    elif bc_method == "nocorr":
        estFE = estNocorr[numpara:].reshape(-1)
    else:
        estFE = estEstcorr[numpara:].reshape(-1)
    # [> Asymptotic standard errors <]
    failSEestimation = 0
    if bc_method == "likelihood" and seps==1 and drop_separation==False:
        if X is not None: X_tc = torch.from_numpy(X)
        if Z is not None: Z_tc = torch.from_numpy(Z)
        G_tc = torch.from_numpy(G)
        negHess_allparas = (torch.autograd.functional.hessian(negLogLike_tc, (torch.from_numpy(estLikeCorr).reshape(-1,1)).requires_grad_(True))[:,0,:,0]).numpy()
        try:
            model_paraSE = np.sqrt(np.diag(np.linalg.inv( negHess_allparas ))[:numpara]).reshape(-1)
        except:
            model_paraSE = np.copy(estTheta)*0+np.nan
            failSEestimation = 1
            sucMess=False
    else:
        model_paraSE = np.sqrt(np.diag(np.linalg.inv(negHessAll(estNocorr.reshape(-1,1))))[:numpara]).reshape(-1)
    end_time = time.time()
    # [> APE <]
    apeest=np.nan; apese=np.nan; apebc=np.nan
    if ape_compute==True:
        if bc_method == "likelihood": apeest=apeBC(estLikeCorr)
        if bc_method != "likelihood": apeest=apeCalc(estNocorr)[0]
        if failSEestimation ==1:
            apese=np.copy(apeest)*0+np.nan
        else:
            if bc_method == "likelihood" and seps==1 and drop_separation==False: 
                apese=apeSE(estLikeCorr)
            else:
                apese=apeSE(estNocorr)
    return estTheta, model_paraSE, estFE, sucMess, -funcLogLike, end_time - start_time, apeest, apese
#not oOo