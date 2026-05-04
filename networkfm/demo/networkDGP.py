"""
Python functions for data generating process
of various network formation models

By Zizhong Yan (helloyzz@gmail.com)

Created on Thu Sep 25 2025
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import numpy as np
import scipy as sp
from scipy import stats
import logging
try: 
    import networkx as nx 
except Exception as e:
    logging.warning(e)
    logging.warning("To use networkfm command, please install networkfm package.")
#----------------------------------------------------------
# Network formation: 
# ---  Generate `mc_sweep` copies of data for running Monte Carlo
#----------------------------------------------------------
def NetworkGenDataMC(N, directed=True, mutual=True,
                     specification="A1",mc_sweep=1000):
    numParas = 1
    if directed==True and mutual==True: numParas = 2
    G_bank = np.zeros((N,N,mc_sweep))+np.nan
    X_bank = np.zeros((N,N,mc_sweep))+np.nan
    Z_bank = np.zeros((N,N,mc_sweep))+np.nan
    density_mc      = np.zeros((mc_sweep))+np.nan
    degree_mc       = np.zeros((mc_sweep))+np.nan
    transitivity_mc = np.zeros((mc_sweep))+np.nan
    separation_mc   = np.zeros((mc_sweep))+np.nan
    # Empty matrices for save MC results
    numMethods = 10 # up to Ten estimation methods will be used
    thetaEst_mc = np.zeros((numMethods,numParas,mc_sweep))+np.nan
    thetaSE_mc  = np.copy(thetaEst_mc)
    CP95_mc     = np.copy(thetaEst_mc)
    CP90_mc     = np.copy(thetaEst_mc)
    apethetaEst_mc = np.copy(thetaEst_mc)
    apethetaSE_mc  = np.copy(thetaEst_mc)
    apeCP95_mc     = np.copy(thetaEst_mc)
    apeCP90_mc     = np.copy(thetaEst_mc)
    networkFE_succ_mc = np.zeros((2,mc_sweep))+1
    clogit_succ_mc = np.zeros((mc_sweep))+np.nan
    # Generate data
    for s in (range(mc_sweep)):
        (G,X,Z,density,degree,transitivity,separation,_) = GenData(N, directed=directed, mutual=mutual,specification=specification,seed=s)
        G_bank[:,:,s] = G
        X_bank[:,:,s] = X[:,:,0]
        Z_bank[:,:,s] = Z[:,:,0]
        density_mc[s]      = density     
        degree_mc[s]       = degree      
        transitivity_mc[s] = transitivity
        separation_mc[s]   = separation
    return G_bank,X_bank,Z_bank,density_mc,degree_mc,transitivity_mc,separation_mc,thetaEst_mc,thetaSE_mc,CP95_mc,CP90_mc,networkFE_succ_mc,clogit_succ_mc,apethetaEst_mc,apethetaSE_mc,apeCP95_mc,apeCP90_mc
#----------------------------------------------------------
# Network formation DGP: 
# ---  based on Graham (2017) and has been extended to
#      directed networks
#----------------------------------------------------------
def GenData(N, directed=True, mutual=True,specification="A1",seed=111):
    np.random.seed(seed)
    # Covariates in mutual utility
    M = np.random.randint(low=0,high=2,size=N)+0.0
    M[M==0]=-1
    Zmat=np.zeros((N,N))
    for ii in range(N):
        for jj in range(N):
            Zmat[ii,jj]= M[ii]*M[jj]
    np.fill_diagonal(Zmat,0)
    # Covariates in directed utility
    Xmat = np.random.randint(low=0,high=2,size=(N,N))
    np.fill_diagonal(Xmat,0)
    # FE
    if specification=="A1": aL = -0.5;  aH=-0.5
    if specification=="A2": aL = -1.0;  aH=-1.0
    if specification=="A3": aL = -2.0;  aH=-2.0
    if specification=="B1": aL = -2/3;  aH=-1/6
    if specification=="B2": aL = -7/6;  aH=-2/3
    if specification=="B3": aL = -13/6; aH=-5/3
    if specification=="A1" or specification=="A2" or specification=="A3":
        lambda0 = 1
        lambda1 = 1
    if specification=="B1" or specification=="B2" or specification=="B3":
        lambda0 = 1/4
        lambda1 = 3/4
    Vi1 = np.random.beta(a=lambda0,b=lambda1,size=N) - lambda0/(lambda0+lambda1)
    Vi2 = np.random.beta(a=lambda0,b=lambda1,size=N) - lambda0/(lambda0+lambda1)
    Alphai = aL*(M==-1)+aH*(M==1) + Vi1
    Gammai = aL*(M==-1)+aH*(M==1) + Vi2
    if directed==True and mutual==False: Gammai = np.copy(Alphai)
    #if directed==True and mutual==True:  Alphai[-1] = 0; Gammai[-1] = 0
    Alphai_mat = np.tile(Alphai.reshape(1,N),N).reshape(N,N).T
    Alphaj_mat = np.tile(Alphai.reshape(1,N),N).reshape(N,N)
    Gammai_mat = np.tile(Gammai.reshape(1,N),N).reshape(N,N).T
    Gammaj_mat = np.tile(Gammai.reshape(1,N),N).reshape(N,N)
    # Generate dependent variable
    if directed==False:
        err_mat = sp.stats.logistic.rvs( loc=0, scale=1, size=(N,N) )
        err_mat = np.triu(err_mat,1)+np.triu(err_mat,1).T
        ystar = Zmat + Alphai_mat + Alphaj_mat - err_mat
        G=np.copy(ystar)
        G[ystar >= 0] = 1
        G[ystar < 0]  = 0
        np.fill_diagonal(G, 0)
    if directed==True and mutual==False: 
        Xmat = Zmat # This DGP gives undirected network --- following Jochmans paper
        err_mat = sp.stats.logistic.rvs( loc=0, scale=1, size=(N,N) )
        ystar = Xmat  + Alphai_mat + Gammaj_mat - err_mat
        G=np.copy(ystar)
        G[ystar >= 0] = 1
        G[ystar < 0]  = 0
        np.fill_diagonal(G, 0)
    if directed==True and mutual==True:  
        G = np.zeros((N,N))
        Gij = np.zeros((N,N))
        Gji = np.zeros((N,N))
        for rep in range(1000):
            error_mat = sp.stats.logistic.rvs( loc=0, scale=1, size=(N,N) )
            ystar_ij = Xmat + G.T*Zmat + Alphai_mat + Gammaj_mat - error_mat
            Gij[ystar_ij>=0]=1
            Gij[ystar_ij<0] =0  
            G = np.tril(G, -1) + np.triu(Gij, 1)
            ystar_ji = Xmat + G.T*Zmat + Alphai_mat + Gammaj_mat - error_mat
            Gji[ystar_ji>=0]=1
            Gji[ystar_ji<0] =0  
            G = np.tril(Gji, -1) + np.triu(G, 1) 
    # Check identification's condition.
    while G.sum(0)[-1]==0 or G.sum(1)[-1]==0:
        G2 =np.copy(G)
        G2[1:,1:]=G[:-1,:-1]
        G2[0,0]=G[-1,-1]
        G2[0,1:]=G[-1,:-1]
        G2[1:,0]=G[:-1,-1]

        Xmat2 =np.copy(Xmat)
        Xmat2[1:,1:]=Xmat[:-1,:-1]
        Xmat2[0,0]=Xmat[-1,-1]
        Xmat2[0,1:]=Xmat[-1,:-1]
        Xmat2[1:,0]=Xmat[:-1,-1]

        Zmat2 =np.copy(Zmat)
        Zmat2[1:,1:]=Zmat[:-1,:-1]
        Zmat2[0,0]=Zmat[-1,-1]
        Zmat2[0,1:]=Zmat[-1,:-1]
        Zmat2[1:,0]=Zmat[:-1,-1]

        Alphai2 = np.copy(Alphai)
        Alphai2[1:]=Alphai[:-1]
        Alphai2[0]=Alphai[-1]

        Gammai2 = np.copy(Gammai)
        Gammai2[1:]=Gammai[:-1]
        Gammai2[0]=Gammai[-1]

        G=np.copy(G2)
        Xmat=np.copy(Xmat2)
        Zmat=np.copy(Zmat2)
        Alphai=np.copy(Alphai2)
        Gammai=np.copy(Gammai2)

    separation=0
    if (np.sum(G.astype('i'),axis=0)==N).sum()!=0:
        separation=1
    if (np.sum(G.astype('i'),axis=0)==0).sum()!=0:
        separation=1
    if (np.sum(G.astype('i'),axis=1)==N).sum()!=0:
        separation=1
    if (np.sum(G.astype('i'),axis=1)==0).sum()!=0:
        separation=1
    # Network stats.
    density= nx.density(nx.DiGraph(G))
    degree= np.mean(np.sum(G,axis=0))
    transitivity= nx.transitivity(nx.DiGraph(G))
    # Keep the true parameters for simulatiing the true APE
    if directed==False: trueParameter = np.hstack((1,Alphai))
    if directed==True and mutual==False: trueParameter = np.hstack((1,Alphai,Gammai))
    if directed==True and mutual==True: trueParameter = np.hstack((1,1,Alphai,Gammai))
    return G,Xmat.reshape(N,N,1),Zmat.reshape(N,N,1),density,degree,transitivity,separation,trueParameter
