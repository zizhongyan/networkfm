"""
Model fitting interface for `networkfm`.

The main public entry point is :class:`networkfm.fit`.

Created on Thu Sep 25 2025

Last modified on Sat Feb 14 2026

Authors: Zizhong Yan
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import sys
import numpy as np
import scipy as sp
import time
from scipy import stats
from ..utils.networkMLE import networkFits
from ..lib.netrics import tetrad_logit
from ..lib.quadlogit import fit as quadlogitfit
import warnings 
warnings.filterwarnings('ignore')
#----------------------------------------------------------
# Class of various network regression
#----------------------------------------------------------
class fit:
    """
    This is the main entry point of the package. Most users will call
    :func:`networkfm.fit` to estimate model parameters and average partial effects.

    Parameters
    ----------
    G : array_like, shape (N, N)
        Network adjacency matrix (dependent variable). ``G[i, j]`` indicates
        whether there is a link from node ``i`` to node ``j``.

        - Must be a square 2D array-like object (NumPy array).
        - Self-links are typically excluded. Diagonal entries ``G[i, i]`` are
          ignored (and may be overwritten internally, depending on the model).

        Interpretation depends on ``directed`` / ``mutual``:

        - Directed models: ``G`` is treated as directed (``G[i, j]`` and
          ``G[j, i]`` may differ).
        - Mutual-only (undirected) models: the estimation uses an undirected
          version of the network (see Notes).

    X : array_like, shape (N, N, Kx), optional
        Dyad-level covariates entering the **directed-utility** component.

        - Required when ``directed=True``.
        - Ignored when ``directed=False``.
        - No symmetry is required: generally ``X[i, j, :]`` may differ from
          ``X[j, i, :]``.

    Z : array_like, shape (N, N, Kz), optional
        Dyad-level covariates entering the **mutual (reciprocity)** component.

        - Required when ``mutual=True``.
        - Ignored when ``mutual=False``.
        - Must be symmetric (up to numerical tolerance):
          ``Z[i, j, :] == Z[j, i, :]``.

    directed : bool, default True
        Whether to include a directed utility component.

    mutual : bool, default True
        Whether to include a mutual utility component. When
        ``directed=False`` and ``mutual=True``, this corresponds to a mutual-only
        or undirected network model.

    bc_method : {"uncorr", "likelihood", "estimator", "conditional"}, default "likelihood"
        Bias-correction / benchmark method.

        - ``"uncorr"``: uncorrected fixed-effects MLE.
        - ``"likelihood"``: likelihood correction. 
          When ``ape=True``, produces debiased APEs under the paper’s correction.
        - ``"estimator"``: estimator-based bias correction (may face feasibility
          issues in sparse networks). Produces uncorrected plug-in APEs.
        - ``"conditional"``: conditional-likelihood benchmarks (tetrad logit for
          undirected; quadruple logit for directed-utility only). Plug-in APEs
          are not available under this option.

    algorithm : {"JML", "FP"}, default "JML"
        Numerical routine used for likelihood-based estimation.

        - ``"JML"``: joint maximum likelihood (Newton-type optimizer).
        - ``"FP"``: fixed-point iteration.

        In typical cases both converge to the same solution; ``"FP"`` may be a
        more robust alternative when Newton steps are slow or unstable.

    drop_separation : bool, default False
        How to handle separated nodes in sparse networks.

        - If ``True``, iteratively drops separated nodes (e.g., out-degree zero
          and/or in-degree zero in directed settings) until the remaining
          subnetwork contains no separated nodes.
        - If ``False``, the function keeps all nodes **unless** feasibility
          requires otherwise (see Notes).

    ape : bool, default True
        If ``True``, compute average partial effects (APEs) in addition to
        parameter estimates. Not available when ``bc_method="conditional"``.

    X_names : list of str, optional
        Names for the ``Kx`` directed-utility regressors (labels only). If
        provided, length must equal ``Kx``.

    Z_names : list of str, optional
        Names for the ``Kz`` mutual-utility regressors (labels only). If
        provided, length must equal ``Kz``.

    silent : bool, default False
        If ``True``, do not print the summary table. The fitted result object is
        still returned.

    indices : object, optional
        Precomputed tetrad/quadruple indices for conditional-likelihood methods
        (``bc_method="conditional"``). Precomputing can greatly reduce runtime
        when repeatedly estimating models with the same ``N`` (e.g., Monte Carlo,
        bootstrap, robustness loops).

        See also: :func:`networkfm.netrics.generate_tetrad_indices`,
        :func:`networkfm.quadlogit.generate_quad_indices`.

    sv : object, optional
        Starting values for optimization / iteration. If omitted, the routine
        uses a default initialization (typically zeros or an internally
        constructed default consistent with the chosen specification).

    Returns
    -------
    result : object
        A fitted result object collecting estimation outputs (parameters, standard
        errors, and optional APEs). Commonly used attributes include:

        - ``result.paras`` : ndarray
            Estimated **common parameters** (e.g., regression coefficients).
        - ``result.se`` : ndarray
            Asymptotic standard errors for entries of ``result.paras``.
        - ``result.N`` : int
            Number of nodes actually used in estimation (after any dropping).
        - ``result.Nold`` : int
            Number of nodes in the original input network.
        - ``result.success`` : int
            Convergence/termination flag (``1`` typically indicates success).

        The following outputs are **not available** under
        ``bc_method="conditional"``:

        - ``result.feparas`` : fixed effects (degree heterogeneity parameters)
        - ``result.fun`` : final objective function value
        - ``result.ape`` / ``result.apese`` : APEs and their standard errors

    Notes
    -----
    **Model class selection**

    The pair (``directed``, ``mutual``) determines the model class:

    - ``directed=True, mutual=True``: directed utility + mutual component.
    - ``directed=True, mutual=False``: directed utility only.
    - ``directed=False, mutual=True``: mutual-only (undirected network model).

    **Mutual-only / undirected input convenience**

    When ``directed=False`` and ``mutual=True``, the routine works with an
    undirected version of the network. If the input ``G`` is asymmetric, it is
    converted internally by keeping a link only when **both directions are
    present** (mutual ties). One-way links are set to 0. This allows you to pass
    the same raw ``G`` used in directed analyses.

    **Separation handling**

    Sparse networks can generate separated nodes that make fixed-effects
    estimation infeasible under some methods. Even if ``drop_separation=False``
    is passed, the routine may enforce separation handling for feasibility when
    ``bc_method`` is ``"uncorr"`` or ``"estimator"`` (i.e., it will behave as if
    ``drop_separation=True`` internally).

    See Also
    --------
    networkfm.netrics.generate_tetrad_indices
        Precompute tetrad indices for undirected conditional likelihood.
    networkfm.quadlogit.generate_quad_indices
        Precompute quadruple indices for directed utility only conditional likelihood.
 
    """
    def __init__(self, G, X=None, Z=None, directed=True, mutual=True,
                  bc_method="likelihood", drop_separation=False, algorithm="JML", 
                  X_names=None, Z_names=None, sv=None, silent=False,ape=True,
                  indices=None):
        #----------------------------------------------------------
        # Preparations
        #----------------------------------------------------------
        # [> Check compatibility of arguments <]
        if bc_method != "likelihood" and bc_method != "nocorr" and bc_method != "estimator" and bc_method != "conditional" and bc_method != "estimatorChain":
            sys.exit("Error: `bc_method` option is not correctly defined. Please see the helpfile: help(fit)")
        if algorithm != "JML" and algorithm != "FP" and algorithm != "Iter":
            sys.exit("Error: `algorithm` option is not correctly defined. Please see the helpfile: help(fit)")
        # [> Check compatibility of input variables <]
        # Check adjacency matrix
        if G.shape[0]!=G.shape[1]: 
            sys.exit("Error: adjacency matrix G is not a square matrix.")
        if G.ndim!=2: 
            sys.exit("Error: adjacency matrix G is not a 2d NumPy array. Please see the helpfile: help(fit)")
        if np.all(np.unique(G)==np.array([0,1]))!=True:
            sys.exit("Error: adjacency matrix G is not correctly defined, or is not binary.")
        # Check covariates
        if X is not None: 
            if X.ndim == 2 and X.shape[0] == X.shape[1]: X=X[:,:,None] 
            if X.ndim!=3: sys.exit("Error: covariate X is not a 3d NumPy array. Please see the helpfile: help(fit)")
        if Z is not None: 
            if Z.ndim == 2 and Z.shape[0] == Z.shape[1]: Z=Z[:,:,None] 
            if Z.ndim!=3: sys.exit("Error: covariate Z is not a 3d NumPy array. Please see the helpfile: help(fit)")
            if np.allclose(Z.transpose(1,0,2),Z)!=True:
                sys.exit("Error: at least one component of covariate Z is not symmetric in i and j.")
        # [> Check compatibility for conditional likelihood methods <]
        if directed==True and mutual==True and bc_method == "conditional":
            sys.exit("Error: `bc_method=conditional` is compatibile with the model with both directed and mutual utilities")
        #if directed==True and mutual==False and bc_method == "conditional":

        #if directed==False and mutual==True and bc_method == "conditional":
        # [> For undirected formation, if G is not symmetric, change it to a symmetric one. <]
        nonsymmetricG=0
        if directed==False and np.allclose(G,G.T)==False:
            nonsymmetricG = 1 
            G = G.T+G
            G[G==1] = 0
            G[G==2] = 1
        # [> Nodes with zero/full in-degree or out-degree will be dropped if not using likelihood correction. <]
        self.N = np.shape(G)[0]
        self.Nold = np.copy(self.N)
        if bc_method == "nocorr": drop_separation=True
        if bc_method == "estimator": drop_separation=True
        if bc_method == "conditional": drop_separation=False
        if directed==False and Z is not None: X=Z
        if directed==False and X is not None: Z=X
        if directed==True and mutual==False and X is not None: Z=X
        if directed==True and mutual==False and Z is not None: X=Z
        # Mark if data has separtion:
        if np.unique(G.sum(axis=1))[0]==0 or np.unique(G.sum(axis=0))[0]==0 or np.unique(G.sum(axis=0))[-1]==self.N-1 or np.unique(G.sum(axis=1))[-1]==self.N-1: 
            self.separated=1
        else:
            self.separated=0
        # Drop nodes until the network has no separtion
        if drop_separation==True:
            # if np.unique(G.sum(axis=1))[0]==0 or np.unique(G.sum(axis=0))[0]==0 or np.unique(G.sum(axis=0))[-1]==self.N-1 or np.unique(G.sum(axis=1))[-1]==self.N-1:
            #     while np.unique(G.sum(axis=1))[0]==0 or np.unique(G.sum(axis=0))[0]==0 or np.unique(G.sum(axis=0))[-1]==self.N-1 or np.unique(G.sum(axis=1))[-1]==self.N-1:
            #         for axisdim in (0,1):
            #             G_update = G[G.sum(axis=axisdim)!=0,:]
            #             if X is not None: X = X[G.sum(axis=axisdim)!=0,:,:]
            #             if X is not None: X = X[:,G.sum(axis=axisdim)!=0,:]
            #             if Z is not None: Z = Z[G.sum(axis=axisdim)!=0,:,:]
            #             if Z is not None: Z = Z[:,G.sum(axis=axisdim)!=0,:]
            #             G = G_update[:,G.sum(axis=axisdim)!=0]
            #         self.N = np.shape(G)[0]
            iteration = 1
            active_nodes = np.arange(G.shape[0])

            while True:
                Nofnode = G.shape[0]
                if Nofnode <= 1:
                    print("Network is empty or trivial after trimming.")
                    break
                    
                out_degree = np.sum(G, axis=1)
                in_degree = np.sum(G, axis=0)
                
                # 分类识别节点
                zero_out_mask = (out_degree == 0)
                zero_in_mask = (in_degree == 0)
                full_out_mask = (out_degree == Nofnode - 1)
                full_in_mask = (in_degree == Nofnode - 1)
                
                # 汇总需要剔除的节点 (使用逻辑或，处理节点可能同时满足多项条件的情况)
                remove_mask = zero_out_mask | zero_in_mask | full_out_mask | full_in_mask
                num_remove = np.sum(remove_mask)
                
                if num_remove == 0:
                    if silent is False: print(f"Iteration {iteration}: 0 nodes removed. Network trimming complete.")
                    if silent is False: print(f"Remaining valid nodes: {Nofnode}")
                    break
                    
                # 统计各类型数量
                num_zero_out = np.sum(zero_out_mask)
                num_zero_in = np.sum(zero_in_mask)
                num_full_out = np.sum(full_out_mask)
                num_full_in = np.sum(full_in_mask)
                
                if silent is False: print(f"Iteration {iteration}: Removed {num_remove} nodes.")
                if silent is False: print(f"  - Zero out-degree : {num_zero_out}")
                if silent is False: print(f"  - Zero in-degree  : {num_zero_in}")
                if silent is False: print(f"  - Full out-degree : {num_full_out}")
                if silent is False: print(f"  - Full in-degree  : {num_full_in}")
                
                # 剔除节点，更新邻接矩阵和索引
                keep_mask = ~remove_mask
                G = G[keep_mask][:, keep_mask]
                if X is not None: X = X[keep_mask][:, keep_mask, :]
                if Z is not None: Z = Z[keep_mask][:, keep_mask, :]
                iteration = iteration+1
            self.N = np.shape(G)[0]
                    

        # [> Change all input variables to float 64bit <]
        if G.dtype != 'float64': G = G.astype('float64')
        if X is not None:  
            if X.dtype != 'float64': X = X.astype('float64')
        if Z is not None:  
            if Z.dtype != 'float64': Z = Z.astype('float64')
        # [> Check whether there are dummy regressors -- mark for generating correct APE <]
        if X is not None:  
            dummyIndicatorX = np.zeros(X.shape[2],dtype='int')
            if ape==True:
                for kx in range(X.shape[2]):
                    if np.unique(X[:,:,kx]+np.eye(self.N)).size==2:
                        dummyIndicatorX[kx] = np.array_equal(np.unique(X[:,:,kx]), np.array([0,1]))
        if Z is not None:  
            dummyIndicatorZ = np.zeros(Z.shape[2],dtype='int')
            if ape==True:
                for kz in range(Z.shape[2]):
                    if np.unique(Z[:,:,kz]+np.eye(self.N)).size==2:
                        dummyIndicatorZ[kz] = np.array_equal(np.unique(Z[:,:,kz]), np.array([0,1]))
        #----------------------------------------------------------
        # Estimation
        #----------------------------------------------------------
        self.success = 1
        if directed==False and bc_method == "conditional":
            ape = False
            start_time = time.time()
            self.paras, var, self.tetrad_frac_TL, Nchoose4, self.success = tetrad_logit(D=G, W=list(np.moveaxis(X, 2, 0)), silent=True, dtcon=indices)
            self.se = np.sqrt(var).reshape(-1)
            self.paras = self.paras.reshape(-1)
            end_time = time.time() - start_time
        if directed==True and mutual==False and bc_method == "conditional":
            ape = False
            start_time = time.time()
            output = quadlogitfit(G, X,  silent=True, indices=indices)
            self.se = output.se
            self.paras = output.paras
            Nchoose4 = output.Nchoose4
            end_time = time.time() - start_time

        if bc_method != "conditional":
            try:
                self.paras, self.se, self.feparas, self.success, self.fun, end_time, self.ape, self.apese = networkFits(G,X,Z,directed,mutual,bc_method,drop_separation,algorithm,sv,silent,ape,self.separated,dummyIndicatorX,dummyIndicatorZ)
            except:
                self.success = 0
                sys.exit("Estimation failed.")
        #----------------------------------------------------------
        # Broadcasting
        #----------------------------------------------------------
        if silent is False: 
            print("",)
            print("--------------------------------------------------------------------------------")            
            print("---- ESTIMATION RESULTS --------------------------------------------------------")             
            if directed==False:
                print("                    UNDIRECTED NETWORK FORMATION MODEL")
            if directed==True and mutual==True:
                print("        DIRECTED NETWORK FORMATION MODEL WITH MUTUAL UTILITY",)
            if directed==True and mutual==False:
                print("        DIRECTED NETWORK FORMATION MODEL WITHOUT MUTUAL UTILITY")
            if bc_method=="likelihood":
                print("              Bias correction method: Penalized likelihood")
            if bc_method=="estimator" or bc_method=="estimatorChain":
                print("        Bias correction method: Analytical correction on estimator")
            if bc_method=="nocorr":
                print("                         Without bias correction")
            if bc_method != "conditional":
                print("--------------------------------------------------------------------------------")            
                print("Number of agents used in estimation: %3s                Log-likelihood: %8.2f" % (self.N,self.fun))
                if algorithm=="JML":
                    print("Algorithm: Joint MLE                              Time spent (seconds): %8.3f" % end_time)
                if algorithm=="FP":
                    print("Algorithm: Joint MLE with fixed point iterations  Time spent (seconds): %8.3f" % end_time)
                if algorithm=="Iter":
                    print("Algorithm: Iterative MLE (concentration scheme)   Time spent (seconds): %8.3f" % end_time)
            if directed==False  and bc_method == "conditional":
                print("                         TETRAD LOGIT ESTIMATION")
                print("--------------------------------------------------------------------------------")            
                print("Number of agents: %3s                       Number of tetrads: %3s" % (self.N,Nchoose4))
                print("                                            Time spent (seconds): %5.3f" % end_time)
            if directed==True and mutual==False and bc_method == "conditional":
                print("                       QUADRUPLE LOGIT ESTIMATION")
                print("--------------------------------------------------------------------------------")            
                print("Number of agents: %3s                       Number of quadruples: %3s" % (self.N,Nchoose4))
                print("                                            Time spent (seconds): %5.3f" % end_time)
            print("--------------------------------------------------------------------------------")            
            print("Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
            print("--------------------------------------------------------------------------------")            
            if directed==False and Z_names is not None: X_names=Z_names
            if directed==False and X_names is not None: Z_names=X_names
            if directed==True and mutual==False and X_names is not None: Z_names=X_names
            if directed==True and mutual==False and Z_names is not None: X_names=Z_names
            if directed==False: print("Mutual utility:")
            if directed==True: print("Directed utility:")
            if X is not None:
                if X_names is None:
                    X_names = []
                    for kk in range(0,np.shape(X)[2]):
                        X_names.append("X" + str(kk+1))
            if Z is not None:
                if Z_names is None:
                    Z_names = []
                    for kk in range(0,np.shape(Z)[2]):
                        Z_names.append("Z" + str(kk+1))
            if directed==False: 
                for kk in range(0,np.shape(Z)[2]):
                    print("%20s%15s%14s%8.3f%12s%11s" % (Z_names[kk][:15],
                                                str(self.paras[kk])[:11],
                                                str(self.se[kk])[:10],
                                                2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                                str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                                str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            if directed==True: 
                for kk in range(0,np.shape(X)[2]):
                    print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                str(self.paras[kk])[:11],
                                                str(self.se[kk])[:10],
                                                2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                                str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                                str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            if directed==True and mutual==True: 
                print("Mutual utility:")
                for kk in range(np.shape(X)[2],np.shape(X)[2]+np.shape(Z)[2]):
                    print("%20s%15s%14s%8.3f%12s%11s" % (Z_names[kk-np.shape(X)[2]][:15],
                                                str(self.paras[kk])[:11],
                                                str(self.se[kk])[:10],
                                                2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                                str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                                str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            print("--------------------------------------------------------------------------------")     
            if ape==True:
                print("")
                print("--------------------------------------------------------------------------------")     
                apedisp = self.ape
                if bc_method=="likelihood":       
                    print("---- AVERAGE PARTIAL EFFECTS (bias corrected) ----------------------------------")
                else:
                    print("---- AVERAGE PARTIAL EFFECTS (uncorrected) -------------------------------------")
                print("--------------------------------------------------------------------------------")            
                print("Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
                print("--------------------------------------------------------------------------------")            
                if directed==False: X=Z; X_names=Z_names
                if directed==False: print("Mutual utility (average probability of mutually linked):")
                if directed==True: print("Directed utility (average probability of unilateral linked):")
                if X is not None:
                    if X_names is None:
                        X_names = []
                        for kk in range(0,np.shape(X)[2]):
                            X_names.append("X" + str(kk+1))
                if Z is not None:
                    if Z_names is None:
                        Z_names = []
                        for kk in range(0,np.shape(Z)[2]):
                            Z_names.append("Z" + str(kk+1))
                for kk in range(0,np.shape(X)[2]):
                    print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                str(apedisp[kk])[:11],
                                                str(self.apese[kk])[:10],
                                                2*sp.stats.norm.sf(abs(apedisp[kk]/self.apese[kk])),
                                                str(apedisp[kk]-1.9599*self.apese[kk])[:8]  ,
                                                str(apedisp[kk]+1.9599*self.apese[kk])[:8]  ))
                if directed==True and mutual==True: 
                    print("Mutual utility (average probability of mutually linked):")
                    for kk in range(np.shape(X)[2],np.shape(X)[2]+np.shape(Z)[2]):
                        print("%20s%15s%14s%8.3f%12s%11s" % (Z_names[kk-np.shape(X)[2]][:15],
                                                    str(apedisp[kk])[:11],
                                                    str(self.apese[kk])[:10],
                                                    2*sp.stats.norm.sf(abs(apedisp[kk]/self.apese[kk])),
                                                    str(apedisp[kk]-1.9599*self.apese[kk])[:8]  ,
                                                    str(apedisp[kk]+1.9599*self.apese[kk])[:8]  ))
                print("--------------------------------------------------------------------------------")            
                if bc_method!="likelihood":
                    print("Note: Uncorrected average partial effects are displayed. Bias correction on the")
                    print("      average partial effects is available with the likelihood correction setup")
                    print("      (i.e., set bc_method='likelihood')")
            if self.Nold != self.N:
                print("Note: Network contains zero or full in-degree or out-degree agents;")
                print("      Dropped %1.0f out of %1.0f agents." % (self.Nold-self.N, self.Nold))
            if nonsymmetricG == 1:
                print("Note: The input adjancecy matrix is asymmetric. Estimation is based on a modif-")
                print("      ied symmetric adjancecy matrix,  in which each entry equals 1 if the both")
                print("      agents are mutually linked, and otherwise zero.")
            if ape==True:
                if directed==False: 
                    if np.any(dummyIndicatorZ==1):
                        print("Note: ",end="")
                        numDis = np.sum(dummyIndicatorZ)
                        count = 1
                        for kkk in range(Z.shape[2]):
                            if dummyIndicatorZ[kkk]==True: 
                                if numDis>1:
                                    if count%7==0 and count!=numDis:
                                        print("\n      %s" % (Z_names[kkk][:15]),end=", ")
                                    if count%7!=0 and count!=numDis:
                                        print("%s" % (Z_names[kkk][:15]),end=", ")
                                    if count==numDis:
                                        print("and %s" % (Z_names[kkk][:15]),end=" ")
                                    count=count+1
                                if numDis==1:
                                        print("%s" % (z_names[kkk][:15]),end=" ")
                        if numDis==1: print("is a dummy variable.")
                        if numDis>1:  print("are dummy variables.")
                        print("      The average partial effect of a dummy variable is calculated as the disc-")
                        print("      rete change in probability as the dummy variable changes from 0 to 1.")
                if directed==True and mutual==False: 
                    if np.any(dummyIndicatorX==1):
                        print("Note: ",end="")
                        numDis = np.sum(dummyIndicatorX)
                        count = 1
                        for kkk in range(X.shape[2]):
                            if dummyIndicatorX[kkk]==True: 
                                if numDis>1:
                                    if count%7==0 and count!=numDis:
                                        print("\n      %s" % (X_names[kkk][:15]),end=", ")
                                    if count%7!=0 and count!=numDis:
                                        print("%s" % (X_names[kkk][:15]),end=", ")
                                    if count==numDis:
                                        print("and %s" % (X_names[kkk][:15]),end=" ")
                                    count=count+1
                                if numDis==1:
                                        print("%s" % (X_names[kkk][:15]),end=" ")
                        if numDis==1: print("is a dummy variable.")
                        if numDis>1:  print("are dummy variables.")
                        print("      The average partial effect of a dummy variable is calculated as the disc-")
                        print("      rete change in probability as the dummy variable changes from 0 to 1.")
                if directed==True and mutual==True: 
                    if np.any(dummyIndicatorX==1):
                        print("Note: In directed utility, ",end="")
                        numDis = np.sum(dummyIndicatorX)
                        count = 1
                        for kkk in range(X.shape[2]):
                            if dummyIndicatorX[kkk]==True: 
                                if numDis>1:
                                    if count%6==0 and count!=numDis:
                                        print("\n      %s" % (X_names[kkk][:15]),end=", ")
                                    if count%6!=0 and count!=numDis:
                                        print("%s" % (X_names[kkk][:15]),end=", ")
                                    if count==numDis:
                                        print("and %s" % (X_names[kkk][:15]),end=" ")
                                    count=count+1
                                if numDis==1:
                                        print("%s" % (X_names[kkk][:15]),end=" ")
                        if numDis==1: print("is a dummy variable.")
                        if numDis>1:  print("are dummy variables.")
                    if np.any(dummyIndicatorZ==1):
                        print("      In mutual utility, ",end="")
                        numDis = np.sum(dummyIndicatorZ)
                        count = 1
                        for kkk in range(Z.shape[2]):
                            if dummyIndicatorZ[kkk]==True: 
                                if numDis>1:
                                    if count%6==0 and count!=numDis:
                                        print("\n      %s" % (Z_names[kkk][:15]),end=", ")
                                    if count%6!=0 and count!=numDis:
                                        print("%s" % (Z_names[kkk][:15]),end=", ")
                                    if count==numDis:
                                        print("and %s" % (Z_names[kkk][:15]),end=" ")
                                    count=count+1
                                if numDis==1:
                                        print("%s" % (X_names[kkk][:15]),end=" ")
                        if numDis==1: print("is a dummy variable.")
                        if numDis>1:  print("are dummy variables.")
                    if np.any(dummyIndicatorX==1) or np.any(dummyIndicatorZ==1):
                        print("      The average partial effect of a dummy variable is calculated as the disc-")
                        print("      rete change in probability as the dummy variable changes from 0 to 1.")
            print("")
#not oOo