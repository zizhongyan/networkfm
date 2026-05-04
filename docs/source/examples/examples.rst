Examples Gallery
================

This page is a **tutorial-style examples gallery** for the ``networkfm`` package.

In this page, each section starts with a short explanation of the
econometric object being estimated, then walks through the corresponding
``networkfm.fit()`` call, and finally points out what to look for in the
printed output.


Data requirements used throughout this page
------------------------------------------

All examples in this page use the same core data objects:

- ``G``: the dependent variable, i.e., the **network adjacency matrix** of size ``N × N``.
  The entry ``G[i, j]`` indicates whether there is a link from node ``i`` to node ``j``.

- ``X``: covariates entering the **directed-utility** component (if the model includes directed utility).
  It must be an array of size ``N × N × K_x``. The slice ``X[:, :, k]`` is the ``k``-th dyadic regressor.

- ``Z``: covariates entering the **mutual (reciprocity) utility** component (if the model includes mutual utility).
  It must be an array of size ``N × N × K_z``.
  Importantly, mutual-utility covariates must be **symmetric in (i, j)**, i.e.,
  ``Z[i, j, k] = Z[j, i, k]`` for all ``k``. (Directed-utility covariates ``X`` need not be symmetric.)

Data sources in the examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Example 1.** We use the bilateral trade data of
  Helpman, Melitz, and Rubinstein (2008). The package provides a ready-to-use
  loader ``networkfm.database.helpman09()``

- **Example 2.** We generate simulated networks using  ``networkfm.demo.GenData(...)``
  See the **API reference** for the generator’s full syntax and returned objects.



.. code-block:: python

    # Import the networkfm package
    import networkfm

Example 1: Trade network application (Helpman, Melitz and Rubinstein, 2008)
---------------------------------------------------------------------------

Example 1 below illustrates a typical applied workflow: 1. Loading an
example dataset (Helpman–Melitz–Rubinstein trade network), 2. Building
dyadic covariates, and 3. Running various estimations.

Data loading and preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the processed trade-network dataset available via
``networkfm.database.helpman09()``.

.. code-block:: python

    # Load the processed dataset shipped with the package
    data = networkfm.database.helpman09()
    
    # Network adjacency matrix (directed trade links), N = 158
    G = data["trade"].to_numpy().reshape(158, 158)
    
    # Covariates (N × N × K), here K = 11
    Covariates = data.loc[:, [
        "ln_distance", "border", "islands", "landlock", "legalsystem_same",
        "common_lang", "colonial", "cu", "fta",
        "religion_same_recoded", "constant"]].to_numpy().reshape(158, 158, 11)
    
    # Labels used in output tables (same ordering as the covariate array)
    varLabels = [
        "ln(Distance)", "Land border", "Island", "Landlock",
        "Legal", "Language", "Colonial ties", "Currency union",
        "FTA", "Religion", "Constant" ]

Example 1.1: Dyadic network formation with both directed and mutual utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This first example uses the **full model** with **directed utility**
*and* a **mutual (reciprocity) component**, implemented by setting
``directed=True`` and ``mutual=True``.

| **Step 1 — Choose the bias-correction method.**
| We set ``bc_method="likelihood"`` to apply the **likelihood-based
  correction** proposed in Yan, Li, and Zhang (2026). With this option,
  the reported table also includes **APE (average partial effects)
  estimates that are corrected** under the likelihood-based approach.

| **Step 2 — Choose the numerical algorithm.**
| We use ``algorithm="JML"``, i.e., joint maximum likelihood estimation
  (implemented via a Newton-type optimizer). As a numerical alternative,
  you may set ``algorithm="FP"`` to use a fixed-point iteration. In our
  experiments, the two algorithms converge to the same solution (up to
  numerical tolerance).

| **Practical note on separation.**
| In this dataset with 158 countries, one country (Congo) has
  **in-degree = 48** but **out-degree = 0**, which creates a classic
  **separation** issue in directed models: under standard estimation,
  the corresponding fixed effect is not identified.

A practical advantage of ``bc_method="likelihood"`` is that the
likelihood correction allows us to **retain this country in
estimation**, rather than dropping it due to separation.

.. code-block:: python

    Result = networkfm.fit(G, X=Covariates, X_names=varLabels,
                           Z=Covariates, Z_names=varLabels,
                           directed=True, mutual=True,
                           bc_method="likelihood", algorithm="JML")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
            DIRECTED NETWORK FORMATION MODEL WITH MUTUAL UTILITY
                  Bias correction method: Penalized likelihood
    --------------------------------------------------------------------------------
    Number of agents used in estimation: 158                Log-likelihood: -6291.62
    Algorithm: Joint MLE                              Time spent (seconds):    4.468
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Directed utility:
            ln(Distance)    -0.83123325    0.05398158   0.000    -0.93703   -0.72543
            Land boarder    -1.83110257    0.36697542   0.000    -2.55033   -1.11186
                  Island    0.546851362    0.17951677   0.002    0.195016   0.898686
                Landlock    -0.08819382    0.27085949   0.745    -0.61905   0.442663
                   Legal    0.102670776    0.07478693   0.170    -0.04390   0.249245
                Language    0.421560983    0.08778855   0.000    0.249504   0.593617
           Colonial ties    -1.97597871    1.27620877   0.122    -4.47722   0.525262
          Currency union    0.795944040    0.34742669   0.022    0.115022   1.476865
                     FTA    2.221560668    1.12023175   0.047    0.026018   4.417102
                Religion    0.437302700    0.14668143   0.003    0.149821   0.724783
                Constant    1.509189939    0.45377394   0.001    0.619838   2.398541
    Mutual utility:
            ln(Distance)    -0.29548201    0.08459119   0.000    -0.46127   -0.12969
            Land boarder    2.355238619    0.65996862   0.000    1.061766   3.648711
                  Island    -0.16321256    0.28318081   0.564    -0.71821   0.391793
                Landlock    0.898106722    0.48416593   0.064    -0.05081   1.847023
                   Legal    0.120213185    0.12604947   0.340    -0.12683   0.367257
                Language    -0.12786251    0.13655681   0.349    -0.39550   0.139775
           Colonial ties    3.570553019    2.06665101   0.084    -0.47987   7.620982
          Currency union    -0.06088408    0.63414685   0.924    -1.30374   1.181980
                     FTA    0.649661619    1.65187878   0.694    -2.58785   3.887178
                Religion    -0.24270061    0.23355345   0.299    -0.70044   0.215040
                Constant    2.978671559    0.38025236   0.000    2.233414   3.723928
    --------------------------------------------------------------------------------
    
    --------------------------------------------------------------------------------
    ---- AVERAGE PARTIAL EFFECTS (bias corrected) ----------------------------------
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Directed utility (average probability of unilateral linked):
            ln(Distance)    -0.07389809    0.00562788   0.000    -0.08492   -0.06286
            Land boarder    -0.17462455    0.03005695   0.000    -0.23353   -0.11571
                  Island    0.062732538    0.02119082   0.003    0.021200   0.104264
                Landlock    -0.00975685    0.02967606   0.742    -0.06791   0.048405
                   Legal    0.011405021    0.00828407   0.169    -0.00483   0.027640
                Language    0.047678114    0.01021317   0.000    0.027661   0.067694
           Colonial ties    -0.18582302    0.09282672   0.045    -0.36775   -0.00389
          Currency union    0.092383245    0.04148600   0.026    0.011074   0.173691
                     FTA    0.265009462    0.12668409   0.036    0.016721   0.513297
                Religion    0.038876977    0.01313299   0.003    0.013137   0.064616
                Constant    0.134169634    0.04063399   0.001    0.054531   0.213808
    Mutual utility (average probability of mutually linked):
            ln(Distance)    -0.02320193    0.00666782   0.001    -0.03627   -0.01013
            Land boarder    0.191990602    0.05371784   0.000    0.086708   0.297272
                  Island    -0.01274714    0.02196627   0.562    -0.05579   0.030304
                Landlock    0.072091606    0.03934259   0.067    -0.00501   0.149199
                   Legal    0.009431237    0.00986909   0.339    -0.00991   0.028773
                Language    -0.01001492    0.01064978   0.347    -0.03088   0.010857
           Colonial ties    0.287847688    0.14622765   0.049    0.001256   0.574439
          Currency union    -0.00477085    0.04930994   0.923    -0.10141   0.091871
                     FTA    0.052025278    0.13254079   0.695    -0.20774   0.311791
                Religion    -0.01905741    0.01833650   0.299    -0.05499   0.016880
                Constant    0.233892201    0.03162463   0.000    0.171911   0.295873
    --------------------------------------------------------------------------------
    Note: In directed utility, Land boarder, Island, Landlock, Legal, Language, 
          Colonial ties, Currency union, and FTA are dummy variables.
          In mutual utility, Land boarder, Island, Landlock, Legal, Language, 
          Colonial ties, Currency union, and FTA are dummy variables.
          The average partial effect of a dummy variable is calculated as the disc-
          rete change in probability as the dummy variable changes from 0 to 1.
    


Example 1.2: Dyadic network formation with directed utility only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now consider a **directed-only** specification by setting
``directed=True`` and ``mutual=False``. This turns off the mutual
(reciprocity) component and estimates a model with **directed utility
only**.

Here we use ``bc_method="estimator"``, which applies an
**estimator-based bias correction**, and we set ``algorithm="FP"`` to
estimate the model via **fixed-point iteration**.

Under ``bc_method="estimator"``, the reported table **does not apply the
likelihood-based correction to APEs**. If you would like APEs to be
corrected as well, use ``bc_method="likelihood"`` instead.

| **Practical note on separation.**
| Because Congo is separated in the directed network (out-degree = 0),
  it cannot be included under ``bc_method="estimator"`` (which does not
  use the likelihood correction). As a result, the estimation is
  conducted on **157 countries** in this example.

.. code-block:: python

    Result = networkfm.fit(G, X=Covariates, X_names=varLabels,
                       directed=True, mutual=False,
                       bc_method="estimator", algorithm="FP")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
            DIRECTED NETWORK FORMATION MODEL WITHOUT MUTUAL UTILITY
            Bias correction method: Analytical correction on estimator
    --------------------------------------------------------------------------------
    Number of agents used in estimation: 157                Log-likelihood: -6918.60
    Algorithm: Joint MLE with fixed point iterations  Time spent (seconds):    0.883
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Directed utility:
            ln(Distance)    -1.26697388    0.03856449   0.000    -1.34255   -1.19139
            Land boarder    -0.78526806    0.16966712   0.000    -1.11779   -0.45273
                  Island    0.610595330    0.13546335   0.000    0.345100   0.876089
                Landlock    0.384593889    0.18946383   0.042    0.013263   0.755924
                   Legal    0.191249236    0.05377164   0.000    0.085862   0.296636
                Language    0.522268963    0.06918872   0.000    0.386665   0.657871
           Colonial ties    0.394292595    0.53165032   0.458    -0.64768   1.436274
          Currency union    0.853621728    0.23903657   0.000    0.385133   1.322109
                     FTA    3.204452408    0.55275024   0.000    2.121117   4.287787
                Religion    0.408183132    0.10622318   0.000    0.199996   0.616369
                Constant    5.512787496    0.45536120   0.000    4.620325   6.405249
    --------------------------------------------------------------------------------
    
    --------------------------------------------------------------------------------
    ---- AVERAGE PARTIAL EFFECTS (uncorrected) -------------------------------------
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Directed utility (average probability of unilateral linked):
            ln(Distance)    -0.11025752    0.00570372   0.000    -0.12143   -0.09907
            Land boarder    -0.06402985    0.01396972   0.000    -0.09140   -0.03665
                  Island    0.053941856    0.01267544   0.000    0.029099   0.078784
                Landlock    0.032546034    0.01724098   0.059    -0.00124   0.066336
                   Legal    0.015702062    0.00476470   0.001    0.006363   0.025040
                Language    0.045414125    0.00652835   0.000    0.032619   0.058209
           Colonial ties    0.047242635    0.04901774   0.335    -0.04882   0.143312
          Currency union    0.077481764    0.02258488   0.001    0.033217   0.121745
                     FTA    0.317761849    0.04731161   0.000    0.225035   0.410487
                Religion    0.037870952    0.00948614   0.000    0.019279   0.056462
                Constant    0.338217942    0.04250648   0.000    0.254909   0.421526
    --------------------------------------------------------------------------------
    Note: Uncorrected average partial effects are displayed. Bias correction on the
          average partial effects is available with the likelihood correction setup
          (i.e., set bc_method='likelihood')
    Note: Network contains zero or full in-degree or out-degree agents;
          Dropped 1 out of 158 agents.
    Note: Land boarder, Island, Landlock, Legal, Language, Colonial ties, 
          Currency union, and FTA are dummy variables.
          The average partial effect of a dummy variable is calculated as the disc-
          rete change in probability as the dummy variable changes from 0 to 1.
    


Example 1.3: Dyadic network formation with mutual utility only (undirected network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we illustrate the **mutual-only** specification, which
corresponds to an **undirected network model**. We implement this by
setting ``directed=False`` and ``mutual=True``.

| **Undirected interpretation and automatic conversion.**
| When ``directed=False``, the model is interpreted as undirected. If
  the input adjacency matrix ``G`` is directed, ``networkfm``
  automatically converts it to an undirected network by keeping a link
  only when **both directions are present** (mutual ties). One-way links
  are set to 0.

**(a) No bias correction**

We start with ``bc_method="nocorr"``, i.e., no bias correction. Under
this option, not all countries necessarily enter the estimation; in this
dataset, **156 countries** are non-separated and can be estimated.

.. code-block:: python

    Result = networkfm.fit(G, Z=Covariates, Z_names=varLabels,
                       directed=False, mutual=True,
                       bc_method="nocorr", algorithm="JML")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
                        UNDIRECTED NETWORK FORMATION MODEL
                             Without bias correction
    --------------------------------------------------------------------------------
    Number of agents used in estimation: 156                Log-likelihood: -2993.91
    Algorithm: Joint MLE                              Time spent (seconds):    0.387
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Mutual utility:
            ln(Distance)    -1.55607450    0.06105220   0.000    -1.67573   -1.43641
            Land boarder    -0.65880678    0.24831632   0.008    -1.14548   -0.17213
                  Island    0.763674819    0.21169451   0.000    0.348774   1.178574
                Landlock    0.571137315    0.29912583   0.056    -0.01511   1.157394
                   Legal    0.315594169    0.08416752   0.000    0.150634   0.480554
                Language    0.606605357    0.10923078   0.000    0.392523   0.820686
           Colonial ties    0.794478957    0.69084771   0.250    -0.55951   2.148471
          Currency union    0.977501740    0.37324546   0.009    0.245977   1.709025
                     FTA    4.252552570    0.74232963   0.000    2.797660   5.707444
                Religion    0.643210621    0.16672017   0.000    0.316455   0.969965
                Constant    3.997925939    0.75915534   0.000    2.510057   5.485794
    --------------------------------------------------------------------------------
    
    --------------------------------------------------------------------------------
    ---- AVERAGE PARTIAL EFFECTS (uncorrected) -------------------------------------
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Mutual utility (average probability of mutually linked):
            ln(Distance)    -0.11910125    0.00664961   0.000    -0.13213   -0.10606
            Land boarder    -0.04839194    0.01757391   0.006    -0.08283   -0.01394
                  Island    0.060613455    0.01750997   0.001    0.026295   0.094931
                Landlock    0.044979989    0.02423727   0.063    -0.00252   0.092482
                   Legal    0.024142290    0.00650334   0.000    0.011396   0.036888
                Language    0.047134036    0.00878931   0.000    0.029907   0.064360
           Colonial ties    0.063437828    0.05730608   0.268    -0.04887   0.175752
          Currency union    0.078437326    0.03131070   0.012    0.017071   0.139803
                     FTA    0.363508447    0.06159339   0.000    0.242791   0.484225
                Religion    0.049231058    0.01291026   0.000    0.023928   0.074533
                Constant    0.305999498    0.05934053   0.000    0.189697   0.422301
    --------------------------------------------------------------------------------
    Note: Uncorrected average partial effects are displayed. Bias correction on the
          average partial effects is available with the likelihood correction setup
          (i.e., set bc_method='likelihood')
    Note: Network contains zero or full in-degree or out-degree agents;
          Dropped 2 out of 158 agents.
    Note: The input adjancecy matrix is asymmetric. Estimation is based on a modif-
          ied symmetric adjancecy matrix,  in which each entry equals 1 if the both
          agents are mutually linked, and otherwise zero.
    Note: Land boarder, Island, Landlock, Legal, Language, Colonial ties, 
          Currency union, and FTA are dummy variables.
          The average partial effect of a dummy variable is calculated as the disc-
          rete change in probability as the dummy variable changes from 0 to 1.
    


**(b) Likelihood-based correction** 

Next, we set ``bc_method="likelihood"`` to apply Yan, Li, and Zhang
(2026). In this case, **all 158 countries** can be included, and **APEs
are corrected** using the same likelihood-based procedure.

.. code-block:: python

    Result = networkfm.fit(G, Z=Covariates, Z_names=varLabels,
                       directed=False, mutual=True,
                       bc_method="likelihood", algorithm="JML")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
                        UNDIRECTED NETWORK FORMATION MODEL
                  Bias correction method: Penalized likelihood
    --------------------------------------------------------------------------------
    Number of agents used in estimation: 158                Log-likelihood: -2810.40
    Algorithm: Joint MLE                              Time spent (seconds):    0.976
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Mutual utility:
            ln(Distance)    -1.51680996    0.06005518   0.000    -1.63451   -1.39910
             Land border    -0.64251041    0.24505052   0.009    -1.12278   -0.16223
                  Island    0.742967668    0.20886576   0.000    0.333611   1.152323
                Landlock    0.557411625    0.29536565   0.059    -0.02147   1.136298
                   Legal    0.307110724    0.08305148   0.000    0.144338   0.469883
                Language    0.590682653    0.10780044   0.000    0.379404   0.801960
           Colonial ties    0.726339129    0.66889609   0.278    -0.58463   2.037308
          Currency union    0.958983034    0.36820867   0.009    0.237330   1.680635
                     FTA    4.118422897    0.72559459   0.000    2.696330   5.540515
                Religion    0.629813468    0.16456313   0.000    0.307286   0.952340
                Constant    3.874025847    0.74873484   0.000    2.406580   5.341471
    --------------------------------------------------------------------------------
    
    --------------------------------------------------------------------------------
    ---- AVERAGE PARTIAL EFFECTS (bias corrected) ----------------------------------
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Mutual utility (average probability of mutually linked):
            ln(Distance)    -0.11618563    0.00657147   0.000    -0.12906   -0.10330
             Land border    -0.04724829    0.01744088   0.007    -0.08143   -0.01306
                  Island    0.059007850    0.01729617   0.001    0.025109   0.092906
                Landlock    0.043927891    0.02387313   0.066    -0.00286   0.090716
                   Legal    0.023510764    0.00642162   0.000    0.010925   0.036096
                Language    0.045935752    0.00867668   0.000    0.028930   0.062941
           Colonial ties    0.057897624    0.05388635   0.283    -0.04771   0.163509
          Currency union    0.077022220    0.03090484   0.013    0.016451   0.137592
                     FTA    0.353827696    0.06102773   0.000    0.234219   0.473435
                Religion    0.048242878    0.01275563   0.000    0.023243   0.073242
                Constant    0.296745255    0.05851166   0.000    0.182068   0.411422
    --------------------------------------------------------------------------------
    Note: The input adjancecy matrix is asymmetric. Estimation is based on a modif-
          ied symmetric adjancecy matrix,  in which each entry equals 1 if the both
          agents are mutually linked, and otherwise zero.
    Note: Land border, Island, Landlock, Legal, Language, Colonial ties, 
          Currency union, and FTA are dummy variables.
          The average partial effect of a dummy variable is calculated as the disc-
          rete change in probability as the dummy variable changes from 0 to 1.
    



Example 2: Conditonal-likelihood methods in ``networkfm``
---------------------------------------------------------

This section illustrates two **conditional-likelihood** estimators that
are widely used in the network formation literature. Both originate from
dedicated external codebases, but ``networkfm`` exposes them through the
same **data interface** and the same **``networkfm.fit()`` syntax** as
the likelihood-based and estimator-based methods in the package. The
practical benefit is consistency: once you have constructed ``G``,
``X``, and/or ``Z``, you can switch between model classes and correction
methods without rewriting your workflow.

**Methods covered**

``networkfm`` provides access to two classic conditional-likelihood
approaches:

+-------------+-------------+-------------+-------------+-------------+
| Method      | Network     | Economic    | Key         | Upstream    |
|             | type        | feature     | reference   | imp         |
|             |             |             |             | lementation |
+=============+=============+=============+=============+=============+
| **Tetrad    | Undirected  | Degree      | Graham      | ``netrics`` |
| logit**     |             | he          | (2017,      | (Graham,    |
|             |             | terogeneity | *EMTR*)     | 2016)       |
+-------------+-------------+-------------+-------------+-------------+
| **Quadruple | Directed    | Send        | Jochmans    | ``          |
| logit**     | (no mutual  | er/receiver | (2018,      | quadlogit`` |
|             | utility)    | he          | *JBES*)     | (Hu et al., |
|             |             | terogeneity |             | 2026)       |
+-------------+-------------+-------------+-------------+-------------+

A small orientation guide:

-  **Tetrad logit** is designed for **undirected** models (in our
   notation: ``directed=False, mutual=True``).

-  **Quadruple logit** targets **directed** models **without** a mutual
   utility component (in our notation: ``directed=True, mutual=False``).

Artificial data for the examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To keep the examples fully reproducible, we generate an artificial
undirected network with ``N=100`` using ``networkfm.demo.GenData``:

.. code-block:: python

    G, Xmat, Zmat, _, _, _, _, _ = networkfm.demo.GenData(
                                     N=100, directed=False, mutual=True,
                                     specification="A1", seed=111)

Here: - ``G`` is the adjacency matrix, - ``Xmat`` / ``Zmat`` are
covariate arrays prepared in the shape expected by ``networkfm.fit``.

Example 2.1: Tetrad logit (undirected network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first run **tetrad logit** for an **undirected** specification. In
``networkfm``, this corresponds to: - ``directed=False``,
``mutual=True`` - ``bc_method="conditional"`` (activates the
conditional-likelihood engine)

Conditional-likelihood estimators are designed for identifying and
estimating **common parameters** without estimating the full set of
fixed effects. As a consequence, they **do not directly deliver APEs**,
and the output from ``bc_method="conditional"`` will not include APE
estimates.

.. code-block:: python

    Result = networkfm.fit(G, X=Xmat, Z=Zmat,
                           directed=False, mutual=True,
                           bc_method="conditional")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
                        UNDIRECTED NETWORK FORMATION MODEL
                             TETRAD LOGIT ESTIMATION
    --------------------------------------------------------------------------------
    Number of agents: 100                       Number of tetrads: 3921225
                                                Time spent (seconds): 16.508
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Mutual utility:
                      Z1    0.960354695    0.04421333   0.000    0.873700   1.047008
    --------------------------------------------------------------------------------
    


Example 2.2: Quadruple logit (directed, no mutual utility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we switch to a **directed** model **without** mutual utility,
which is the environment targeted by **quadruple logit**: -
``directed=True``, ``mutual=False`` - ``bc_method="conditional"``

Again, no APEs are reported under ``bc_method="conditional"``.

.. code-block:: python

    Result = networkfm.fit(G, X=Xmat, Z=Zmat,
                           directed=True, mutual=False,
                           bc_method="conditional")

.. parsed-literal::

    
    --------------------------------------------------------------------------------
    ---- ESTIMATION RESULTS --------------------------------------------------------
            DIRECTED NETWORK FORMATION MODEL WITHOUT MUTUAL UTILITY
                           QUADRUPLE LOGIT ESTIMATION
    --------------------------------------------------------------------------------
    Number of agents: 100                       Number of quadruples: 2173568
                                                Time spent (seconds): 23.259
    --------------------------------------------------------------------------------
    Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]
    --------------------------------------------------------------------------------
    Directed utility:
                      X1    0.003022374    0.04511431   0.947    -0.08539   0.091441
    --------------------------------------------------------------------------------
    


Example 2.3: Speed-up via precomputing tetrad/quadruple indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For both tetrad logit and quadruple logit, the major computation is the
construction of the **tetrad/quadruple index sets**. When you repeatedly
estimate models on the same ``N`` (e.g., in Monte Carlo simulations,
robustness checks, or bootstrap loops), it is usually worth
**precomputing indices once** and passing them into ``networkfm.fit()``
via ``indices`` option.

**Step 1: Generate indices**

.. code:: python

   # Precompute tetrad indices (example: N=100)
   tetrad_idx = networkfm.netrics.generate_tetrad_indices(N=100)

   # Precompute quadruple indices (example: N=100)
   quad_idx = networkfm.quadlogit.generate_quad_indices(N=100)

**Step 2: Pass indices to ``networkfm.fit``**

.. code:: python

   # Tetrad logit with precomputed indices
   networkfm.fit(
           G, X=Xmat, Z=Zmat,
           directed=False, mutual=True,
           bc_method="conditional",
           indices=tetrad_idx)

   # Quadruple logit with precomputed indices
   networkfm.fit(
           G, X=Xmat, Z=Zmat,
           directed=True, mutual=False,
           bc_method="conditional",
           indices=quad_idx)


**References.** 

- Graham, Bryan S. (2016). “netrics: a Python 3.7 package for econometric analysis of networks,” (Version 0.0.1) [Computer program]. Available at https://github.com/bryangraham/netrics (Accessed 04 October 2018) 

- Graham, Bryan S. (2017). “An econometric model of link formation with degree heterogeneity,” *Econometrica* 85 (4): 1033 - 1063 

- Helpman, Elhanan, Marc Melitz, and Yona Rubinstein (2008). “Estimating Trade Flows: Trading Partners and Trading Volumes.” *Quarterly Journal of Economics* 123: 441–487. 

- Hu, Shiran and Guo, Muyang and Cheng, Xinran and Zhou, Xuan. (2026). “Quadlogit: Quadruple Logit Regression for Network Formation Models,” (Version 0.2.1) [Computer program]. Available at https://github.com/HuNeedHelp/quadlogit (Accessed 30 April 2026) 

- Jochmans, Koen. (2018). “Semiparametric analysis of network formation.” *Journal of Business & Economic Statistics* 36, no. 4 (2018): 705-713.


**Credit and provenance.** 

The ``quadlogit`` package is developed by Hu
et al. (2026). It is a Python re-implementation and optimization of
Jochmans (2018)’s original MATLAB code, with substantial speed
improvements and support for multiple covariates. It also provides
utilities for **precomputing quadruple indices**, including ready-to-use
index files for cases with ``N ≤ 100``, which can greatly reduce
runtime.

Hu et al. (2026) were undergraduate students of the ``networkfm``
maintainer (Zizhong Yan) at the time of development, and they built
``quadlogit`` as a research-side project outside of class.


Where to go next
----------------
-  For the full API reference, see the docs page `API reference on Read
   the Docs <https://networkfm.readthedocs.io/en/latest/api/api.html>`__.
