=============
API Reference
=============

This page documents the main public APIs of ``networkfm``. Throughout, inputs are NumPy arrays (unless stated otherwise).

.. contents::
   :local:
   :depth: 2


Data generation for examples and testing
----------------------------------------

``networkfm.database.helpman09``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.database.helpman09()

   Load the pre-processed Helpman–Melitz–Rubinstein trade network dataset shipped with
   ``networkfm``.

   :returns:
      A ``pandas.DataFrame`` containing the pre-processed dataset read from the package’s
      ``database/helpman09.csv``.
   :rtype:
      pandas.DataFrame

   **Notes**

   Users typically reshape columns from the returned DataFrame into:

   - ``G``: adjacency matrix, shape ``(N, N)``
   - ``X`` and/or ``Z``: covariate arrays, shape ``(N, N, K)``


``networkfm.demo.GenData``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.demo.GenData(N, directed=True, mutual=True, specification="A1", seed=111)

   Generate one synthetic dyadic network dataset from the package’s DGP.

   :param int N:
      Number of agents (nodes).

   :param bool directed:
      Whether to generate a directed utility component.

   :param bool mutual:
      Whether to generate a mutual/reciprocity component.

   :param str specification:
      DGP setting identifier (e.g., ``"A1"``, ``"A2"``, ``"A3"``, ``"B1"``, ``"B2"``, ``"B3"``),
      controlling fixed-effect heterogeneity and other DGP constants.

   :param int seed:
      Random seed for reproducibility.

   :returns:
      A tuple ``(G, X, Z, density, degree, transitivity, separation, trueParameter)`` where

      - ``G`` is a NumPy adjacency matrix of shape ``(N, N)``
      - ``X`` is a NumPy array of shape ``(N, N, 1)`` (directed covariate in the current DGP)
      - ``Z`` is a NumPy array of shape ``(N, N, 1)`` (mutual covariate in the current DGP)
      - ``density``, ``degree``, ``transitivity`` are network summary statistics
      - ``separation`` is an indicator for separation in the generated sample
      - ``trueParameter`` stores DGP parameters (used in truth/APE simulation workflows)
   :rtype:
      tuple


``networkfm.demo.NetworkGenDataMC``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.demo.NetworkGenDataMC(N, directed=True, mutual=True, specification="A1", mc_sweep=1000)

   Generate a Monte Carlo bank of synthetic datasets and pre-allocated result containers.

   :param int N:
      Number of agents.

   :param bool directed:
      Whether to include directed utility.

   :param bool mutual:
      Whether to include mutual utility.

   :param str specification:
      DGP setting identifier.

   :param int mc_sweep:
      Number of Monte Carlo replications.

   :returns:
      A long tuple of NumPy arrays used for Monte Carlo experiments. Key elements include:

      - ``G_bank``: shape ``(N, N, mc_sweep)``, adjacency matrices across replications
      - ``X_bank``: shape ``(N, N, mc_sweep)``, directed covariate bank (single covariate)
      - ``Z_bank``: shape ``(N, N, mc_sweep)``, mutual covariate bank (single covariate)
      - ``density_mc``, ``degree_mc``, ``transitivity_mc``, ``separation_mc``: shape ``(mc_sweep,)``,
        network statistics across replications
      - ``thetaEst_mc``, ``thetaSE_mc`` and related arrays: pre-allocated containers for estimates,
        standard errors, and coverage rates (parameters and APEs)
   :rtype:
      tuple

   **Notes**

   This function is designed for simulation pipelines: it returns both generated datasets and
   empty arrays for storing Monte Carlo outputs.


Tetrad/quadruple indices
------------------------

``networkfm.netrics.generate_tetrad_indices``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.netrics.generate_tetrad_indices(N, full_set=True)

   Precompute index mappings required by **tetrad logit** (conditional likelihood for
   undirected networks).

   :param int N:
      Number of agents.

   :param bool full_set:
      If ``True``, also compute a dyad→tetrads dictionary (more memory/time).
      If ``False``, compute only the tetrad→dyads mapping (recommended for repeated estimation
      and resampling loops).

   :returns:
      A list ``[tetrad_to_dyads_indices, dyad_to_tetrads_dict]`` where

      - ``tetrad_to_dyads_indices`` is a NumPy array of shape ``(n_tetrads, 6)``, giving for each tetrad
        the six dyad indices in a vectorized ``(N, N)`` layout;
      - ``dyad_to_tetrads_dict`` is a dictionary mapping each dyad to a set/list of tetrads containing it
        (or ``None`` if ``full_set=False``).
   :rtype:
      list

   **Usage**

   Precompute indices once and pass them via ``indices=...`` in
   ``networkfm.fit(..., bc_method="conditional")`` to avoid repeated combinatorial work.


``networkfm.quadlogit.generate_quad_indices``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.quadlogit.generate_quad_indices(N)

   Load and adapt precomputed indices for **quadruple logit** (conditional likelihood for
   directed networks without mutual utility).

   :param int N:
      Number of agents. Index files are shipped for ``N=100``. For ``N < 100``, this function
      filters the ``N=100`` indices down to the subset relevant for the requested ``N``.

   :returns:
      A tuple ``(rearragement_index, permutations)`` of NumPy arrays (dtype ``intp``), both 0-based.
   :rtype:
      tuple

   **Notes**

   This helper is intended for repeated estimation on the same ``N`` (Monte Carlo, robustness checks).
   The returned indices can be passed via the ``indices`` argument of
   ``networkfm.fit(..., bc_method="conditional")``.





Estimation interface
--------------------

``networkfm.fit``
~~~~~~~~~~~~~~~~~

.. py:function:: networkfm.fit(G, X=None, Z=None, directed=True, mutual=True, bc_method="likelihood", algorithm="JML", drop_separation=False, ape=True, X_names=None, Z_names=None, silent=False, indices=None, sv=None)

   Fit dyadic network formation models with degree heterogeneity, with optional
   bias correction and (when available) average partial effects (APEs).

   :param numpy.ndarray G:
      Network adjacency matrix (dependent variable), shape ``(N, N)``.
      Typically binary (0/1). The diagonal is ignored (and is commonly set to 0).

   :param numpy.ndarray X:
      Covariates entering the **directed utility** component, shape ``(N, N, Kx)``.
      Required when ``directed=True``. The directed covariates may be asymmetric in
      ``(i, j)`` (i.e., ``X[i, j, :]`` need not equal ``X[j, i, :]``).
      Use ``X=None`` when ``directed=False``.

   :param numpy.ndarray Z:
      Covariates entering the **mutual/reciprocity** component, shape ``(N, N, Kz)``.
      Required when ``mutual=True``. Mutual covariates must be symmetric in ``(i, j)``
      (i.e., ``Z[i, j, :] == Z[j, i, :]``).
      Use ``Z=None`` when ``mutual=False``.

   :param bool directed:
      Whether to include a directed (sender→receiver) utility component.

   :param bool mutual:
      Whether to include a mutual/reciprocity component. The undirected-network case is
      obtained by setting ``directed=False`` and ``mutual=True``.

   :param str bc_method:
      Bias-correction / estimation method. One of:

      - ``"nocorr"``: uncorrected fixed-effects MLE (no bias correction).
        (In some code examples you may see ``"nocorr"`` used as a synonym.)
      - ``"likelihood"``: likelihood correction (main method of the paper); supports
        APE debiasing when ``ape=True``.
      - ``"estimator"``: estimator-based correction (for comparison).
      - ``"conditional"``: conditional-likelihood methods (tetrad/quadruple logit);
        fixed effects and APE outputs are not available under this option.

   :param str algorithm:
      Numerical algorithm for the likelihood-based estimators.

      - ``"JML"``: joint maximum likelihood via a Newton-type optimizer.
      - ``"FP"``: fixed-point iteration (numerical alternative).

   :param bool drop_separation:
      Whether to iteratively drop *separated* nodes in sparse networks. When enabled,
      the routine removes nodes with no links (or no in-/out-links in directed settings),
      recomputes degrees, and repeats until no separated nodes remain.

      **Important:** For ``bc_method="nocorr"`` and ``bc_method="estimator"``, separation
      dropping may be enforced internally to ensure estimability, regardless of the user
      setting. The likelihood correction (and conditional likelihood) is designed to
      remain operational in sparse settings and may allow keeping all nodes.

   :param bool ape:
      If ``True``, compute APEs when supported. APE outputs are **not available** under
      ``bc_method="conditional"``.

   :param list[str] X_names:
      Optional names for the directed covariates, used in printed summaries.

   :param list[str] Z_names:
      Optional names for the mutual covariates, used in printed summaries.

   :param bool silent:
      If ``True``, suppress printing the summary table. The fitted result object is
      still returned.

   :param object indices:
      Optional precomputed index objects used by conditional likelihood
      (``bc_method="conditional"``). See :func:`networkfm.netrics.generate_tetrad_indices`
      and :func:`networkfm.quadlogit.generate_quad_indices`.

   :param dict sv:
      Optional starting values for optimization. If omitted, the routine initializes all
      parameters at zero.

   :returns:
      A fitted **result object** storing parameter estimates, standard errors, and (when
      applicable) APEs.
   :rtype:
      object


