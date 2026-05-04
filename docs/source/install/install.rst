Getting Started
===============

.. contents::
   :local:
   :depth: 2


Prerequisites
-------------

- Python 3.8+ is recommended.
- ``networkfm`` relies on standard scientific Python libraries, including **NumPy** and **PyTorch**. 

.. note::
    PyTorch is required, because ``networkfm`` uses automatic differentiation (autograd) functionality to support numerical optimization routines. You can install PyTorch by following the official instructions at: https://pytorch.org/. **GPU/CUDA is not required.** For most users, the **CPU-only** version of PyTorch is sufficient.


Installation options
--------------------

1. Install from PyPI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the latest released version from PyPI:

.. code-block:: bash

   pip install networkfm


2. Install the development version from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/zizhongyan/networkfm.git
   cd networkfm
   pip install -e .

The ``-e`` option performs an *editable install*, meaning local code changes take effect immediately without reinstalling.

3. Install from a local copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you downloaded the source code as a ZIP file from `github.com/zizhongyan/networkfm <https://github.com/zizhongyan/networkfm>`_, and unzipped it locally, you may install the package from the repository root:

.. code-block:: bash

   cd /path/to/networkfm
   pip install -e .
 



Main entry: ``networkfm.fit()``
-------------------------------

Most users will interact with the package through a single function:

.. code-block:: python

   networkfm.fit(
       G, X=None, Z=None, 
       directed=True, mutual=True,
       bc_method="likelihood", algorithm="JML" 
       drop_separation=False, 
       ape=True,
       X_names=None, Z_names=None, 
       silent=False, 
       indices=None, sv=None
   )


Inputs: adjacency matrix ``G``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``G`` is the dependent variable: the network adjacency matrix. 
  It must be a square ``N × N`` array-like object (e.g., a 2D NumPy array), where ``G[i, j]`` indicates whether there is a link from node ``i`` to node ``j``.
  Self-links are typically excluded. If ``G[i, i]`` is nonzero, the package will ignore or overwrite diagonal entries depending on the model configuration.
- For **directed** models, ``G`` is treated as directed: ``G[i, j]`` and ``G[j, i]`` may differ.
- For **undirected** (mutual-only) models, ``G`` is interpreted as an undirected network. 


Inputs: regressors ``X`` and ``Z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``X`` contains dyad-level covariates entering the *directed utility* component. 
  It must be shaped as ``N × N × Kx`` (a 3D NumPy array), where ``Kx`` is the number of directed-utility regressors.
  No symmetry is required: in general, ``X[i, j, :]`` may differ from ``X[j, i, :]``.
- ``X`` is **required** when ``directed=True``. If ``directed=False``, ``X`` is ignored and can be left as ``None``.

- ``Z`` contains dyad-level covariates entering the *mutual (reciprocity) utility* component. 
  It must be shaped as ``N × N × Kz`` (a 3D NumPy array), where ``Kz`` is the number of mutual-utility regressors.
  In this component, covariates must be **symmetric**: ``Z[i, j, :] == Z[j, i, :]`` (up to numerical tolerance).
- ``Z`` is **required** when ``mutual=True``. If ``mutual=False``, ``Z`` is ignored and can be left as ``None``.

- Variable names (optional).
  ``X_names`` and ``Z_names`` are optional lists of strings used only for labeling output tables. 
  Their lengths must match ``Kx`` and ``Kz`` respectively. If omitted, the package uses default labels.


Model selection: ``directed`` and ``mutual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The arguments ``directed`` and ``mutual`` determine the model class:

- ``directed=True, mutual=True``:
  directed utility **plus** a mutual utility component (main setting of Yan, Li, and Zhang, 2026).

- ``directed=True, mutual=False``:
  directed utility only (no mutual component).

- ``directed=False, mutual=True``:
  mutual utility only, which corresponds to an **undirected network** model.

.. note::

   When ``directed=False, mutual=True`` (mutual-only / undirected), the package will internally work with an
   undirected version of the network. If the input adjacency matrix ``G`` is asymmetric (directed), the package will automatically converts it to an undirected network by keeping a link only when **both directions are present** (mutual ties). One-way links are set to 0. This makes the input requirements convenient for users:
   you can pass the same raw ``G`` you use in directed network analyses.


Bias correction method: ``bc_method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``bc_method="nocorr"``:  
  uncorrected fixed-effects MLE (baseline; subject to incidental-parameter bias and feasibility issues in sparse networks). This method provide uncorrected plug-in APE estimates.
    
- ``bc_method="likelihood"``:  
  likelihood correction proposed in Yan, Li, and Zhang (2026). This is the **recommended** option in most applications.  
  When `ape=True`, this option also produces **debiased APE estimates** as described in the paper.
    
- ``bc_method="estimator"``:  
  estimator-based bias correction (subject to feasibility issues in sparse networks). This method provide uncorrected plug-in APE estimates.
    
- ``bc_method="conditional"``:  
  conditional-likelihood methods used as benchmarks: tetrad logit (undirected) and quadruple logit (directed utility only). These methods do not provide plug-in APE estimates.


Estimation algorithm: ``algorithm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The argument ``algorithm`` controls the numerical routine used for likelihood-based estimation:

- ``algorithm="JML"`` (default): joint maximum likelihood estimation using a Newton-type optimizer.
- ``algorithm="FP"``: fixed-point iteration.

In typical use, both algorithms converge to the same solution; ``FP`` can be useful as a robust numerical
alternative when Newton steps are slow or unstable in challenging designs.


Separated nodes: ``drop_separation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse networks can generate **separated** nodes that make fixed-effects estimation infeasible. A common example in directed networks is a node with out-degree zero (and/or in-degree zero), for which the corresponding degree effect is not identified under uncorrected likelihood.

The option ``drop_separation`` controls how the sample is handled:

- If ``drop_separation=True``, the function iteratively removes separated nodes (e.g., nodes with no outgoing links
  or no incoming links) until the remaining subnetwork contains no separated nodes.  
  Because dropping nodes can create *new* separated nodes, the procedure is applied **iteratively** until it reaches
  a stable connected set.

- Default behavior is ``drop_separation=False``. In practice:

  - ``bc_method="likelihood"`` is designed to remain feasible under the paper’s likelihood correction and can
    typically keep separated nodes in the estimation sample.
  - ``bc_method="conditional"`` also remains feasible without dropping separated nodes in the same way.


.. note::

   By contrast, when ``bc_method`` is ``"nocorr"`` or ``"estimator"``, the package will enforce separation handling for feasibility. Concretely, even if ``drop_separation=False`` is passed, the routine will internally behave as if ``drop_separation=True`` so that estimation can proceed.


Output control: ``silent``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``silent`` (default ``False``): if set to ``True``, the function will not print the summary table to screen.
The fitted result object is still returned.


Average partial effects: ``ape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ape`` (default ``True``): if set to ``True``, the function computes average partial effects (APEs) in addition to
parameter estimates. With ``bc_method="conditional"``, APEs are not available (conditional likelihood does not identify the same plug-in APE objects in this framework).


Tetrad/quadruple ``indices``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional-likelihood methods (for ``bc_method="conditional"``) rely on enumerating **tetrads** (undirected) or **quadruples** (directed utility only). Constructing these index sets can be the dominant runtime cost. If you repeatedly estimate models with the same number of nodes ``N`` (e.g., Monte Carlo simulations, robustness checks, or bootstrap loops), it is usually worth **precomputing indices once** and passing them into ``networkfm.fit()`` via the ``indices`` option.

To generate indices, see API refrence page for ``networkfm.netrics.generate_tetrad_indices`` and ``networkfm.quadlogit.generate_quad_indices`` functions.

See also **Example 2.3** in the examples gallery.


Starting values: ``sv``
~~~~~~~~~~~~~~~~~~~~~~~

``sv`` allows users to provide starting values for optimization. If omitted, the default initialization uses
all zeros (or an internally constructed default consistent with the chosen model/method).



Returned object and key attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`networkfm.fit` returns a **result object** that collects the main outputs from estimation—parameter estimates, standard errors, and (when requested and supported) average partial effects (APEs). The most frequently used attributes are:

- ``self.paras``  
  Estimated **common parameters** (e.g., regression coefficients). This corresponds to the common-parameter vector :math:`\theta`.

- ``self.se``  
  Asymptotic **standard errors** for the entries of ``self.paras``.

- ``self.N``  
  The number of nodes **actually used in estimation** (after any separation handling).

- ``self.Nold``  
  The number of nodes in the **original input network**. When separation handling is active (e.g., under sparse networks), it is possible that ``self.N < self.Nold`` because some nodes are dropped iteratively.

- ``self.success``  
  A convergence/termination flag for the estimation routine. ``1`` indicates successful convergence (or successful termination of the iteration); other values indicate that the optimizer/iteration did not complete as intended.

The following outputs are **not available** under ``bc_method="conditional"``:

- ``self.feparas``  
  Estimated **fixed effects** (degree heterogeneity parameters), including sender and receiver effects when the model is directed.

- ``self.fun``  
  The final value of the objective function at the reported solution. Under ``algorithm="JML"``, this is the final (possibly corrected/penalized) log-likelihood value used by the optimizer.

- ``self.ape``  
  Estimated **average partial effects (APEs)** when ``ape=True``. For discrete regressors, APEs are reported as finite changes; for continuous regressors, they are reported as average marginal effects.

- ``self.apese``  
  Asymptotic **standard errors** for the APE estimates in ``self.ape``.

For the API reference of ``networkfm.fit()``, `please consult here: <https://networkfm.readthedocs.io/en/latest/api/api.html#estimation-interface>`_.

Where to go next
----------------

- For worked examples of ``networkfm.fit()``, see the `Examples section <https://networkfm.readthedocs.io/en/latest/examples/examples.html>`_.



