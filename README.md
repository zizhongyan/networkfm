<h1 align="center">
<img src="https://raw.githubusercontent.com/zizhongyan/StataImproved/master/pictures/networkfmlogo6.png" width="600">
</h1><br>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-v.0.8.1-orange)](https://pypi.org/project/networkfm/)
[![Docs](https://readthedocs.org/projects/NetworkFm/badge/?version=latest)](https://networkfm.readthedocs.io/en/latest/)
[![PyTorch](https://img.shields.io/badge/PyTorch-blue?logo=pytorch&logoColor=white)](https://pytorch.org/)

**`NetworkFm`** is a Python package for the econometric analysis of the **dyadic network formation models** with **degree heterogeneity**. The package accompanies the paper:

> Yan, Zizhong; Li, Jingrong; Zhang, Yi (2026). *Penalized Likelihood for Dyadic Network Formation Models with Degree Heterogeneity.*  (arXiv:2605.00771)

**Quick links**
- **Paper**: [https://arxiv.org/abs/2605.00771](https://arxiv.org/abs/2605.00771)
- **Project Documentation**:
  - **Documentation homepage**: [NetworkFm.readthedocs.io](https://networkfm.readthedocs.io/)
  - **Tutorials**: [Getting started](https://networkfm.readthedocs.io/en/latest/install/install.html)
  - **Examples and API**: [Examples](https://networkfm.readthedocs.io/en/latest/examples/examples.html) and [API reference](https://networkfm.readthedocs.io/en/latest/api/api.html)


## What does `NetworkFm` do?

**`NetworkFm`** provides a single, unified interface for fitting several workhorse **dyadic network formation** models:

1. **Directed utility and mutual (reciprocity) utility** (the main setting of the paper)
2. **Directed utility only** (no mutual component), and
3. **Mutual utility only**, which corresponds to **undirected networks**.

A key motivation for `NetworkFm` is that empirical network data are often **sparse**. In sparse networks, standard fixed-effects estimation routines can break down. Not only because of the classic **incidental parameter problem**, but also because estimation may be **infeasible** due to _existence / separation–type_ issues. In practice, this can turn a seemingly standard application into “the optimizer never converges” or “some fixed effects are not identified,” even when the model is conceptually appropriate.

The package implements the **likelihood correction** proposed in **Yan, Li, and Zhang (2026)**. This correction is designed to address **both** challenges at once:

- it delivers **bias reduction** for common parameters in the presence of high-dimensional degree heterogeneity, and
- it remains operational in sparse-network settings under the paper’s **convex-hull feasibility condition**, substantially enlarging the set of datasets for which estimation is practically usable.


In addition, once the likelihood correction is applied, `NetworkFm` provides **debiased plug-in estimates of average partial effects (APEs)**, for continuous and discrete regressors.


### Estimators included

- **Likelihood correction (Yan, Li, and Zhang, 2026)**  
  - Implements the paper’s likelihood-based adjustment for all three model classes above.
  - Addresses the incidental-parameter bias **and** the sparse network problem (the convex-hull condition).
  - Reports **debiased APEs** for continuous and discrete/finite-change effects.

- **Estimator-based correction (benchmarks / comparisons)**
  - For **directed + mutual utility**: estimator-based correction following Yan, Li, and Zhang (2026).
  - For **directed utility only**: estimator-based correction following Yan et al. (2019).
  - For **mutual utility only / undirected networks**: estimator-based correction following Graham (2017).

- **Conditional likelihood method (benchmarks / comparisons)**
    - **Tetrad logit** for undirected networks (Graham, 2017), via a vendored copy of `netrics` (Graham, 2016).
    - **Quadruple logit** for directed networks without mutual utility (Jochmans, 2018), via the bundled `quadlogit` package by Hu et al. (2026).
    - `NetworkFm` exposes these methods through the same `networkfm.fit()` interface and data conventions, so users can compare likelihood correction, estimator correction, and conditional likelihood without rewriting data pipelines.

--- 


## Installation 

1. **Installation via `pip install`** (easiest and recommended)
```bash
pip install networkfm
````

2. **Install the development version** via GitHub
```bash
git clone https://github.com/zizhongyan/networkfm.git
cd networkfm
pip install -e .
```

3. **Manual installation**

Downloaded this Github source code as a ZIP file and unzipped it locally, run the following from the repository root:
```bash
cd /path/to/networkfm
pip install -e .
```

## Quick start

```python
import networkfm

res = networkfm.fit(
    G, # the network adjacency matrix
    X, Z, # covariates in directed and mutual utilities respectively
    directed=True, mutual=True,   # directed + mutual utility
    bc_method="likelihood", # likelihood correction
    algorithm="JML" 
)

print(res.paras)  # parameter estimates
print(res.se)     # standard errors
print(res.ape)    # average partial effects
```

Visit the [project's documentation](https://networkfm.readthedocs.io/) for tutorials, examples gallery, and API reference.

## Dependencies

- **Python ≥ 3.8**
- **PyTorch** is required (CPU-only is fine). It is used for **autograd**.
- The repository bundles (vendors) several research codes used as benchmarks:
  - `ipt` (Graham, 2016) for common logit routines,
  - `netrics` (Graham, 2016) for tetrad logit,
  - `quadlogit` the package Hu et al. (2026) based on the paper Jochmans (2018).

## Citing `NetworkFm`

Please cite the paper and the software when using this package for implementing the penalized likelihood methods.

- **Paper**: Yan, Zizhong; Li, Jingrong; Zhang, Yi (2026). “*Penalized Likelihood for Dyadic Network Formation Models with Degree Heterogeneity*”. *arXiv e-prints*, arXiv:2605.00771.

- **Software**: Yan, Zizhong (2026). “*NetworkFm: A Python Package for Dyadic network formation models with degree heterogeneity*”   (Version 0.8.1) [Computer software]. Available at `https://github.com/zizhongyan/networkfm`  (Accessed 14 February 2026) .

- If using the tetrad logit estimation, please kindly cite Graham (2016) and Graham (2017). 

## References
- Graham, Bryan S. (2016). “netrics: a Python 3.7 package for econometric analysis of networks,” (Version 0.0.1) [Computer program]. Available at https://github.com/bryangraham/netrics (Accessed 04 October 2018) 
- Graham, Bryan S. (2017). “An econometric model of link formation with degree heterogeneity,” *Econometrica* 85 (4): 1033 - 1063 
- Helpman, Elhanan, Marc Melitz, and Yona Rubinstein (2008). “Estimating Trade Flows: Trading Partners and Trading Volumes.” *Quarterly Journal of Economics* 123: 441–487. 
- Hu, Shiran, Muyang Guo, Xinran Cheng and Xuan Zhou. (2026). “Quadlogit: Quadruple Logit Regression for Network Formation Models,” (Version 0.2.1) [Computer program]. Available at https://github.com/HuNeedHelp/quadlogit (Accessed 30 April 2026) 
- Jochmans, Koen. (2018). “Semiparametric analysis of network formation.” *Journal of Business & Economic Statistics* 36, no. 4 (2018): 705-713.
- Yan, Ting, Binyan Jiang, Stephen E. Fienberg and Chenlei Leng. (2019). “Statistical inference in a directed network model with covariates.” *Journal of the American Statistical Association* 114(526), 857-868.


---

**Code maintainer:** Zizhong Yan, Institute for Economic and Social Research (IESR), Jinan University, Guangzhou, China.  Email: `helloyzz@gmail.com`; Jingrong Li, College of Economics and Management, South China Agricultural University, Guangzhou, China. Email: `Jingronglijnu@gmail.com`. 
