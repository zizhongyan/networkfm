# -*- coding: utf-8 -*-
"""
networkfm
===========

networkfm is a Python package for the econometric analysis on 
the dyadic network formation models with degree heterogeneity.

Project links
-------------
- Source code & replication materials (GitHub): https://github.com/zizhongyan/networkfm
- Documentation: https://networkfm.readthedocs.io/

Main entry point
----------------
- :class:`networkfm.fit` — fit network formation models and compute APEs.

Notes
-----
- For full replication workflow of Yan et al. (2026), see the replication notebooks in the GitHub repository.
- For usage examples and model-by-model demonstrations, see the documentation tutorials and examples gallery.

Version
-------
{version}
"""

__version__ = "0.8.1"

# Inject version into the module docstring shown by help(networkfm)
__doc__ = (__doc__ or "").format(version=__version__)

from .api.networkModels import fit
from . import database
from . import demo
from .lib import netrics
from .lib import quadlogit


