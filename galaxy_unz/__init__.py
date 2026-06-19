"""
galaxy_unz
==========
Far-infrared fine-structure line fitting tool for measuring
gas-phase metallicity without the effects of extinction.

Typical usage
-------------
>>> from galaxy_unz import DataLoader, ModelGrid, BayesianFitter
>>> data   = DataLoader.from_fluxes("input/fluxes.csv")
>>> grid   = ModelGrid.from_file("data/cloudy_models.csv").refine_by_range(
...              range_u=(-4.0, -2.0), range_n=(1.0, 3.0),
...              range_z=(0.05, 2.0),  new_shape=(40, 40, 40, 9))
>>> result = BayesianFitter(data, grid).fit(galaxy_index=0)
>>> result.summary()
"""

from .io      import DataLoader     # noqa: F401
from .models  import ModelGrid      # noqa: F401
from .fitter  import BayesianFitter # noqa: F401
from .results import FitResult      # noqa: F401

__version__ = "0.1.0"
__author__  = "Skarleth Motiño Flores"
