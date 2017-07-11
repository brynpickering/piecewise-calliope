------------------------
Piecewise Calliope model
------------------------

Installion
----------

This study built from [Calliope 0.4.1-dev](https://github.com/calliope-project/calliope/commit/a4e49c3b7d37f908bafc84543510eec0b4cf5d9f) to create a piecewise model.

The easiest way to install the necessary packages is to create a conda environment: `conda create -n piecewise-calliope plotly`. Activate the environment and install the necessary development branch:

``pip install git+https://github.com/brynpickering/calliope.git@Piecewise_constrainedcurves``

The study utilised CPLEX 12.6.2.0 for MILP solving.

Running models
--------------

Once Calliope installed, models can be run and results can be analysed as per the [documentation](http://calliope.readthedocs.io/en/v0.4.1).

Model run configurations can be found in the `models` folder, where each run configuration has been named as: `run_[season]_[No. of breakpoints]bp_[linearisation method].yaml`

The models will save their solutions as netCDF to an Output folder. These solution files are required for ex-post analysis functions to operate correctly.

Piecewise linearisation
-----------------------
The functions `get_piecewise.py` and `minimize_linearisation_error.py` are used to find breakpoint positions along technology nonlinear characteristic curves. Breakpoint allocation is done equidistantly as well as optimised (to reduce error vs. the nonlinear curve). The result of `get_piecewise.py` is a YAML file that is used by this modified version of Calliope to create piecewise constraints.

Ex-post analysis
----------------

ex-post.py contains a function for analysis of the error incurred by linearisation of consumption curves. Each linear model run is compared to its nonlinear counterpart, by applying the relevant nonlinear consumption curves to the technology outputs obtained using the linear optimisation.
The output of this function is two xarray DataArrays, ``error`` and ``error_t``. The former provides per technology & location error in consumption, heat to power ratio (for CHP) and cost. The latter additionally provides consumption and HTP error for each timestep.

How to cite
-----------

If you use this model or work derived from it in an academic publication, please cite the following paper:

Pickering, B. and Choudhary, R.. "Applying Piecewise Linear Characteristic Curves in District Energy Optimisation" 30th INTERNATIONAL CONFERENCE on Efficiency, Cost, Optimisation, Simulation and Environmental Impact of Energy Systems. San Diego, USA. 2017.

License
-------

Copyright (c) 2016-2017 Bryn Pickering

[![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](http://creativecommons.org/licenses/by-sa/4.0/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
