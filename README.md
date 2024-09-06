# Are Deep Learning Methods Suitable for Downscaling Global Climate Projections? Review and Comparison of Existing Models.

This repository contains all the code needed to reproduce the generation of projections for the paper *Are Deep Learning Methods Suitable for Downscaling Global Climate Projections? Review and Comparison of Existing Models.*

The only dependencies for this repository are:

- xarray
- dask
- numba
- bottleneck
- netcdf4
- pytorch
- scipy

To further simplify the installation of the environment, we have included a [environment.yml](https://github.com/jgonzalezab/downscaling-intercomparison-spain/blob/main/requirements/environment.yml) file to easily create the Conda environment.

The [main.py](https://github.com/jgonzalezab/downscaling-intercomparison-spain/blob/main/main.py) script contains all the code necessary to train models and compute the projections. It is accompanied by detailed explanations to guide users through the script.
