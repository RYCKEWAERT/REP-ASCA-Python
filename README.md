
<a name="readme-top"></a>

![Project logo](images/repasca.png)

# REP-ASCA — Python implementation

Short description
-----------------
REP-ASCA (Reduction of Repeatability Error for Analysis of variance – Simultaneous Component Analysis) is a multivariate analysis method that extends ASCA to reduce the influence of repeatability (instrumental or sample-related) errors. This repository contains a Python implementation and an example notebook to demonstrate how to run REP-ASCA on spectral or multivariate datasets.

Why use REP-ASCA?
------------------
- Helps separate systematic factors of interest from repeatability noise.
- Useful for preprocessing spectral data (e.g., NIR) before chemometric modeling.

Quick start
-----------
1. Install dependencies (recommended inside a virtual environment):

   - Python 3.8+ (3.10 or later recommended)
   - numpy, scipy, matplotlib

2. Open the example notebook `main.ipynb` and run the cells. The notebook shows a minimal example that loads `data/data.mat`, computes REP components, and visualizes results.

Data format (what the notebook expects)
--------------------------------------
The example notebook expects a MATLAB `.mat` file at `data/data.mat` containing at least the following variables (names are case-sensitive):

- `X`: data matrix (observations × variables). This is the primary data used for ASCA/REP-ASCA.
- `d`: design matrix or factor information used by ASCA routines (format depends on the implementation in `librairies/repasca.py`).
- `X_rep`: replicate measurements (if available) used to estimate repeatability components.
- `d_rep`: design information for the replicate measurements.
- `lambda`: optional parameter(s) used by specific functions (check `librairies/repasca.py` for details).

The notebook prints variable shapes after loading the file, so you can confirm you provided the right variables.

Usage (notebook workflow)
------------------------
1. Set analysis parameters in the notebook (for example, `klimit`, the maximum number of REP components to inspect).
2. Run the REP-ASCA computation to obtain explained variances and error-related loadings.
3. Inspect the explained-variance plots to choose how many REP components to remove.
4. Apply the selected number of components to reduce residual repeatability and rerun ASCA/SCA.

References
----------
- Ryckewaert, M., Gorretta, N., Henriot, F., Marini, F., & Roger, J.-M. (2019). Reduction of repeatability error for Analysis of variance-Simultaneous Component Analysis (REP-ASCA): Application to NIR spectroscopy on coffee samples. Analytica Chimica Acta, 1101. https://doi.org/10.1016/j.aca.2019.12.024

Applications mentioned in published work
--------------------------------------
- Potential of high-spectral resolution for field phenotyping in plant breeding: application to maize under water stress — (Ryckewaert et al.)
- A generic workflow combining deep learning and chemometrics for processing close-range spectral images to detect drought stress in Arabidopsis thaliana — (Mishra et al.) https://doi.org/10.1016/j.chemolab.2021.104373

License
-------
This project is distributed under the terms of the included `LICENSE` file. See that file for details.

If you want, I can also rewrite the notebook text cells to clearer, step-by-step English (recommended).  
