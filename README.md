# AD_biomarkers_clustering
<div id="top"></div>

## Description

This is the code repository for the paper entitled [A data-driven approach to complement the A/T/(N) classification system using CSF biomarkers](). The repository follows the methodology and results presented in the abovementioned work. 

The Python scripts present in this repository are organized as follows:

* [prepare_datasets_HCSC.py](prepare_datasets_HCSC.py) - prepare data for HCSC dataset
* [prepare_datasets_ADNI.py](prepare_datasets_ADNI.py) - prepare data for ADNI dataset
* [kmeans_clustering.py](kmeans_clustering.py) - script for KMeans clustering using CSF biomarkers data fron the different data sources
* [clusters_description.py](clusters_description.py) - main functions to obtain several metrics from the obtained clusters

Moreover, there are several Python Jupyter Notebooks done specifically to some tasks:

* [datasets_description.ipynb](prepare_datasets_HCSC.ipynb) - dataset description statistics (number, sociodemo, MMSE, biomarkers values)
* [clustering_statistics.ipynb](prepare_datasets_ADNI.ipynb) - clusters description statistics (number, sociodemo, MMSE, biomarkers values, tests)
* [survival_analysis.ipynb](survival_analysis.ipynb) - survival analysis using Kaplan-Meier plots and Cox regression models

Other subdirectories present in this repository:

* [data](data) contains several data files used in this work. Please note that data files are not available in this repository due to privacy reasons.
* [results](results) SI scores os clustering results. Again, other results files are not available in this repository due to privacy reasons. 
* [figures](figures) figures obtained for the manuscript.

## Implementation

The code in this work was built using:

* [Scikit-Learn](https://scikit-learn.org/stable/) for building clustering models.
* [SciPy](https://scipy.org/) for statistical analyses.
* [lifelines](https://lifelines.readthedocs.io/en/latest/) for survival analyses.

## Contact
Please refer any questions to:
Laura Hernández-Lorenzo - [GitHub](https://github.com/laurahdezlorenzo) - [email](laurahl@ucm.es)
