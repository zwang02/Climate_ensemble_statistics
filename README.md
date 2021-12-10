# NCAR Internal Variability Package

The NCAR-iv package aims to provide statistical tools for identifying distributional difference in single climate runs compared to an initial state climate model ensemble. All the analysis are designed for annually, globally averaged datasets. The original model used is CESM1 and users need to adjust the variable list, simulation list before applying the code to their own model. The NCAR Tech Note documenting the statistical analysis carried out using this package is available [here](https://opensky.ucar.edu/islandora/object/technotes%3A591). 


This is an on-going project and is being actively maintained. 

## Current to-dos:
- Separate Installation/Data Loading script from the notebook.

- Create "library" that takes in files and outputs the necessary dataframe

Functions to write:
- lib.input
